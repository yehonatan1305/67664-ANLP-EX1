from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
import evaluate
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import wandb
from transformers import Trainer, TrainingArguments, HfArgumentParser
import time

# Load metric once at module level
metric = evaluate.load("glue", "mrpc")

DATASETNAME = "mrpc"
MODELNAME = "google-bert/bert-base-uncased"
SEED = 0


@dataclass
class ScriptArguments:
    max_train_samples: int = field(default=12)
    max_eval_samples: int = field(default=12)
    max_predict_samples: int = field(default=3)
    num_train_epochs: int = field(default=1)
    lr: float = field(default=5e-5)
    batch_size: int = field(default=1)
    do_train: bool = field(default=False)
    do_predict: bool = field(default=True)
    model_path: str = field(default="./saved_models/final_model")

def get_args():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def init_wandb(script_args, entity="yehonata-hirshcel-hebrew-university-of-jerusalem", project="Test-Project"):
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    run = wandb.init(
        entity=entity,
        project=project,
        name=f"lr{script_args.lr}-e{script_args.num_train_epochs}-b{script_args.batch_size}-{cur_time}",
        group="experiment_runs",
        job_type="training" if script_args.do_train else "prediction",
        config={
            "learning_rate": script_args.lr,
            "model_name": MODELNAME,
            "architecture": "BERT",
            "dataset": DATASETNAME,
            "epochs": script_args.num_train_epochs,
            "batch_size": script_args.batch_size,
            "max_train_samples": script_args.max_train_samples,
            "max_eval_samples": script_args.max_eval_samples,
            "max_predict_samples": script_args.max_predict_samples,
        },
    )
    return run

def pre_process_function(examples):
    # Tokenize the input text with truncation to max length and dynamic padding
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        padding=False,
        max_length=tokenizer.model_max_length
    )

def init_trainer(model, args, train_dataset=None, eval_dataset=None, compute_metrics=None, data_collator=None):
    # Initialize wandb
    print("Initializing wandb")
    run = init_wandb(args)
    training_args = TrainingArguments(
        output_dir=f"./saved_models/{run.name}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        do_eval=True if eval_dataset else False,
        logging_dir="./logs",
        logging_steps=1,  # Log every step
        eval_strategy="steps" if eval_dataset else "no",  # Only evaluate if we have eval data
        eval_steps=1 if eval_dataset else None,  # Only set eval steps if we have eval data
        report_to="wandb",
        save_strategy="epoch",  # Save model at the end of each epoch
        save_total_limit=1,  # Keep only the latest model
        run_name=run.name,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    return trainer

def train_model(args):
    print("Loading model")
    config = AutoConfig.from_pretrained(MODELNAME, num_labels=2)  # MRPC is binary classification
    model = AutoModelForSequenceClassification.from_pretrained(MODELNAME, config=config)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print("Initializing trainer")
    trainer = init_trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    # Train the model
    print("Starting training")
    trainer.train()
    print("Training complete.")
    # Save the model after training
    trainer.save_model(f"./saved_models/lr{args.lr}-e{args.num_train_epochs}-b{args.batch_size}")
    #report to res.txt the validation accuracy
    eval_results = trainer.evaluate()
    with open("res.txt", "a") as f:
        f.write(f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {eval_results['eval_accuracy']:.4f}\n")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def predict_model(args):
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer for prediction
    trainer = init_trainer(
        model=model,
        args=args,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    trainer.model.eval()  # Set the model to evaluation mode
    
    # Run prediction
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(-1)
    
    # Get raw sentences from test dataset
    test_samples = ds["test"]
    if args.max_predict_samples > -1:
        test_samples = test_samples.select(range(args.max_predict_samples))
    
    # Write predictions to file
    with open("predictions.txt", "w") as f:
        for idx, pred_label in enumerate(predicted_labels):
            sentence1 = test_samples[idx]["sentence1"]
            sentence2 = test_samples[idx]["sentence2"]
            f.write(f"{sentence1}###{sentence2}###{pred_label}\n")
    
    # Print metrics
    print(predictions.metrics)


if __name__ == "__main__":
    print("Loading args")
    args = get_args()
    print("Loading dataset")
    ds = load_dataset("nyu-mll/glue", DATASETNAME)
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODELNAME)
    print("splitting dataset")
    train_dataset = ds["train"].map(pre_process_function, batched=True)
    if args.max_train_samples > -1:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    eval_dataset = ds["validation"].map(pre_process_function, batched=True)
    if args.max_eval_samples > -1:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    test_dataset = ds["test"].map(pre_process_function, batched=True)
    if args.max_predict_samples > -1:
        test_dataset = test_dataset.select(range(args.max_predict_samples))
    if args.do_predict:
        predict_model(args)
    if args.do_train:
        train_model(args)
