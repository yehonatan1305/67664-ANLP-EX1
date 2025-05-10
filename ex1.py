print("Starting script")
from dataclasses import dataclass, field
from datasets import load_dataset
print("Importing evaluate")
import evaluate
print("Done importing evaluate")
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
print("Import wandb")
import wandb
print("Done importing wandb")
from transformers import Trainer, TrainingArguments, HfArgumentParser
import time
print("Imports done")

ISJUPYTER = False
PATH = "/content/drive/MyDrive/ANLP/EX1" if ISJUPYTER else "."

metric = evaluate.load("glue", "mrpc")

DATASETNAME = "mrpc"
MODELNAME = "google-bert/bert-base-uncased"
SEED = 0
run_name_format = lambda args, cur_time: f"epoch_num_{args.num_train_epochs}_lr_{args.lr}_batch_size_{args.batch_size}_{int(cur_time)}"

@dataclass
class ScriptArguments:
    max_train_samples: int = field(default=-1)
    max_eval_samples: int = field(default=-1)
    max_predict_samples: int = field(default=-1)
    num_train_epochs: int = field(default=3)
    lr: float = field(default=5e-4)
    batch_size: int = field(default=64)
    do_train: bool = field(default=False)
    do_predict: bool = field(default=True)
    model_path: str = field(default=f"{PATH}/saved_models/epoch_num_2_lr_5e-05_batch_size_32_1746905852")

def get_args():
    if ISJUPYTER:
        return ScriptArguments()
    else:
        parser = HfArgumentParser(ScriptArguments)
        return parser.parse_args_into_dataclasses()[0]


def init_wandb(script_args,cur_time,entity="yehonata-hirshcel-hebrew-university-of-jerusalem", project="ANLP-EX1", ):
    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name_format(script_args, cur_time),
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

def init_trainer(model, args, cur_time, train_dataset=None, eval_dataset=None, compute_metrics=None, data_collator=None):
    # Initialize wandb
    print("Initializing wandb")
    run = init_wandb(args, cur_time)
    training_args = TrainingArguments(
        output_dir=f"{PATH}/saved_models/{run.name}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        do_eval=True if eval_dataset else False,
        logging_dir=f"{PATH}/logs",
        logging_steps=1,  # Log every step
        eval_strategy="steps" if eval_dataset else "no",  # Only evaluate if we have eval data
        eval_steps=1 if eval_dataset else None,  # Only set eval steps if we have eval data
        report_to="wandb",
        save_strategy="no",  # Do not save the model during training
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
    cur_time = time.time()
    trainer = init_trainer(
        model=model,
        args=args,
        cur_time=cur_time,
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
    trainer.save_model(f"{PATH}/saved_models/{run_name_format(args,cur_time)}")
    #report to res.txt the validation accuracy
    eval_results = trainer.evaluate()
    with open(f"{PATH}/res.txt", "a") as f:
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
        cur_time=time.time(),
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
    with open(f"{PATH}/predictions.txt", "w") as f:
        for idx, pred_label in enumerate(predicted_labels):
            sentence1 = test_samples[idx]["sentence1"]
            sentence2 = test_samples[idx]["sentence2"]
            f.write(f"{sentence1}###{sentence2}###{pred_label}\n")

    # Print metrics
    print(predictions.metrics)

if __name__ == "__main__":
    print("Loading args")
    # Set parse_args=True when running from command line, False when running in Jupyter
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
