# Advanced NLP Exercise 1: Fine Tuning

This is the code base for ANLP HUJI course exercise 1,fine tuning pretrained models for the Microsoft Research Paraphrase Corpus (MRPC) task from the GLUE benchmark.

## Installation
```bash
pip install -r requirements.txt
```

## Model Details
The project uses the `google-bert/bert-base-uncased` model for fine-tuning on the MRPC task. MRPC is a binary classification task where the goal is to predict whether two sentences are paraphrases of each other.

## Usage
Run the script with:
```bash
python ex1.py [OPTIONS]
```

### Required Arguments
- `--do_train`: Flag to run training
- `--do_predict`: Flag to run prediction on test set

### Optional Arguments
- `--max_train_samples`: Number of training samples to use (-1 for all)
- `--max_eval_samples`: Number of validation samples to use (-1 for all)
- `--max_predict_samples`: Number of test samples to use (-1 for all)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--lr`: Learning rate (default: 5e-4)
- `--batch_size`: Batch size for training and evaluation (default: 64)
- `--model_path`: Path to the model for prediction (required when using --do_predict)

## Output Files
- `predictions.txt`: Generated when using `--do_predict`, contains predictions for test samples in format: `sentence1###sentence2###prediction`
- `res.txt`: Contains evaluation results during training in format: `epoch_num: {epochs}, lr: {learning_rate}, batch_size: {batch_size}, eval_acc: {accuracy}`

## Example Usage
To train a model:
```bash
python ex1.py --do_train --num_train_epochs 3 --lr 5e-4 --batch_size 64 --max_train_samples 1000
```

To predict using a trained model:
```bash
python ex1.py --do_predict --model_path saved_models/your_model_directory
