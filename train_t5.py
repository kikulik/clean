# train_t5.py
import json
import torch
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "t5-base"
DATASET_PATH = "training_dataset.json"
OUTPUT_DIR = "t5-base-finetuned-broadcast-schema"


def load_data(file_path: str) -> Dataset:
    """Loads the JSON data and formats it for T5."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # T5 requires a prefix for text-to-text tasks. We'll use a descriptive one.
    prefix = "translate English to JSON: "

    inputs = [prefix + item['instruction'] for item in data]
    # The target needs to be a string, so we dump the JSON object.
    targets = [json.dumps(item['output']) for item in data]

    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict({"input_text": inputs, "target_text": targets})
    return dataset


def preprocess_function(examples, tokenizer, max_length=512):
    """Tokenizes the input and target texts."""
    # Tokenize inputs
    model_inputs = tokenizer(
        examples['input_text'], max_length=max_length, padding="max_length", truncation=True
    )

    # Tokenize targets (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['target_text'], max_length=max_length, padding="max_length", truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    # 1. Load Tokenizer and Model
    print(f"Loading tokenizer and model for '{MODEL_NAME}'...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # 2. Load and Preprocess Dataset
    print(f"Loading and preprocessing data from '{DATASET_PATH}'...")
    dataset = load_data(DATASET_PATH)

    # Split data into training and validation sets (90% train, 10% eval)
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    tokenized_train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )
    print("Data tokenization complete.")

    # 3. Define Training Arguments
    # These are crucial settings for your training run.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,                     # Total number of training epochs
        per_device_train_batch_size=4,          # Batch size per GPU for training
        per_device_eval_batch_size=4,           # Batch size per GPU for evaluation
        # Number of warmup steps for learning rate scheduler
        warmup_steps=500,
        weight_decay=0.01,                      # Strength of weight decay
        logging_dir='./logs',                   # Directory for storing logs
        logging_steps=100,                      # Log every X updates steps
        evaluation_strategy="steps",            # Evaluate during training
        eval_steps=500,                         # Evaluate every X steps
        save_strategy="steps",                  # Save checkpoints during training
        save_steps=500,                         # Save a checkpoint every X steps
        # Load the best model at the end of training
        load_best_model_at_end=True,
        save_total_limit=2,                     # Only keep the last 2 checkpoints
        # Use mixed precision training (requires NVIDIA Ampere or newer for best performance)
        fp16=True,
        report_to="none",                       # Disable reporting to services like W&B
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )

    # 5. Start Training
    print("Starting model training... 🚀")
    trainer.train()
    print("Training finished successfully! 🎉")

    # 6. Save the Final Model and Tokenizer
    print(f"Saving final model to '{OUTPUT_DIR}'...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved.")


if __name__ == "__main__":
    main()
