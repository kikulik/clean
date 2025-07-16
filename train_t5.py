import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# --- CONFIGURATION ---
MODEL_ID = "t5-base"
DATASET_PATH = "training_dataset.json"
OUTPUT_DIR = "./results_qlora_advanced"
ADAPTERS_DIR = "./trained_lora_adapters_advanced"

def check_gpu():
    """Checks for GPU availability and prints device information."""
    if torch.cuda.is_available():
        print("✅ GPU is available.")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   Memory: {gpu_memory:.2f} GB")
        return True
    else:
        print("❌ GPU not available. Training will be extremely slow.")
        return False

def main():
    """Main function to run the advanced training and inference pipeline."""
    if not check_gpu():
        return

    # --- 1. Load the Dataset ---
    print(f"\n[1/6] Loading the dataset from '{DATASET_PATH}'...")
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        # The prefix is crucial for guiding the T5 model
        prefix = "generate json: "
        df['input_text'] = prefix + df['instruction'].astype(str)
        df['target_text'] = df['output'].apply(json.dumps)

        raw_dataset = Dataset.from_pandas(df[['input_text', 'target_text']])
        # Split the dataset: 90% for training, 10% for validation
        split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        print(f"✓ Dataset loaded and split: {len(train_dataset)} training examples, {len(eval_dataset)} evaluation examples.")
    except FileNotFoundError:
        print(f"\n--- 🔴 ERROR: '{DATASET_PATH}' not found! ---")
        print("--- Please run the data_prep.py script first to generate the dataset. ---")
        return

    # --- 2. Load Tokenizer and Quantized Model ---
    print(f"\n[2/6] Loading tokenizer and 4-bit quantized model for '{MODEL_ID}'...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID, legacy=False)

    # 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, # A more aggressive quantization
    )

    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", # Automatically maps model layers to available devices
    )
    model = prepare_model_for_kbit_training(model)
    print("✓ Model loaded.")

    # --- 3. Configure and Apply PEFT (LoRA) ---
    # Enhanced LoRA configuration for higher accuracy
    lora_config = LoraConfig(
        r=32,  # Increased rank for more expressive power
        lora_alpha=64, # Increased alpha to scale the learned weights
        target_modules=["q", "v", "k", "o"], # Target all attention projection layers
        lora_dropout=0.1, # Increased dropout for better regularization
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    print("✓ Applying LoRA adapters to the model...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. Tokenize the Dataset ---
    print("\n[4/6] Tokenizing the dataset...")
    def tokenize_function(examples):
        # Increased max_length for complex inputs
        model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length")
        # Increased max_length for potentially large JSON outputs
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target_text'], max_length=2048, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['input_text', 'target_text'])
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['input_text', 'target_text'])
    print("✓ Tokenization complete.")

    # --- 5. Set Up and Run the Trainer ---
    print("\n[5/6] Configuring and starting the training process...")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_dir,
        num_train_epochs=40, # Increased epochs as requested
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2, # Can be larger for evaluation
        gradient_accumulation_steps=8, # Effective batch size = 1 * 8 = 8
        learning_rate=1e-4, # A good learning rate for LoRA
        fp16=True, # Use mixed precision for speed
        logging_steps=50,
        save_total_limit=2, # Save only the best and the last checkpoint
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch", # Save at the end of each epoch
        load_best_model_at_end=True, # Load the best model based on eval loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit", # Memory-efficient optimizer
        lr_scheduler_type="cosine", # Cosine learning rate decay
        report_to="none", # Disable reporting to external services
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset, # Provide the evaluation set
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("🚀 Starting QLoRA training... This will take a while.")
    trainer.train()
    print("🎉 Training complete!")

    # --- 6. Save the Final Adapters ---
    print(f"\n[6/6] Saving the best LoRA adapters to '{ADAPTERS_DIR}'...")
    model.save_pretrained(ADAPTERS_DIR)
    tokenizer.save_pretrained(ADAPTERS_DIR)
    print(f"✓ Adapters saved successfully.")

    # --- Optional: Run Inference Example ---
    run_inference(tokenizer, instruction="generate json: Find all devices made by Sony.")

def run_inference(tokenizer, instruction):
    """Loads the fine-tuned model and runs a test inference."""
    print("\n--- Running Inference with Trained Adapters ---")
    
    # Load the base model and apply the trained adapters
    base_model = T5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    # Merge the LoRA adapters into the base model for faster inference
    model = PeftModel.from_pretrained(base_model, ADAPTERS_DIR)
    model = model.merge_and_unload()
    model.eval()

    print(f"Input: {instruction}")
    input_ids = tokenizer(instruction, return_tensors="pt").input_ids.to('cuda')

    outputs = model.generate(
        input_ids,
        max_length=2048,
        num_beams=5, # Use beam search for higher quality output
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n✅ Generated JSON Output:")
    try:
        # A more robust way to find and parse the JSON object
        json_start = generated_text.find('{')
        json_end = generated_text.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            json_str = generated_text[json_start:json_end]
            parsed_json = json.loads(json_str)
            print(json.dumps(parsed_json, indent=2))
        else:
            raise json.JSONDecodeError("No JSON object found in the output.", generated_text, 0)
    except json.JSONDecodeError as e:
        print(f"🔴 Could not decode the generated string into valid JSON. Error: {e}")
        print("   Raw output:", generated_text)

if __name__ == "__main__":
    main()
