from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

# Load your JSONL dataset with prompt/response pairs
dataset = load_dataset("json", data_files="optimized_finetune_dataset.jsonl", split="train")

# Format text combining prompt and response (adjust format as needed)
def format_prompt(example):
    example["text"] = f"### Question: {example['prompt']}\n### Answer: {example['response']}"
    return example

dataset = dataset.map(format_prompt)

# Load tokenizer and model (TinyLlama or GPT2, here example GPT2)
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # For GPT-2 padding

model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu")

# Tokenize inputs with truncation and padding (shorter max_length for faster training)
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=200)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response", "text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args optimized for CPU and 8 cores
training_args = TrainingArguments(
    output_dir="./finetuned_model_m2_cpu",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,           # keep small for CPU RAM
    gradient_accumulation_steps=1,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    dataloader_num_workers=8,                 # use all 8 CPU cores for data loading
    dataloader_pin_memory=False,              # no pinned memory on CPU
    report_to="none",
    learning_rate=2e-4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Save your fine-tuned model and tokenizer
model.save_pretrained("./finetuned_model_m2_cpu")
tokenizer.save_pretrained("./finetuned_model_m2_cpu")
