import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel

import time

# === CONFIG ===
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
lora_model = "./skinc_llm_tinyllama"  # path to fine-tuned adapter

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32, 
    bnb_4bit_quant_type="nf4"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD MODEL ===
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Merging LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_model)
model.eval()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# === TEST PROMPT ===
prompt = "### Question: What are the benefits of using mango seed butter on skin?\n### Answer:"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# === GENERATION CONFIG ===
generation_config = GenerationConfig(
    max_new_tokens=60,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id
)

# === INFERENCE ===
print("Generating response...")
start = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        generation_config=generation_config
    )
end = time.time()

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- RESPONSE ---\n")
print(response)
print(f"\n⏱️ Inference time: {end - start:.2f} seconds")
