import os
import shutil
import time
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Remove any existing cache and environment variables
if "HF_HOME" in os.environ:
    del os.environ["HF_HOME"]
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]

# Clean up any existing cache
cache_paths = ["/tmp/hf_cache_lora", "/root/.cache/huggingface", "/home/.cache/huggingface"]
for path in cache_paths:
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except:
            pass

app = Flask(__name__)

def load_model_with_retry(model_id, model_type="tokenizer", max_retries=3):
    """Load model with retry mechanism and force download"""
    for attempt in range(max_retries):
        try:
            if model_type == "tokenizer":
                return AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    use_fast=True,
                    force_download=True,  # Force fresh download
                    resume_download=False,  # Don't resume partial downloads
                    local_files_only=False  # Always try to download
                )
            elif model_type == "model":
                return AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                    force_download=True,  # Force fresh download
                    resume_download=False,  # Don't resume partial downloads
                    local_files_only=False  # Always try to download
                )
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in 10 seconds...")
                time.sleep(10)
                continue
            else:
                raise e

# Initialize models
print("Loading models...")
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_model_id = "stutipandey/llama_skinchat_lora"

try:
    # Load tokenizer
    tokenizer = load_model_with_retry(base_model_id, "tokenizer")
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = load_model_with_retry(base_model_id, "model")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_model_id)
    model.eval()
    
    print("Models loaded successfully!")
    
except Exception as e:
    print(f"Error loading models: {e}")
    # You might want to exit or use a fallback model here
    raise e

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Tokenize input
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):].strip()
        
        return jsonify({"response": response_text})
        
    except Exception as e:
        print(f"Error in ask endpoint: {e}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)