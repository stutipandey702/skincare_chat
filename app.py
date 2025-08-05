import os
import tempfile
import time
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Create a proper cache directory
cache_dir = tempfile.mkdtemp(prefix="hf_cache_")
os.environ["HF_HOME"] = cache_dir
# Also set the old env var for compatibility
os.environ["TRANSFORMERS_CACHE"] = cache_dir

app = Flask(__name__)

def load_model_with_retry(model_id, model_type="tokenizer", max_retries=3):
    """Load model with retry mechanism for permission errors"""
    for attempt in range(max_retries):
        try:
            if model_type == "tokenizer":
                return AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    use_fast=True
                )
            elif model_type == "model":
                return AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    device_map="auto" if torch.cuda.is_available() else None
                )
        except OSError as e:
            if "PermissionError" in str(e) and attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying in 5 seconds...")
                time.sleep(5)
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