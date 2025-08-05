import os
import tempfile
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Force HF to use a writable directory
os.environ["HF_HOME"] = "/app"
os.environ["TRANSFORMERS_CACHE"] = "/app"
os.environ["HF_HUB_CACHE"] = "/app"

app = Flask(__name__)

def download_model_to_app_dir():
    """Download model directly to /app directory"""
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_model_id = "stutipandey/llama_skinchat_lora"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        cache_dir="/app/tokenizer_cache",
        local_files_only=False,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir="/app/model_cache",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=False
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_model_id)
    model.eval()
    
    return tokenizer, model

# Try to load models
try:
    tokenizer, model = download_model_to_app_dir()
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # Emergency fallback - use a tiny model that's more reliable
    print("🔄 Trying fallback model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="/app")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir="/app")
        print("✅ Fallback model loaded!")
    except Exception as fallback_error:
        print(f"❌ Even fallback failed: {fallback_error}")
        raise fallback_error

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
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from response
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):].strip()
        
        return jsonify({"response": response_text})
        
    except Exception as e:
        return jsonify({"error": f"Generation error: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)