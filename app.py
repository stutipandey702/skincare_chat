import os
import sys

# CRITICAL: Set ALL cache environment variables BEFORE importing transformers
os.environ["HF_HOME"] = "/app/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/hf_cache"
os.environ["HF_HUB_CACHE"] = "/app/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/app/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/app/hf_cache"

# Create cache directory if it doesn't exist
cache_dir = "/app/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

# Try to set permissions (ignore errors in case of restrictions)
try:
    os.chmod(cache_dir, 0o777)
    print(f"✅ Cache directory created at: {cache_dir}")
except:
    print(f"⚠️ Could not set permissions on: {cache_dir}")

from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app = Flask(__name__)

def load_models_with_custom_cache():
    """Load models with explicit cache directory"""
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_model_id = "stutipandey/llama_skinchat_lora"
    
    print("Loading tokenizer with custom cache...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        use_fast=False  # Use slow tokenizer to avoid additional cache issues
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Loading base model with custom cache...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_model_id)
    model.eval()
    
    return tokenizer, model

def load_fallback_model():
    """Fallback to a simpler model if TinyLlama fails"""
    print("🔄 Loading fallback model (GPT-2)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        cache_dir=cache_dir
    )
    
    return tokenizer, model

# Try to load models
print("🚀 Starting model loading...")
try:
    tokenizer, model = load_models_with_custom_cache()
    model_type = "TinyLlama + LoRA"
    print("✅ TinyLlama + LoRA loaded successfully!")
except Exception as e:
    print(f"❌ TinyLlama failed: {e}")
    try:
        tokenizer, model = load_fallback_model()
        model_type = "GPT-2 Fallback"
        print("✅ GPT-2 fallback loaded successfully!")
    except Exception as fallback_error:
        print(f"❌ Complete failure: {fallback_error}")
        sys.exit(1)

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
        
        # Format prompt based on model type
        if "TinyLlama" in model_type:
            formatted_prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        else:
            formatted_prompt = prompt
        
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response
        if "TinyLlama" in model_type and "<|assistant|>" in response_text:
            response_text = response_text.split("<|assistant|>")[-1].strip()
        elif response_text.startswith(formatted_prompt):
            response_text = response_text[len(formatted_prompt):].strip()
        
        return jsonify({
            "response": response_text,
            "model_used": model_type
        })
        
    except Exception as e:
        print(f"Error in generation: {e}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": model_type,
        "cache_dir": cache_dir
    })

@app.route("/debug", methods=["GET", "POST"])
def debug():
    if request.method == "GET":
        return jsonify({
            "status": "Server is running",
            "model_loaded": "generator" in globals(),
            "cache_dir": "/app/cache"
        })
    else:
        # Test POST
        return jsonify({"response": "Debug POST works!"})

@app.route("/test")
def test():
    try:
        # Test the generator
        result = generator("Hello", max_length=20, num_return_sequences=1)
        return jsonify({
            "test_result": result[0]['generated_text'],
            "status": "Model is working"
        })
    except Exception as e:
        return jsonify({"error": f"Model test failed: {e}"})

@app.route("/metrics")
def metrics():
    """Simple metrics endpoint for HF Spaces monitoring"""
    return '''# HELP python_info Python platform information.
# TYPE python_info gauge
python_info{implementation="CPython",major="3",minor="9"} 1.0
# HELP app_status Application status
# TYPE app_status gauge
app_status{status="healthy"} 1.0
# HELP model_loaded Model loading status
# TYPE model_loaded gauge
model_loaded{model="gpt2"} 1.0
'''

if __name__ == "__main__":
    print(f"🎉 Server starting with {model_type}")
    app.run(host="0.0.0.0", port=7860, debug=False)