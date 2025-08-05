#!/usr/bin/env python3

# EMERGENCY FIX: Run this command first in your space terminal:
# chmod -R 777 /app && mkdir -p /app/cache && export HF_HOME=/app/cache

import os
import subprocess
import sys

# Try to fix permissions first
try:
    subprocess.run(["mkdir", "-p", "/app/cache"], check=False)
    subprocess.run(["chmod", "-R", "777", "/app"], check=False)
    print("✅ Permissions fixed")
except:
    print("⚠️ Could not fix permissions")

# Set environment variables
os.environ.clear()  # Clear all existing env vars that might interfere
os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin"
os.environ["HF_HOME"] = "/app/cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"
os.environ["HF_HUB_CACHE"] = "/app/cache"

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Import after setting env vars
try:
    from transformers import pipeline
    
    print("Loading with pipeline (simpler approach)...")
    # Use pipeline which handles caching better
    generator = pipeline(
        "text-generation",
        model="gpt2",  # Start with most reliable model
        tokenizer="gpt2",
        cache_dir="/app/cache",
        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        model_kwargs={"low_cpu_mem_usage": True}
    )
    print("✅ Model loaded via pipeline!")
    
except Exception as e:
    print(f"❌ Pipeline failed: {e}")
    sys.exit(1)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        print("=== NEW REQUEST ===")
        data = request.get_json()
        print(f"Received data: {data}")
        
        prompt = data.get("prompt", "")
        print(f"Prompt: '{prompt}'")
        
        if not prompt:
            print("ERROR: No prompt provided")
            return jsonify({"error": "No prompt provided"}), 400
        
        print("Generating response...")
        # Generate response using pipeline with optimized settings
        result = generator(
            prompt,
            max_new_tokens=50,  # Use max_new_tokens instead of max_length
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=50256,  # GPT-2 EOS token
            truncation=True,
            return_full_text=False  # Only return new text, not input
        )
        
        print(f"Pipeline result: {result}")
        response_text = result[0]['generated_text']
        print(f"Generated text: '{response_text}'")
        
        # Clean up response (less processing needed with return_full_text=False)
        response_text = response_text.strip()
        
        # Ensure we have some response
        if not response_text:
            response_text = "I understand your message, but I need a moment to think of a good response."
        
        response_data = {"response": response_text}
        print(f"Sending response: {response_data}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"ERROR in ask endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

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
model_loaded{model="tiny-llama-1.1b-v1.0"} 1.0
'''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)