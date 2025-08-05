import os
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# Set cache directory
os.environ["HF_HOME"] = "/app/cache"
os.makedirs("/app/cache", exist_ok=True)

app = Flask(__name__)

# Load model
print("Loading GPT-2 model...")
try:
    generator = pipeline(
        "text-generation",
        model="gpt2",
        cache_dir="/app/cache"
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
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
        
        # Generate response
        result = generator(
            prompt,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=1,
            return_full_text=False
        )
        
        response_text = result[0]['generated_text'].strip()
        
        if not response_text:
            response_text = "I understand your question. Could you please rephrase it?"
        
        return jsonify({"response": response_text})
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/metrics")
def metrics():
    return '''# HELP app_status Application status
# TYPE app_status gauge
app_status{status="healthy"} 1.0
'''

@app.route("/health")
def health():
    return jsonify({"status":