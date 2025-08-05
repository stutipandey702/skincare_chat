import os
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Fix cache issues
os.environ["HF_HOME"] = "/tmp/hf_cache_lora"
os.makedirs("/tmp/hf_cache_lora", exist_ok=True)

app = Flask(__name__)

# Load base + LoRA
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # the base fine-tuned
lora_model_id = "stutipandey/llama_skinchat_lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model = PeftModel.from_pretrained(base_model, lora_model_id)
model.eval()

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
