from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer directly from your model repo
MODEL_NAME = "your-username/your-model-repo"  # replace with your actual model repo name

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.eval()


@app.route("/")
def home():
    return render_template("index.html")


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
