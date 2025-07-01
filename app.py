from flask import Flask, request, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load TinyLlama base model + tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/", methods=["GET", "POST"]) # possible requests
def chat():
    response = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100)
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_output.split("### Response:")[-1].strip()
    return render_template("chat.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
