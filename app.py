from flask import Flask, request, render_template, redirect, url_for, session
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session

# Load fine-tuned GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("/home/pandey36/exploration_pp/skincare_chat/finetuned_model_m2_cpu")




@app.route("/", methods=["GET", "POST"])
def chat():
    response = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        prompt = f"User: {user_input}\nBot:"

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        response = full_output.split("Bot:")[-1].strip()
        return render_template("chat.html", response=response)  # No redirect

    # On GET (manual reload), clear previous response
    return render_template("chat.html", response=None)



if __name__ == "__main__":
    app.run(debug=True)