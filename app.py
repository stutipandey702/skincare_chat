from flask import Flask, request, render_template, redirect, url_for, session
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os
import re
import nltk
nltk.download('punkt_tab', download_dir='/home/pandey36/nltk_data')
nltk.data.path.append('/home/pandey36/nltk_data')
from nltk.tokenize import sent_tokenize


app = Flask(__name__)
# app.secret_key = os.urandom(24)  # Required for session
app.secret_key = 'supersecretkey123'

# Load fine-tuned GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("/home/pandey36/exploration_pp/skincare_chat/finetuned_model_m2_cpu")


def clean_response(raw_response):
    # Remove markdown-like headers
    text = re.sub(r'(#+\s*Answer:)', '', raw_response, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Capitalize start of sentence (if model fails)
    text = text[0].upper() + text[1:] if text else ''

    # Truncate to first 2 sentences
    sentences = sent_tokenize(text)
    return ' '.join(sentences[:2])


@app.route("/", methods=["GET", "POST"])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == "POST":
        user_input = request.form["user_input"]

        # Your prompt and model inference logic unchanged...
        prompt = (
            "You are a helpful skincare expert. "
            "Given an ingredient, respond in 1–2 clear, non-repetitive, helpful sentences "
            "explaining what it is and what it does for the skin.\n\n"
            f"User: {user_input}\nBot:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            temperature=0.8,
            top_p=0.9
        )
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        bot_response = full_output.split("Bot:")[-1].strip()
        clean_bot_response = clean_response(bot_response)

        # Append user message and bot response to chat history in session
        session['chat_history'].append({'from_user': True, 'text': user_input})
        session['chat_history'].append({'from_user': False, 'text': clean_bot_response})
        session.modified = True  # mark session modified to save changes

        # Render template with chat_history and no separate `response`
        return render_template("chat.html", chat_history=session['chat_history'])

    # GET request — show page with empty or existing chat_history
    return render_template("chat.html", chat_history=session.get('chat_history', []))


if __name__ == "__main__":
    app.run(debug=True)