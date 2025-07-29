from flask import Flask, request, render_template, redirect, url_for, session
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
import torch
import os
from dotenv import load_dotenv
import requests
import re
import nltk
nltk.download('punkt', download_dir='/home/pandey36/nltk_data')
nltk.data.path.append('/home/pandey36/nltk_data')
from nltk.tokenize import sent_tokenize




load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
ENDPOINT_URL = os.getenv("ENDPOINT_URL")

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

# Make API request and store response
def query_model(prompt): # note that this charges the account on HF
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7
        }
    }

    response = requests.post(ENDPOINT_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        print(f"Error {response.status_code}: {response.text}")
        return "Something went wrong!"


# Save chat history
chat_history = []

# Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    output = ""
    if request.method == "POST":
        prompt = request.form["user_input"]
        response_text = query_model(prompt)
        chat_history.append({"from_user": True, "text": prompt})
        chat_history.append({"from_user": False, "text": response_text})

    return render_template("chat.html", chat_history=chat_history)


# Run the app
if __name__ == "__main__":
    app.run(debug=True)