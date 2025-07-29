from flask import Flask, request, render_template, session
import os
import requests
import re


app = Flask(__name__)
app.secret_key = 'supersecretkey123'

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
ENDPOINT_URL = os.getenv("ENDPOINT_URL")

def clean_response(raw_response):
    text = re.sub(r'(#+\s*Answer:)', '', raw_response, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = sent_tokenize(text)
    return ' '.join(sentences[:2])

def query_model(prompt):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.7}
    }
    response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return clean_response(response.json()['generated_text'])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return "Error querying model."

@app.route("/", methods=["GET", "POST"])
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == "POST":
        user_input = request.form["user_input"]
        bot_response = query_model(user_input)
        session['chat_history'].append({'from_user': True, 'text': user_input})
        session['chat_history'].append({'from_user': False, 'text': bot_response})
        session.modified = True

    return render_template("chat.html", chat_history=session.get('chat_history', []))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
