"""
run_eval.py — Skincare Chatbot Evaluation Script

Runs all questions in test_set.json through your chatbot, scores each
response using Claude as a judge, and prints a final score.

Usage:
    python run_eval.py

Requirements:
    pip install anthropic requests
    export ANTHROPIC_API_KEY=your_key_here
"""
import sys
import json
import time
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from build_rag import build_index
from rag import load_index, retrieve

# runs once at startup
collection, embedder = build_index()
load_index(collection, embedder)

import anthropic

# ── CONFIG ────────────────────────────────────────────────────────────────────

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEST_SET_PATH    = Path(__file__).parent / "test_set.json"
CHATBOT_ENDPOINT = "http://localhost:5000/chat"  # adjust to your Flask route

from transformers import pipeline

PIPE = pipeline("text-generation", model="./llama_skinchat_lora")

# ── CHATBOT CALLER ────────────────────────────────────────────────────────────

def call_chatbot(question: str) -> str:
    
    context = retrieve(question)
    # print(f"  [CONTEXT] {context}")  # add this
    # print("=== CONTEXT ===")
    # print(context[:500])
    # print("================\n")
    prompt = f"""You are a skincare assistant. Use the following information to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    result = PIPE(prompt, max_new_tokens=200)[0]["generated_text"]
    return result.split("Answer:")[-1].strip()

# ── LLM JUDGE ─────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator for a skincare chatbot. You will be given:
- A user question
- A reference answer (key points a good answer should cover)
- The chatbot's actual response

Score the response on three dimensions, each 0-2:

1. Relevance: Did it address the question? (0=off-topic, 1=partial, 2=fully)
2. Accuracy: Is the information correct? (0=errors, 1=mostly correct, 2=accurate)
3. Specificity: Is it concrete and detailed? (0=vague, 1=somewhat, 2=specific)

Return ONLY valid JSON, nothing else:
{
  "relevance": <0|1|2>,
  "accuracy": <0|1|2>,
  "specificity": <0|1|2>,
  "total": <sum>,
  "feedback": "<one sentence on main strength or weakness>"
}
""".strip()

def judge_response(client, question, reference, response) -> dict:
    user_message = f"""
Question: {question}

Reference answer:
{reference}

Chatbot response:
{response}
""".strip()

    try:
        result = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        return json.loads(result.content[0].text.strip())
    except Exception as e:
        print(f"  [ERROR] Judge failed: {e}")
        return {"relevance": 0, "accuracy": 0, "specificity": 0, "total": 0, "feedback": str(e)}

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    with open(TEST_SET_PATH) as f:
        test_set = json.load(f)

    client = anthropic.Anthropic()
    results = []

    print(f"Running eval on {len(test_set)} questions...\n")

    for i, item in enumerate(test_set, 1):
        print(f"[{i:02d}/{len(test_set)}] {item['question'][:70]}...")

        response = call_chatbot(item["question"])
        if not response:
            print("  [SKIP] Empty response\n")
            continue

        scores = judge_response(client, item["question"], item["reference_answer"], response)
        print(f"  {scores['total']}/6 — {scores['feedback']}\n")

        results.append({"category": item["category"], "scores": scores})
        time.sleep(0.5)

    # ── SCORE SUMMARY ─────────────────────────────────────────────────────────

    total     = sum(r["scores"]["total"] for r in results)
    max_score = len(results) * 6
    pct       = round(total / max_score * 100, 1)

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "count": 0}
        categories[cat]["total"] += r["scores"]["total"]
        categories[cat]["count"] += 1

    print("=" * 50)
    print(f"  OVERALL SCORE: {total}/{max_score}  ({pct}%)")
    print("=" * 50)
    for cat, data in sorted(categories.items()):
        cat_max = data["count"] * 6
        cat_pct = round(data["total"] / cat_max * 100, 1)
        print(f"  {cat:<30} {data['total']:>3}/{cat_max}  ({cat_pct}%)")
    print("=" * 50)

if __name__ == "__main__":
    main()