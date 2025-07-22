import json
import re
import random
from pathlib import Path

# Input and output paths
input_path = Path("/home/pandey36/exploration_pp/skincare_chat/finetune_ingredient_dataset_with_targets.jsonl")
output_path = Path("optimized_finetune_dataset.jsonl")

# Question templates to vary how the ingredient is asked about
question_templates = [
    "What is {}?",
    "How does {} benefit the skin?",
    "Can you explain what {} is used for?",
    "What does {} do in skincare?",
    "Tell me about {}.",
]

# Response templates that expect a cleaned description (without ingredient name inside)
response_templates = [
    "{} is used in skincare for its {}.",
    "It helps the skin by {}.",
    "Known for its {}, {} is widely used in cosmetics.",
    "You’ll find {} in products because it {}.",
    "{} supports the skin through {}.",
]

def extract_ingredient(prompt):
    """
    Extract ingredient name from prompt like "What is Mango Seed Butter and what is it good for?"
    """
    match = re.search(r"What is (.+?) and what is it good for\?", prompt)
    if match:
        return match.group(1).strip()
    return prompt.strip()

def clean_response(text, ingredient=None):
    """
    Clean raw response text by:
    - Removing boilerplate phrases
    - Removing ingredient name mentions (case-insensitive)
    - Removing extra whitespace
    - Splitting into meaningful sentences containing keywords
    - Returning up to 2 sentences joined by space
    """
    # Remove boilerplate
    text = re.sub(r"It is especially good for:.*", "", text, flags=re.DOTALL)
    text = re.sub(r"It should be avoided by those with:.*", "", text, flags=re.DOTALL)

    # Normalize whitespace
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Remove ingredient mentions
    if ingredient:
        # Escape special regex chars in ingredient name
        pattern = re.compile(re.escape(ingredient), re.IGNORECASE)
        text = pattern.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Keywords to keep sentences relevant
    keywords = r'\b(is|are|was|has|helps|used|can|does|prevents|supports|benefits|known|offers|found|helps|protects|hydrates)\b'

    meaningful = []
    for s in sentences:
        s = s.strip()
        if len(s) < 10:
            continue
        if not re.search(keywords, s, re.IGNORECASE):
            continue
        meaningful.append(s)

    # Remove duplicates
    seen = set()
    unique_sentences = []
    for s in meaningful:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique_sentences.append(s)

    # Join up to 2 sentences
    response_text = " ".join(unique_sentences[:2])

    # Extra safeguard: skip if cleaned response is empty or too short
    if len(response_text) < 10:
        return ""

    return response_text

def main():
    optimized_entries = []

    with input_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            entry = json.loads(line)
            raw_prompt = entry.get("prompt", "")
            raw_response = entry.get("response", "")

            if not raw_prompt or not raw_response:
                continue

            ingredient = extract_ingredient(raw_prompt)
            clean_resp = clean_response(raw_response, ingredient=ingredient)

            # Skip if cleaning failed or response is meaningless
            if not ingredient or not clean_resp:
                continue

            # Use response templates that insert ingredient only once
            question = random.choice(question_templates).format(ingredient)
            answer = random.choice(response_templates).format(ingredient, clean_resp.lower()).replace("..", ".").replace(" - ", "")

            optimized_entries.append({
                "prompt": f"### Question: {question}\n### Answer:",
                "response": answer
            })

    # Limit dataset size for CPU-friendly finetuning
    max_entries = 300
    random.shuffle(optimized_entries)
    optimized_entries = optimized_entries[:max_entries]

    # Save to JSONL with actual UTF-8 characters
    with output_path.open("w", encoding="utf-8") as outfile:
        for entry in optimized_entries:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")


    print(f"Optimized and saved {len(optimized_entries)} entries to {output_path}")

if __name__ == "__main__":
    main()
