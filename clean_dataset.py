# Re-import necessary modules after code execution state reset
from pathlib import Path
import json
import random

# Reload dataset
dataset_path = Path("finetune_ingredient_dataset_with_targets.jsonl")
samples = []
with dataset_path.open("r") as f:
    for line in f:
        entry = json.loads(line)
        samples.append(entry)

# Clean and shorten
placeholders = [
    "It is especially good for: .",
    "It should be avoided by those with: ."
]

question_templates = [
    "What are the benefits of {}?",
    "How does {} help the skin?",
    "When should you use {}?",
    "Why is {} good for skincare?",
    "Can {} help with dry or sensitive skin?",
    "What is {} used for?",
    "Is {} suitable for all skin types?",
    "What does {} do?",
]

# Clean and reformat
cleaned_dataset = []
for entry in samples:
    ingredient = entry['prompt'].replace("What is ", "").replace(" and what is it good for?", "").strip()
    response = entry['response']
    for phrase in placeholders:
        response = response.replace(phrase, "").strip()
    for _ in range(random.randint(1, 2)):  # fewer examples now
        question = random.choice(question_templates).format(ingredient)
        prompt = f"### Question: {question}\n### Answer:"
        cleaned_dataset.append({"prompt": prompt, "response": response.strip()})

# Take a smaller subset
short_sample_size = 250
short_dataset = random.sample(cleaned_dataset, min(len(cleaned_dataset), short_sample_size))

# Save
short_path = Path("cleaned_finetune_ingredient_dataset.jsonl")
with short_path.open("w") as f:
    for entry in short_dataset:
        f.write(json.dumps(entry) + "\n")

short_path, short_dataset[:3]
