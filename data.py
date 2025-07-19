import pandas as pd
import json

# Load the original multi-column dataset
df = pd.read_csv("ingredientsList.csv")

# Drop rows with missing values in critical columns
df_cleaned = df.dropna(subset=["name", "what_is_it", "what_does_it_do", "who_is_it_good_for", "who_should_avoid"])


# Clean and parse the 'who_is_it_good_for' list
def parse_list_string(s):
    try:
        # Convert string list (e.g., "['Acne', 'Dry skin']") into real list
        items = ast.literal_eval(s)
        # Filter out empty/whitespace items and strip spaces
        return ", ".join([i.strip() for i in items if i.strip()])
    except:
        return ""
    

# Open JSONL output
with open("finetune_ingredient_dataset_with_targets.jsonl", "w", encoding="utf-8") as f:
    for _, row in df_cleaned.iterrows():
        name = row["name"]
        what_is_it = str(row["what_is_it"]).strip()
        what_does_it_do = str(row["what_does_it_do"]).strip()
        who_is_it_good_for = parse_list_string(str(row["who_is_it_good_for"]))
        who_should_avoid = parse_list_string(str(row["who_should_avoid"]))

        prompt = f"What is {name} and what is it good for?"

        # Append the list to the response (you can adjust wording as needed)
        response = f"{what_is_it} {what_does_it_do} It is especially good for: {who_is_it_good_for}. It should be avoided by those with: {who_should_avoid}."

        json_obj = {"prompt": prompt, "response": response}
        f.write(json.dumps(json_obj) + "\n")


print("✅ JSONL fine-tuning dataset created: finetune_ingredient_dataset.jsonl")
