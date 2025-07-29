import csv
import json
import ast

input_csv = "/home/pandey36/exploration_pp/skincare_chat/ingredientsList.csv"
output_jsonl = "finetune_data.jsonl"

with open(input_csv, newline='', encoding='utf-8') as csvfile, open(output_jsonl, 'w', encoding='utf-8') as jsonlfile:
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        ingredient = row['name'].strip()
        what = row['what_does_it_do'].strip().replace("\n-", "It")
        
        prompt = f"What is {ingredient} good for?"
        response = f"{what}."
        
        json_obj = {
            "prompt": prompt,
            "response": response
        }
        
        jsonlfile.write(json.dumps(json_obj) + "\n")





print(f"Saved prompt-response pairs to {output_jsonl}") 
