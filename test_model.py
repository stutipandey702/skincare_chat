from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./finetuned_model_m2_cpu"  # or your model folder

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # or set to a dedicated pad token if available
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()


def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt text only once
    response = response.replace(prompt, "").strip()

    if "### Answer:" in response:
        response = response.split("### Answer:")[-1].strip()

    return response  # just return as is, no slicing by prompt length here


if __name__ == "__main__":
    print("Type your prompt (or 'quit' to exit):")
    while True:
        prompt = input(">> ")
        if prompt.lower() in ["quit", "exit"]:
            break
        output = generate_response(prompt)
        print(f"Model output: {output}\n")


