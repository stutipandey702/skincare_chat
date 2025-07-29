# load the fine-tuned model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_dir = "./lora_finetuned_tinyllama_skinc"

print("Load the model")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(model, model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Load the model DONE!")

# test the model
prompt = "### Question: How to use retinol?\n### Answer:"
input_ids = tokenizer(prompt, return_tensors="pt").to(device)

# Use the custom stopping criteria
print("generating response")
output = model.generate(**input_ids, max_new_tokens=200) # Increased max_new_tokens to potentially get a second marker

# Decode the entire output
print("Decode the entire output")
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

# Find the index of the first '### Answer:' marker
answer_start = decoded_output.find("### Answer:")

if answer_start != -1:
    # Find the index of the next '### Question' marker after the first answer
    question_stop = decoded_output.find("### Question", answer_start + len("### Answer:"))
    if question_stop != -1:
        # Print the text between the markers
        print(decoded_output[answer_start + len("### Answer:"):question_stop].strip())
    else:
        # If no subsequent '### Question' is found, print from the answer start to the end
        print(decoded_output[answer_start + len("### Answer:"):].strip())
else:
    # If no '### Answer:' is found, print the whole decoded output
    print(decoded_output.strip())