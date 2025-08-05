---
tags:
- tiny-llama
- lora
- fine-tuning
- skincare
- causal-lm
license: apache-2.0
---

# Tiny LLaMA 1.1B LoRA Fine-tuned on Skincare Data

This repository hosts a Tiny LLaMA 1.1B model fine-tuned using LoRA on a curated dataset of skincare ingredients and their properties. The model is designed to generate helpful, concise explanations of skincare ingredients and their effects.

## Model Details

- **Base model:** Tiny LLaMA 1.1B
- **Fine-tuning method:** LoRA (Low-Rank Adaptation)
- **Dataset:** Custom dataset of skincare ingredients and descriptions, formatted in JSONL
- **Training environment:** Google Colab (4-bit quantization with BitsAndBytes)
- **Use case:** Generate clear, informative responses about skincare ingredients

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name_or_path = "your-username/your-tiny-llama-skincare-model"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", load_in_4bit=True)

prompt = "What does niacinamide do for the skin?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Limitations

- Model may produce incorrect or incomplete skincare advice; use for informational purposes only.
- Fine-tuned on a niche dataset — may not generalize well outside skincare domain.
- 4-bit quantization helps reduce model size but may affect output quality slightly.

## License

This model is released under the Apache 2.0 License.