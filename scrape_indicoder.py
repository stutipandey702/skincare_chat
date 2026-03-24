import requests, os
from bs4 import BeautifulSoup

os.makedirs("knowledge_base", exist_ok=True)

ingredients = [
    "niacinamide", "retinol", "salicylic-acid", "hyaluronic-acid",
    "glycolic-acid", "ascorbic-acid", "ceramide-np", "benzoyl-peroxide",
    "lactic-acid", "kojic-acid", "tranexamic-acid", "squalane",
    "zinc-oxide", "alpha-arbutin", "azelaic-acid", "peptides"
]

for ing in ingredients:
    url = f"https://incidecoder.com/ingredients/{ing}"
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    with open(f"knowledge_base/{ing}.txt", "w") as f:
        f.write(text)
    print(f"Saved {ing}")