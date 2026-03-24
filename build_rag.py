import os
import chromadb
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import requests

def scrape_ingredient(ing: str) -> str:
    url = f"https://incidecoder.com/ingredients/{ing}"
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    
    text_parts = []
    for div_id in ["quickfacts", "details"]:
        div = soup.find("div", {"id": div_id})
        if div:
            text_parts.append(div.get_text(separator="\n", strip=True))
    
    return "\n\n".join(text_parts)

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

def build_index():
    os.makedirs("knowledge_base", exist_ok=True)
    ingredients = [
        "niacinamide", "retinol", "salicylic-acid", "hyaluronic-acid",
        "glycolic-acid", "ascorbic-acid", "ceramide-np", "benzoyl-peroxide",
        "lactic-acid", "kojic-acid", "tranexamic-acid", "squalane",
        "zinc-oxide", "alpha-arbutin", "azelaic-acid"
    ]

    for ing in ingredients:
        path = f"knowledge_base/{ing}.txt"
        if not os.path.exists(path):  # skip if already scraped
            text = scrape_ingredient(ing)
            with open(path, "w") as f:
                f.write(text)
        print(f"Loaded {ing}")

    all_chunks, all_ids, chunk_id = [], [], 0
    for filename in os.listdir("knowledge_base"):
        with open(f"knowledge_base/{filename}") as f:
            text = f.read()
        for chunk in chunk_text(text):
            all_chunks.append(chunk)
            all_ids.append(str(chunk_id))
            chunk_id += 1

    print(f"Total chunks: {len(all_chunks)}, embedding...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True).tolist()

    client = chromadb.Client()
    collection = client.create_collection("skincare")
    collection.add(documents=all_chunks, embeddings=embeddings, ids=all_ids)
    print("Index built.")
    return collection, embedder

if __name__ == "__main__":
    build_index()