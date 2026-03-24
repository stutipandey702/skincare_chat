import os
import time
import random
import requests
import chromadb
import chromadb.config

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer



# CONFIG

INGREDIENTS = [
    "niacinamide", "retinol", "salicylic-acid", "hyaluronic-acid",
    "glycolic-acid", "vitamin-c", "ceramides",
    "lactic-acid", "kojic-acid", "tranexamic-acid", "squalane",
    "azelaic-acid", "caffeine", "glycoproteins"
]

BASE_URL = "https://renude.co/ingredients"
DATA_DIR = "knowledge_base"
CHROMA_DIR = "chroma_db"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml",
    "Connection": "keep-alive",
    "Referer": "https://renude.co/",
}



# HTTP SESSION

session = requests.Session()
session.headers.update(HEADERS)


def fetch_with_retry(url, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=10)

            if response.status_code == 200:
                return response

            print(f"[WARN] {url} -> Status {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] {url} -> {e}")

        time.sleep(backoff ** attempt)

    return None



# SCRAPING

def scrape_ingredient(ing: str) -> str:
    url = f"{BASE_URL}/{ing}"
    print(f"Scraping: {url}")

    response = fetch_with_retry(url)
    if not response:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    main = soup.find("main")
    text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)

    cutoffs = [
        "Your personalised skincare routine",
        "Personal skincare on any budget",
        "Privacy",
    ]

    for cutoff in cutoffs:
        if cutoff in text:
            text = text.split(cutoff)[0]

    return text.strip()



# CHUNKING

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    return chunks



# BUILD INDEX

def build_index():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Step 1: Scrape + cache
    for ing in INGREDIENTS:
        path = os.path.join(DATA_DIR, f"{ing}.txt")

        if not os.path.exists(path):
            text = scrape_ingredient(ing)

            if text:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"[OK] Saved {ing}")
            else:
                print(f"[SKIP] No content for {ing}")

            time.sleep(random.uniform(1.5, 4))
        else:
            print(f"[CACHE] Loaded {ing}")

    # Step 2: Load + chunk
    all_chunks = []
    all_ids = []
    all_metadata = []
    chunk_id = 0

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        ingredient_name = filename.replace(".txt", "")

        with open(filepath, encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(str(chunk_id))

            # metadata (useful later)
            all_metadata.append({
                "ingredient": ingredient_name
            })

            chunk_id += 1

    print(f"Total chunks: {len(all_chunks)}")

    # Step 3: Embed
    print("Embedding...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True).tolist()

    # Step 4: Persistent Chroma client
    client = chromadb.Client(
        settings=chromadb.config.Settings(
            persist_directory=CHROMA_DIR
        )
    )

    # Reset collection (avoids duplicate ID errors)
    try:
        client.delete_collection("skincare")
        print("[INFO] Deleted old collection")
    except:
        pass

    collection = client.create_collection("skincare")

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadata
    )

    print("✅ Index built and saved to disk.")
    return collection, embedder



# MAIN

if __name__ == "__main__":
    build_index()