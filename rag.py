import chromadb
from sentence_transformers import SentenceTransformer

_embedder = None
_collection = None

def load_index(collection, embedder):
    global _embedder, _collection
    _embedder = embedder
    _collection = collection

def retrieve(question: str, n: int = 3) -> str:
    embedding = _embedder.encode([question]).tolist()
    results = _collection.query(query_embeddings=embedding, n_results=n)
    return "\n\n".join(results["documents"][0])