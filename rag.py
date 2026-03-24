import chromadb
from sentence_transformers import SentenceTransformer

_embedder = None
_collection = None

def load_index(collection, embedder):
    global _embedder, _collection
    _embedder = embedder
    _collection = collection

def retrieve(question: str, n: int = 3) -> str:
    if _embedder is None or _collection is None:
        return ""

    embedding = _embedder.encode([question]).tolist()
    results = _collection.query(query_embeddings=embedding, n_results=n)

    docs = results.get("documents", [[]])[0]

    if not docs:
        return ""

    return "\n\n".join(
        [f"[Source {i+1}]\n{doc}" for i, doc in enumerate(docs)]
    )