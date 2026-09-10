"""Retrieve the rule chunks most similar to a question."""

import json
import pathlib
import sys

import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"
INDEX_DIR = pathlib.Path("index")


def load_index():
    vectors = np.load(INDEX_DIR / "vectors.npy")
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text())
    return vectors, chunks


def embed_query(question):
    """Embed a question. nomic needs the 'search_query: ' prefix here, not
    'search_document: ' — the two prefixes are what make questions and passages
    land near each other in the vector space."""
    response = ollama.embed(model=EMBED_MODEL, input=f"search_query: {question}")
    vector = np.array(response["embeddings"][0], dtype=np.float32)
    return vector / np.linalg.norm(vector)


def search(question, k=5):
    vectors, chunks = load_index()
    query = embed_query(question)

    # every row is unit length and so is the query, so the dot product IS the
    # cosine similarity. One matrix-vector product scores all chunks at once.
    scores = vectors @ query

    # argsort is ascending, so take the tail and flip it to get the best first
    top = np.argsort(scores)[-k:][::-1]
    return [(float(scores[i]), chunks[i]) for i in top]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "how do mortal wounds work?"
    print(f"Q: {question}\n")
    for score, chunk in search(question):
        preview = " ".join(chunk["text"].split())[:150]
        print(f"{score:.3f}  {chunk['number']}  {chunk['title']}  (p{chunk['page']})")
        print(f"        {preview}...\n")
