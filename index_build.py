"""Embed every rule chunk and save the vectors to disk."""

import json
import pathlib

import numpy as np
import ollama

from ingest import parse_pdf

EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 32
INDEX_DIR = pathlib.Path("index")


def chunk_to_document(chunk):
    """The exact text we embed for one chunk.

    The title and rule number go in alongside the body, because a question like
    "what is a mortal wound" should match on the heading, not just the prose.
    nomic-embed-text requires the 'search_document: ' prefix at index time.
    """
    return (
        f"search_document: {chunk['title']} ({chunk['number']})\n{chunk['text']}"
    )


def embed_all(documents):
    """Embed a list of strings, in batches, returning one vector per string."""
    vectors = []
    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        response = ollama.embed(model=EMBED_MODEL, input=batch)
        vectors.extend(response["embeddings"])
        print(f"  embedded {start + len(batch)}/{len(documents)}")
    return np.array(vectors, dtype=np.float32)


def normalise(matrix):
    """Scale every row to length 1, so cosine similarity becomes a dot product."""
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / lengths


def main():
    chunks, rejected = parse_pdf()
    print(f"parsed {len(chunks)} chunks, rejected {len(rejected)} cross-references")

    documents = [chunk_to_document(c) for c in chunks]
    vectors = normalise(embed_all(documents))

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, indent=2))

    print(f"saved {vectors.shape[0]} vectors of {vectors.shape[1]} dims to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
