"""Retrieve the rule chunks most relevant to a question.

Two retrievers, fused:

- vector: nomic embeddings, cosine similarity. Good at meaning ("how far apart"
  ~ "within 2 inches"), weak at exact terms - it ranks DEVASTATING WOUNDS above
  MORTAL WOUNDS for "what is a mortal wound".
- keyword: BM25 over the rule text. Good at exact terms, blind to meaning.

Their scores are on unrelated scales, so they are combined by rank, not score
(reciprocal rank fusion).
"""

import functools
import json
import pathlib
import re
import sys

import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"
INDEX_DIR = pathlib.Path("index")

# "hybrid", "vector" or "keyword". Read at call time, like answer.CHAT_MODEL, so
# the evals can compare retrievers without editing this file.
#
# Vector stays the default because hybrid measured worse (k=5, qwen2.5:7b):
#   retrieval eval R@1: vector 88.5%, hybrid 83.3%, keyword 64.7%
#   answer eval:        vector 80.0%, hybrid 80.0%
# Hybrid does lift 10.05 into the top 5 for "can a unit shoot in the turn it
# advanced?" (rank 7 -> 5), but drops 03.03 for "how far apart can models in
# the same unit be?" (11 -> 19): that question shares no rare words with the
# rule, so keyword search has nothing to find.
MODE = "vector"

# BM25's two standard knobs, at their textbook defaults. K1 caps how much
# repeating a word keeps adding; B is how strongly long rules are penalised.
K1 = 1.5
B = 0.75

# Reciprocal rank fusion constant, from the original RRF paper. It flattens the
# curve so rank 1 vs rank 2 matters less than "in both top tens" vs "in one".
RRF_K = 60

# Restate the question in rulebook language before searching. Read at call time,
# like MODE. A player asks "how far apart can models be?"; the rule says "within
# 2 inches horizontally". Rewriting moves the question onto the rules' side of
# that vocabulary gap.
#
# Off by default because it measured as a wash (k=5, vector):
#   answer eval:        7b 80.0% -> 80.0%, 14b 86.7% -> 86.7%
#   retrieval eval R@1: 88.5% -> 72.4%
# It does fix retrieval for the player-phrased questions (10.05 rank 7 -> 1,
# 03.03 rank 11 -> 3), but the models then misread the rules they are given,
# and the title-based retrieval eval loses the title words it relied on.
REWRITE = False

# Fixed here rather than borrowed from answer.CHAT_MODEL, so that swapping the
# answer model in an eval does not silently change what gets retrieved.
REWRITE_MODEL = "qwen2.5:7b"

REWRITE_PROMPT = (
    "Rewrite the player's question the way a Warhammer 40,000 rulebook would "
    "phrase it, using official rules terminology. Reply with one sentence only, "
    "no explanation."
)


@functools.cache
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


@functools.cache
def rewrite_query(question):
    """The question restated in the rules' own vocabulary, for searching only.

    The rewrite is NOT trustworthy as a statement of the rules: the model makes
    up the content ("within 1 inch" for a rule that says 2). That is fine for
    retrieval, which only needs the right words, but it means the rewrite must
    never reach the answer model - answer() keeps passing the original question.

    Cached because the evals ask the same questions repeatedly and each rewrite
    costs a chat call, far more than the embedding it feeds.
    """
    response = ollama.chat(
        model=REWRITE_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": question},
        ],
        options={"temperature": 0},
    )
    return " ".join(response["message"]["content"].split())


def stem(word):
    """Strip one common suffix so 'advanced', 'advance' and 'advancing' match.

    Deliberately crude - a real stemmer (Porter) handles far more cases. The
    length check stops short words like 'has' being cut down to nothing useful.
    """
    for suffix in ("ing", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text):
    return [stem(word) for word in re.findall(r"[a-z0-9]+", text.lower())]


@functools.cache
def load_bm25():
    """Precompute a BM25 weight for every (chunk, word) pair.

    Returns (weights, vocab): weights is a chunks x words matrix, vocab maps a
    word to its column. Everything that does not depend on the question is done
    here, once, so scoring a question is the same shape as the vector search:
    one matrix-vector product.
    """
    _, chunks = load_index()
    docs = [tokenize(f"{c['title']} {c['text']}") for c in chunks]

    vocab = {}
    for doc in docs:
        for word in doc:
            vocab.setdefault(word, len(vocab))

    # tf[i, j] = how many times word j appears in chunk i
    tf = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for i, doc in enumerate(docs):
        for word in doc:
            tf[i, vocab[word]] += 1

    # idf: a word in few chunks is evidence, a word in every chunk is noise.
    # 'coherency' gets a large weight, 'the' and 'unit' get almost none.
    n = len(docs)
    df = (tf > 0).sum(axis=0)
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5))

    # length normalisation: a word appearing twice in a short rule says more
    # than twice in a long one
    lengths = tf.sum(axis=1, keepdims=True)
    norm = K1 * (1 - B + B * lengths / lengths.mean())

    # saturation: the 1st occurrence counts most, the 10th adds little
    weights = idf * tf * (K1 + 1) / (tf + norm)
    return weights, vocab


def keyword_scores(question):
    weights, vocab = load_bm25()
    # 1 for each distinct question word the corpus knows; unknown words drop out
    query = np.zeros(len(vocab), dtype=np.float32)
    for word in tokenize(question):
        if word in vocab:
            query[vocab[word]] = 1
    return weights @ query


def vector_scores(question):
    vectors, _ = load_index()
    # every row is unit length and so is the query, so the dot product IS the
    # cosine similarity. One matrix-vector product scores all chunks at once.
    return vectors @ embed_query(question)


def ranks(scores):
    """Rank of each chunk under these scores: 1 for the best, 2 for the next..."""
    order = np.argsort(-scores)
    result = np.empty(len(scores), dtype=np.int64)
    result[order] = np.arange(1, len(scores) + 1)
    return result


def hybrid_scores(question):
    """Reciprocal rank fusion: each retriever votes 1/(60 + rank) per chunk.

    A cosine of 0.80 and a BM25 score of 14.2 cannot be added - they are not in
    the same units. Ranks are. A chunk both retrievers put near the top beats
    one that only a single retriever loves.
    """
    return (1 / (RRF_K + ranks(vector_scores(question)))
            + 1 / (RRF_K + ranks(keyword_scores(question))))


SCORERS = {"vector": vector_scores, "keyword": keyword_scores, "hybrid": hybrid_scores}


def search(question, k=5):
    _, chunks = load_index()
    query = rewrite_query(question) if REWRITE else question
    scores = SCORERS[MODE](query)

    # argsort is ascending, so take the tail and flip it to get the best first
    top = np.argsort(scores)[-k:][::-1]
    return [(float(scores[i]), chunks[i]) for i in top]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "how do mortal wounds work?"
    print(f"Q: {question}  [{MODE}]")
    if REWRITE:
        print(f"   searched as: {rewrite_query(question)}")
    print()
    for score, chunk in search(question):
        preview = " ".join(chunk["text"].split())[:150]
        print(f"{score:.3f}  {chunk['number']}  {chunk['title']}  (p{chunk['page']})")
        print(f"        {preview}...\n")
