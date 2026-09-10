"""Measure retrieval quality: when a question is about a rule, does that rule come back?

The test set is built from the rules themselves. Every chunk has a title, so
"what is the rule for mortal wounds?" should retrieve 06.02 MORTAL WOUNDS. That
gives one test case per rule for free, with no hand-written answer key.

This is a floor, not a ceiling. These questions reuse the rule's own title
words, so they are easier than what a player would actually type. A system that
scores badly here is definitely broken; one that scores well is not necessarily
good. The point is to have a number that moves when retrieval changes.
"""

import json
import pathlib

from search import search

INDEX_DIR = pathlib.Path("index")
K = 5


def load_chunks():
    return json.loads((INDEX_DIR / "chunks.json").read_text())


def make_question(chunk):
    """Build a question whose correct answer is this chunk.

    Titles arrive as 'MORTAL WOUNDS' or '[DEVASTATING WOUNDS]'. The brackets
    mark weapon abilities in the printed rules but nobody types them, so they
    come off.
    """
    title = chunk["title"].strip("[] ").lower()
    return f"what is the rule for {title}?"


def rank_of(expected_number, results):
    """Where the correct rule landed: 1 is best, None means it missed entirely."""
    numbers = [chunk["number"] for _, chunk in results]
    if expected_number in numbers:
        return numbers.index(expected_number) + 1
    return None


def evaluate(k=K):
    """Run every test case. Returns (ranks, misses).

    `ranks` has one entry per rule. `misses` holds the cases that did not come
    back at rank 1, which is where the interesting failures live.
    """
    chunks = load_chunks()
    ranks = []
    misses = []

    for i, chunk in enumerate(chunks, 1):
        question = make_question(chunk)
        results = search(question, k=k)
        rank = rank_of(chunk["number"], results)

        ranks.append(rank)
        if rank != 1:
            misses.append((chunk, results, rank))

        if i % 25 == 0:
            print(f"  {i}/{len(chunks)}")

    return ranks, misses


def recall_at(ranks, n):
    """Fraction of questions where the right rule was in the top n."""
    hits = sum(1 for r in ranks if r is not None and r <= n)
    return hits / len(ranks)


def mrr(ranks):
    """Mean reciprocal rank: rewards putting the right answer nearer the top.

    Rank 1 scores 1.0, rank 2 scores 0.5, rank 5 scores 0.2, a miss scores 0.
    One number that captures 'how high up was it', not just 'was it there'.
    """
    return sum(1 / r if r else 0 for r in ranks) / len(ranks)


def main():
    print(f"evaluating retrieval at k={K}\n")
    ranks, misses = evaluate()

    print(f"\ncases:      {len(ranks)}")
    print(f"recall@1:   {recall_at(ranks, 1):.1%}")
    print(f"recall@3:   {recall_at(ranks, 3):.1%}")
    print(f"recall@5:   {recall_at(ranks, 5):.1%}")
    print(f"MRR:        {mrr(ranks):.3f}")

    complete = [c for c, _, r in misses if r is None]
    print(f"\nnot in top {K} at all: {len(complete)}")
    for chunk in complete[:10]:
        print(f"  {chunk['number']}  {chunk['title']}")

    beaten = [(c, res, r) for c, res, r in misses if r is not None]
    print(f"\nfound but outranked: {len(beaten)}")
    for chunk, results, rank in beaten[:10]:
        winner = results[0][1]
        print(f"  {chunk['number']} {chunk['title']} -> rank {rank}, "
              f"beaten by {winner['number']} {winner['title']}")


if __name__ == "__main__":
    main()
