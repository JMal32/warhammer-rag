"""Answer a rules question using only the retrieved chunks as context."""

import sys

import ollama

from search import search

CHAT_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """You are a Warhammer 40,000 rules assistant.

Answer using ONLY the rules text given to you in the user's message.

- Cite the rule number for every claim, like "(rule 06.02)".
- If the given rules do not answer the question, reply exactly:
  "The retrieved rules don't cover this."
  Do not guess, and do not use anything you know outside the given rules.
- Be concise: two or three sentences unless the question needs a list.
"""


def format_context(results):
    """Turn search results into the block of rules text the model reads.

    `results` is the list of (score, chunk) pairs that search() returns. The
    score is deliberately dropped here: 0.750 means nothing to a language
    model, and showing it only invites the model to reason about a number it
    has no calibration for. Scores stay in the debug output instead.

    Each chunk gets a header line carrying its rule number, so the model has
    something concrete to cite, and a visible divider between chunks so one
    rule does not read as a continuation of the last.
    """
    blocks = []
    for _, chunk in results:
        header = f"[Rule {chunk['number']} - {chunk['title']} (page {chunk['page']})]"
        blocks.append(f"{header}\n{chunk['text'].strip()}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question, context):
    """The system + user pair sent to the chat model.

    The rules go in the user message rather than the system prompt, so the
    system prompt stays fixed and cacheable while only the context changes.
    """
    user_content = (
        f"Rules retrieved for this question:\n\n"
        f"{context}\n\n"
        f"===\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def answer(question, k=5):
    """Retrieve, then generate.

    Returns (text, results) rather than just the text, so the caller can show
    which rules were actually used - that is the whole point of a RAG system
    over just asking the model.
    """
    results = search(question, k=k)
    context = format_context(results)
    messages = build_messages(question, context)

    # temperature 0 so the same question gives the same answer; the eval
    # harness needs generation to be the fixed part when retrieval changes.
    response = ollama.chat(
        model=CHAT_MODEL, messages=messages, options={"temperature": 0}
    )
    return response["message"]["content"], results


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "how do mortal wounds work?"
    print(f"Q: {question}\n")

    text, results = answer(question)
    print(text)

    print("\nsources:")
    for score, chunk in results:
        print(f"  {score:.3f}  {chunk['number']}  {chunk['title']}  (p{chunk['page']})")
