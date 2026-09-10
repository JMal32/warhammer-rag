# warhammer-rag

A retrieval-augmented question answering system over the Warhammer 40,000 core
rules, running entirely on local models via [Ollama](https://ollama.com). Ask a
rules question in plain English, get an answer that cites the rule numbers it
came from.

```
$ python answer.py "how do mortal wounds work?"
Q: how do mortal wounds work?

Mortal wounds are resolved by selecting a model in the unit that suffers the
wound, then the selected model loses 1 wound. If this reduces the model's
remaining wounds to 0, the model is destroyed. Mortal wounds are resolved
after all normal damage from an attack has been resolved (rule 06.02).

sources:
  0.804  06.02  MORTAL WOUNDS  (p23)
  0.787  24.10  [DEVASTATING WOUNDS]  (p79)
  0.707  24.23  [LETHAL HITS]  (p81)
  0.699  05.02  WOUND ROLLS  (p17)
  0.689  24.12  FEEL NO PAIN  (p80)
```

## Why this corpus

Rules documents are a genuinely hard retrieval target, which is what makes this
more interesting than a RAG demo over blog posts. The 40k rules are dense with
cross-references, later FAQs override the core text, errata override the FAQs,
and near-identical wording appears in rules that mean different things. Getting
good answers is a retrieval-quality problem, not a prompt-engineering problem.

## How it works

Four small modules, each runnable on its own:

| File | Role |
| --- | --- |
| `ingest.py` | Parses the core rules PDF into one chunk per numbered rule |
| `index_build.py` | Embeds every chunk and saves the vectors to `index/` |
| `search.py` | Embeds a question and returns the top-k most similar rules |
| `answer.py` | Feeds those rules to a chat model constrained to cite them |

Some decisions worth calling out:

**Chunking on rule numbers, not token windows.** The rules are already written
as numbered atomic units (`06.02 MORTAL WOUNDS`), so that is the natural chunk
boundary. Headers are found with a regex, but a regex alone also matches every
cross-reference in body text. The fix is a **sequential-numbering gate**: a
candidate header is only accepted if it is the number that legally follows the
last accepted one, so `06.02` → `06.03` is a header while a mid-paragraph
mention of `06.02` is not. On the core rules this yields 156 chunks spanning
`01.01`–`24.38` and correctly rejects 4 cross-references.

**Normalise at index time.** Vectors are L2-normalised when written to disk, so
cosine similarity at query time collapses to a plain dot product, and scoring
all 156 chunks is one matrix-vector product: `scores = vectors @ query`.

**The embedding prefixes are not optional.** `nomic-embed-text` is trained with
instruction prefixes — `search_document: ` at index time, `search_query: ` at
query time. They are what put a question and its answering passage near each
other in the vector space. Omitting them quietly degrades retrieval.

**Scores are shown to you, not to the model.** The similarity score is useful
for debugging retrieval, but a language model has no calibration for what 0.750
means, so it is stripped from the prompt and printed only in the source list.

## Running it

Requires [Ollama](https://ollama.com) and Python 3.

```sh
ollama serve                        # or: brew services start ollama
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

python -m venv .venv && .venv/bin/pip install pymupdf numpy ollama
```

The rules PDF is **not** in this repository — it is Games Workshop's
copyrighted material. Games Workshop publishes the core rules for free; put the
PDF in `data/` and point `CORE_RULES` in `ingest.py` at it. The pipeline is
bring-your-own-documents and nothing about it is 40k-specific beyond the header
regex.

```sh
python index_build.py               # one-time, builds index/
python answer.py "how does overwatch work?"
python search.py "how does overwatch work?"    # retrieval only, no generation
```

## Current state and known limits

Working: parsing, indexing, retrieval, and cited generation. These are real
limitations, not unknowns:

- **Generation is the weak link.** `qwen2.5:7b` retrieves well but reasons
  poorly over rules text. Asked who fights first in an ongoing combat, it pulled
  exactly the right chunks and then produced a circular non-answer. `CHAT_MODEL`
  is a module constant so swapping generation to an API model is a one-line
  change; embeddings stay local either way.
- **Exact title matches do not always win.** Asked "what is a mortal wound",
  the embedder ranks `24.10 [DEVASTATING WOUNDS]` above `06.02 MORTAL WOUNDS`.
  All the top hits are topically about wounds, but the embedder has no notion
  that a title match should dominate. This is the motivating case for hybrid
  keyword + vector retrieval.
- **The last chunk swallows the appendix**, because nothing after `24.38` looks
  like a header.
- **Page furniture leaks** into chunk bodies — page numbers and banner text.

## Roadmap

- An eval harness: ~50 rules questions with verified answers, scored
  automatically, so retrieval changes can be measured instead of eyeballed.
- Hybrid retrieval to fix the exact-title-match problem.
- The faction packs and event companions, not just the core rules, which brings
  in the FAQ-overrides-core-text precedence problem.
- Rewriting the similarity hot path (`vectors @ query`) as a C++ extension via
  pybind11 and benchmarking it against NumPy at 10k/100k/1M vectors.
