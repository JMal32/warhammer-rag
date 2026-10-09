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
| `eval_retrieval.py` | Scores retrieval: does each rule come back for a question about it? |
| `eval_answers.py` | Scores answers: right rule cited, right facts stated, refuses when it should |

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

Working: parsing, indexing, retrieval, and cited generation, each measured.

```sh
python eval_retrieval.py                 # one question per rule, built from its title
python eval_answers.py [model] [k]       # 15 hand-checked questions, incl. 2 that must be refused
```

| Eval | Result |
| --- | --- |
| Retrieval (156 rules) | R@1 88.5%, R@3 97.4%, R@5 98.7%, MRR 0.928 |
| Answers, `qwen2.5:7b`, k=5 | 12/15 (80.0%): 11/13 answers, 1/2 refusals |
| Answers, `qwen2.5:14b`, k=5 | 13/15 (86.7%): 11/13 answers, 2/2 refusals |

Answer scoring is deterministic — required rule number plus required facts,
each with a list of accepted wordings — rather than a judge model, because the
only local judge is the same model family being tested.

What the numbers show:

- **Retrieval is the bottleneck on player-style questions.** Both models fail
  the same two answer cases, and both are retrieval misses: "can a unit shoot in
  the turn it advanced?" ranks `10.05 ASSAULT SHOOTING` 7th, and "how far apart
  can models in the same unit be?" ranks `03.03 COHERENCY` 11th. The retrieval
  eval cannot see this, because its questions reuse the rule titles.
- **More context makes it worse, not better.** Raising k to fix those misses
  drops `qwen2.5:7b` from 80.0% to 66.7% at k=8 and k=12: it cites the wrong
  rule, hedges, and turns a correct refusal into a confident fabrication.
  Retrieval precision matters more than recall here.
- **The bigger model's edge is refusing.** Asked about an Ork ability that is
  not in the core rules, 7b invents an answer from an unrelated rule; 14b
  refuses. `CHAT_MODEL` in `answer.py` is a one-line swap.
- **Exact title matches do not always win.** Asked "what is a mortal wound",
  the embedder ranks `24.10 [DEVASTATING WOUNDS]` above `06.02 MORTAL WOUNDS`.
  All the top hits are topically about wounds, but the embedder has no notion
  that a title match should dominate. Keyword search ranks it 2nd as well.
- **Hybrid keyword + vector retrieval was tried and did not help.** `search.py`
  implements BM25 and reciprocal rank fusion (`MODE = "hybrid"`), but it scores
  R@1 83.3% against vector's 88.5%, and ties on answers at 80.0%. It does pull
  `10.05` into the top 5 for the advance question, where 7b then misreads it.
  It pushes `03.03` from 11th to 19th for the coherency question: "how far
  apart" shares no rare words with "within 2 inches", so there is nothing for
  keyword matching to find. The remaining misses are a vocabulary gap between
  player questions and rules text.
- **The last chunk swallows the appendix**, because nothing after `24.38` looks
  like a header.
- **Page furniture leaks** into chunk bodies — page numbers and banner text.

## Roadmap

- Query rewriting: have the model restate a player's question in rulebook
  language before embedding it, to close the vocabulary gap.
- The faction packs and event companions, not just the core rules, which brings
  in the FAQ-overrides-core-text precedence problem.
- Rewriting the similarity hot path (`vectors @ query`) as a C++ extension via
  pybind11 and benchmarking it against NumPy at 10k/100k/1M vectors.
