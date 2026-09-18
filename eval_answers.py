"""Measure answer quality: does the generated answer cite the right rule and state the right facts?

eval_retrieval.py scores which chunks come back. It cannot say whether the
answer built from them is correct, so it could not score the qwen2.5 7b vs 14b
comparison, and it could not say whether a chunking change helped or hurt.
This file is that missing measurement.

Scoring is deterministic. Each case names the rule number the answer must cite
and the facts it must state, with each fact written as a list of accepted
wordings. There is no judge model: the only local one is qwen2.5, which we have
already watched garble these exact rules, and a judge you cannot trust produces
a score you cannot act on.

The cost is brittleness - a correct answer can fail on unusual wording. That is
the right way round for a regression test: a failure means "go read this
answer", not "the system is broken". Every expectation below was taken from the
rule text in the corpus, not from knowledge of the game.
"""

import re
import sys

import answer

# How many chunks to retrieve per question. Settable from the command line so
# the effect of a wider context can be measured rather than guessed at.
K = 5

# Each case: the question, the rule that answers it, and the facts the answer
# must contain. must_say holds one entry per fact; each entry lists wordings
# that all count as stating that fact.
CASES = [
    {
        "question": "a unit suffers 3 mortal wounds. which model loses wounds first?",
        "must_cite": "06.02",
        "must_say": [
            ["non-character"],
            ["has lost", "already lost", "lost one or more", "wounded", "damaged"],
        ],
    },
    {
        "question": "no charges happened and combat is still ongoing. which player selects a unit to fight first?",
        "must_cite": "12.04",
        "must_say": [["player whose turn", "active player", "whose turn it is"]],
    },
    {
        "question": "what happens to my shooting attack if the target has the benefit of cover?",
        "must_cite": "13.08",
        "must_say": [
            ["worsen", "worse", "-1", "subtract 1", "reduced by 1"],
            ["bs", "ballistic skill"],
        ],
    },
    {
        "question": "how does feel no pain work?",
        "must_cite": "24.12",
        "must_say": [["d6"], ["not lost", "is not lost", "ignored", "no wound"]],
    },
    {
        "question": "can a unit that made an advance move declare a charge that turn?",
        "must_cite": "09.06",
        "must_say": [["not eligible", "cannot", "can not", "can't", "may not"]],
    },
    {
        "question": "can a unit shoot in the turn it advanced?",
        "must_cite": "10.05",
        "must_say": [["assault"]],
    },
    {
        "question": "a devastating wounds weapon inflicts 3 mortal wounds from one critical wound. how many models can those mortal wounds damage, and what happens to any left over?",
        "must_cite": "24.10",
        "must_say": [
            ["one model", "1 model", "maximum of one", "single model"],
            ["lost", "wasted", "discarded"],
        ],
    },
    {
        "question": "when do I make hazard rolls for a hazardous weapon?",
        "must_cite": "24.15",
        "must_say": [["after"], ["attacks"]],
    },
    {
        "question": "how far apart can models in the same unit be?",
        "must_cite": "03.03",
        "must_say": [['2"', "2 inches"], ['9"', "9 inches"]],
    },
    {
        "question": "how do I work out which player controls an objective?",
        "must_cite": "14.02",
        "must_say": [["oc"], ["highest", "higher", "greater", "more"]],
    },
    {
        "question": "where on the battlefield can a unit with deep strike be set up when it arrives?",
        "must_cite": "24.09",
        "must_say": [['8"', "8 inches"], ["more than", "further than", "at least"]],
    },
    {
        "question": "which units have to take battle-shock rolls in the command phase?",
        "must_cite": "08.03",
        "must_say": [["half-strength", "half strength"], ["battle-shocked"]],
    },
    {
        "question": "when can I use the fire overwatch stratagem?",
        "must_cite": "15.08",
        "must_say": [["movement phase"], ["opponent"]],
    },
]

# Questions the corpus cannot answer. The core rules hold no points values and
# no faction rules, so the only correct response is the refusal string.
REFUSAL_CASES = [
    "how many points does a Land Raider cost?",
    "what does the Ork Waaagh! ability do?",
]


def normalise(text):
    """Lower-case, collapse whitespace, and straighten the PDF's curly quotes."""
    text = text.replace("”", '"').replace("“", '"')
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).lower()


def cited_rules(text):
    """Every rule number the answer mentions, e.g. {'06.02', '24.10'}."""
    return set(re.findall(r"\d{2}\.\d{2}", text))


def check(case, text, results):
    """Score one answer. Returns the list of problems, empty if it passed.

    `results` is what retrieval returned, so a missing rule can be blamed on
    the right half of the system: a rule that was never retrieved is a
    retrieval failure, while one that was retrieved and not cited is a
    generation failure.
    """
    problems = []
    body = normalise(text)

    if case["must_cite"] not in cited_rules(text):
        retrieved = case["must_cite"] in [chunk["number"] for _, chunk in results]
        where = "not cited (rule WAS retrieved)" if retrieved else "not cited (rule was NOT retrieved)"
        problems.append(f"{case['must_cite']} {where}")

    for wordings in case["must_say"]:
        if not any(w in body for w in wordings):
            problems.append(f"missing fact: {wordings[0]!r}")

    return problems


def run_case(case, k):
    text, results = answer.answer(case["question"], k=k)
    return check(case, text, results), text


def run_refusal(question, k):
    """A refusal passes only on the exact wording the prompt asks for.

    Anything else - a hedge, an apology, or an answer drawn from an unrelated
    rule - is a failure, because the point of the fixed string is that calling
    code can recognise it.
    """
    text, _ = answer.answer(question, k=k)
    passed = normalise(text).strip(' "') == normalise(answer.REFUSAL).strip(' "')
    return passed, text


def main():
    if len(sys.argv) > 1:
        answer.CHAT_MODEL = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else K
    print(f"scoring answers from {answer.CHAT_MODEL} at k={k}\n")

    passed = 0
    for case in CASES:
        problems, text = run_case(case, k=k)
        if problems:
            print(f"FAIL  {case['question']}")
            for problem in problems:
                print(f"        {problem}")
            print(f"        answer: {' '.join(text.split())[:160]}")
        else:
            passed += 1
            print(f"pass  {case['question'][:70]}")

    refused = 0
    for question in REFUSAL_CASES:
        ok, text = run_refusal(question, k=k)
        if ok:
            refused += 1
            print(f"pass  [refusal] {question[:60]}")
        else:
            print(f"FAIL  [refusal] {question}")
            print(f"        answered instead: {' '.join(text.split())[:160]}")

    total = len(CASES) + len(REFUSAL_CASES)
    print(f"\nanswers correct: {passed}/{len(CASES)}")
    print(f"refusals correct: {refused}/{len(REFUSAL_CASES)}")
    print(f"overall:         {(passed + refused) / total:.1%}")


if __name__ == "__main__":
    main()
