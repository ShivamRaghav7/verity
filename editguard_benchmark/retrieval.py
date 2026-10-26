"""Dependency-free, declared two-stage lexical RAG baseline. Never reads ground truth."""
import math
import re
from collections import Counter

STOP = set("a an the is are was in on for of to who which what name give only that team's team does as and with".split())


def tokens(text):
    return [word for word in re.findall(r"[a-z0-9]+", text.casefold()) if word not in STOP]


def rank(query, facts):
    documents = [tokens(row["question"] + " " + row["scope"] + " " + row["answer"]) for row in facts]
    counts = [Counter(words) for words in documents]
    average = sum(map(len, documents)) / max(1, len(documents))
    query_tokens = set(tokens(query))
    scores = []
    for row, counts_row, document in zip(facts, counts, documents):
        score = 0.0
        for word in query_tokens:
            frequency = counts_row[word]
            if not frequency:
                continue
            frequency_documents = sum(word in counter for counter in counts)
            inverse = math.log(1 + (len(facts) - frequency_documents + .5) / (frequency_documents + .5))
            score += inverse * frequency * 2.2 / (frequency + 1.2 * (.25 + .75 * len(document) / max(1, average)))
        scores.append((score, row))
    return [row for _, row in sorted(scores, key=lambda item: (-item[0], item[1]["id"]))]


def retrieve(query, facts, k=4):
    if not 1 <= k <= 8:
        raise ValueError("Retrieval k must be between 1 and 8")
    first = rank(query, facts)[:min(2, k)]
    expanded = query + " " + " ".join(row["answer"] for row in first)
    ranked = rank(expanded, facts)
    selected = first + [row for row in ranked if row["id"] not in {item["id"] for item in first}]
    return selected[:k]
