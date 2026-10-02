"""Retrieval evaluation: compares BM25 vs vector vs hybrid (+ reranker if enabled).
Each line of eval/questions.jsonl: {"question": "...", "source": "file.pdf", "page": 12}
A hit = the correct (source, page) appears in the top-k results."""
import json
import sys

from config import ROOT
from rag import RAG

K = 5


def main():
    path = ROOT / "eval" / "questions.jsonl"
    qs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not qs:
        sys.exit("Add questions to eval/questions.jsonl first.")
    rag = RAG()

    print(f"{len(qs)} questions | metric: hit@{K} and MRR")
    for mode in ("bm25", "vector", "hybrid"):
        hits, rr = 0, 0.0
        for q in qs:
            res = rag.retrieve(q["question"], k_final=K, mode=mode)
            ranks = [i for i, c in enumerate(res, 1)
                     if c["source"] == q["source"] and c["page"] == q["page"]]
            if ranks:
                hits += 1
                rr += 1 / ranks[0]
        print(f"{mode:8s} hit@{K}={hits/len(qs):.2%}  MRR={rr/len(qs):.3f}")


if __name__ == "__main__":
    main()
