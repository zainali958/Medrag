"""Hybrid retrieval (BM25 + vectors, fused with RRF) + optional reranker + local LLM."""
import json
import re
import time

import chromadb
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (CHUNKS_FILE, DB_DIR, EMBED_MODEL, LLM_MODEL, OLLAMA_URL,
                    RERANK_MODEL, TOP_K_FINAL, TOP_K_RETRIEVE, USE_RERANKER)

SYSTEM_PROMPT = """You are a medical-guidelines assistant. Answer ONLY using the numbered \
context passages. Cite passages like [1], [2] after each claim. If the context does not \
contain the answer, reply exactly: "I could not find this in the provided guidelines." \
Never diagnose, never recommend specific doses unless a passage states them, and answer in \
the same language as the question (English or Urdu). Be concise."""

DISCLAIMER = "\n\n_This is informational only, not medical advice. Consult a qualified clinician._"


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


class RAG:
    def __init__(self):
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.col = chromadb.PersistentClient(path=str(DB_DIR)).get_collection("guidelines")
        self.reranker = CrossEncoder(RERANK_MODEL) if USE_RERANKER else None

    # ---------- retrieval ----------
    def retrieve(self, query: str, k_final: int = TOP_K_FINAL, mode: str = "hybrid"):
        k = TOP_K_RETRIEVE
        vec_ids, bm_ids = [], []

        if mode in ("hybrid", "vector"):
            q = self.embedder.encode([query], normalize_embeddings=True)
            res = self.col.query(query_embeddings=q.tolist(), n_results=k)
            vec_ids = res["ids"][0]
        if mode in ("hybrid", "bm25"):
            scores = self.bm25.get_scores(tokenize(query))
            top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            bm_ids = [self.chunks[i]["id"] for i in top]

        # Reciprocal Rank Fusion
        fused = {}
        for ranking in (vec_ids, bm_ids):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
        cand = [self.by_id[c] for c, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]

        if self.reranker and cand:
            scores = self.reranker.predict([(query, c["text"]) for c in cand])
            cand = [c for _, c in sorted(zip(scores, cand), key=lambda x: -x[0])]
        return cand[:k_final]

    # ---------- generation ----------
    def answer(self, query: str):
        t0 = time.time()
        passages = self.retrieve(query)
        context = "\n\n".join(
            f"[{i}] ({p['source']}, p.{p['page']})\n{p['text']}"
            for i, p in enumerate(passages, start=1)
        )
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": LLM_MODEL,
            "stream": False,
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        }, timeout=300)
        resp.raise_for_status()
        text = resp.json()["message"]["content"].strip()
        return {
            "answer": text + DISCLAIMER,
            "sources": [{"n": i, "source": p["source"], "page": p["page"],
                         "snippet": p["text"][:200]} for i, p in enumerate(passages, 1)],
            "latency_s": round(time.time() - t0, 2),
        }
