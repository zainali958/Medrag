"""Parse PDFs -> chunk -> embed -> store in Chroma + chunks.jsonl (for BM25)."""
import json
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (CHUNKS_FILE, CHUNK_OVERLAP, CHUNK_SIZE, DB_DIR,
                    EMBED_MODEL, RAW_DIR)


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)      # re-join hyphenated words
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, size: int, overlap: int):
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # try to end on a sentence boundary
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + size // 2:
                end = cut + 1
        yield text[start:end].strip()
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)


def load_chunks():
    chunks = []
    for pdf in sorted(Path(RAW_DIR).glob("*.pdf")):
        reader = PdfReader(str(pdf))
        for page_no, page in enumerate(reader.pages, start=1):
            text = clean(page.extract_text() or "")
            if len(text) < 50:
                continue
            for i, piece in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
                chunks.append({
                    "id": f"{pdf.stem}-p{page_no}-{i}",
                    "text": piece,
                    "source": pdf.name,
                    "page": page_no,
                })
    return chunks


def main():
    chunks = load_chunks()
    if not chunks:
        raise SystemExit(f"No PDFs found in {RAW_DIR}. Add some and re-run.")
    print(f"{len(chunks)} chunks from {RAW_DIR}")

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection("guidelines")
    except Exception:
        pass
    col = client.create_collection("guidelines", metadata={"hnsw:space": "cosine"})

    batch = 64
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        embs = model.encode([c["text"] for c in part], normalize_embeddings=True)
        col.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            embeddings=embs.tolist(),
            metadatas=[{"source": c["source"], "page": c["page"]} for c in part],
        )
        print(f"  embedded {min(i + batch, len(chunks))}/{len(chunks)}")
    print("Done.")


if __name__ == "__main__":
    main()
