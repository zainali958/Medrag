from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"          # put guideline PDFs here
DB_DIR = ROOT / "data" / "chroma"
CHUNKS_FILE = ROOT / "data" / "chunks.jsonl"

EMBED_MODEL = "BAAI/bge-m3"              # multilingual, handles English + Urdu
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
USE_RERANKER = True

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b-instruct"        # use qwen2.5:3b-instruct if RAM is tight

CHUNK_SIZE = 800                         # characters
CHUNK_OVERLAP = 120
TOP_K_RETRIEVE = 20                      # candidates before reranking
TOP_K_FINAL = 5                          # passages sent to the LLM
