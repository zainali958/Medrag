# MedRAG: bilingual (English/Urdu) medical-guidelines assistant

Local, privacy-friendly RAG over clinical guidelines. Answers cite sources and refuse
when the guidelines don't contain the answer.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. install Ollama (https://ollama.com) then:
ollama pull qwen2.5:7b-instruct

# 2. put guideline PDFs in data/raw/ (WHO, CDC, national health ministry docs)
python src/ingest.py

# 3. run the API
cd src && uvicorn api:app --reload
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What is the first-line treatment for ...?"}'

# 4. evaluate retrieval (from the repo root)
python src/evaluate.py
```

## Roadmap
- [x] Ingestion, hybrid retrieval, reranker, local LLM, API
- [ ] Eval set (50-100 Qs) + results table in this README
- [ ] Faithfulness eval (LLM-as-judge) + refusal tests on out-of-scope questions
- [ ] Streamlit UI with source viewer + thumbs up/down logging
- [ ] Dockerfile, GitHub Actions, live demo
- [ ] Urdu test set
