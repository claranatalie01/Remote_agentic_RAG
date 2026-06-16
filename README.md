# Agentic RAG Library Assistant

A production‑ready, containerised **Retrieval‑Augmented Generation (RAG)** system that answers library‑related questions using local open‑source models. The system uses **LlamaIndex** for indexing, retrieval, and reranking, and **LangGraph** for safety, intent routing, and answer generation. All components run inside Docker containers with GPU support.

---

## 📖 Table of Contents

- [Architecture](#architecture)
- [Why LlamaIndex?](#why-llamaindex)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Component Guide](#detailed-component-guide)
- [Data Ingestion & Test Dataset](#data-ingestion--test-dataset)
- [Logging & Debugging](#logging--debugging)
- [API Usage](#api-usage)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)

---

## 🏗️ Architecture (5 containers)

| Container | Role | Model / Technology | GPU |
|-----------|------|-------------------|-----|
| `postgres` | Vector database (pgvector) | – | CPU |
| `embedding` | Converts text to vectors | `Qwen3‑Embedding‑0.6B‑GGUF` (llama.cpp) | GPU 0 |
| `reranker` | Re‑ranks retrieved documents | `Qwen3‑Reranker‑0.6B‑Q8_0‑GGUF` (llama.cpp) | GPU 0 |
| `llm` | Generates final answer | `Qwen3.5‑9B‑MTP‑GGUF` (llama.cpp) | GPU 1 |
| `langgraph-agent` | Orchestrates safety, intent, retrieval (via LlamaIndex), answer streaming | LangGraph + FastAPI + LlamaIndex | CPU |

All services communicate via Docker’s internal network using service names (`postgres`, `embedding`, `reranker`, `llm`).

---

## 🧠 Why LlamaIndex?

Your earlier version used raw SQL and manual HTTP calls. While it worked, it lacked:

- **Benchmarking** – no easy way to measure retrieval quality (hit rate, MRR)
- **Flexibility** – changing embedding model required rewriting SQL and dimension handling
- **Maintainability** – retrieval logic was scattered across custom functions

LlamaIndex provides a standardised, modular interface. You now have:

- **`VectorStoreIndex`** and **`PGVectorStore`** for seamless pgvector integration (or an in‑memory index loaded from the database)
- **`BaseEmbedding`** abstraction – swap embedding models without touching LangGraph
- **`BaseNodePostprocessor`** for reranking (your existing HTTP reranker is now a postprocessor)
- Built‑in evaluation tools (e.g., `RetrieverEvaluator`)

LangGraph now only handles decision‑making and final LLM call – exactly as required.

---

## 📦 Prerequisites

- **Linux server** (or any machine) with:
  - Docker Engine ≥ 20.10
  - Docker Compose V2
  - NVIDIA drivers + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  - At least **one NVIDIA GPU** (two recommended: GPU0 for embedding+reranker, GPU1 for LLM)
- **Git** – to clone the repository
- **Python 3.11+** (optional – only needed for local ingestion; you can also ingest inside a container)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/agentic-RAG.git
cd agentic-RAG

# 2. (Optional) Create a .env file to change the database password
echo "DB_PASSWORD=postgres" > .env   # default is "postgres"

# 3. Start all containers (first start downloads models – 10–15 minutes)
docker compose up -d

# 4. Verify everything is running
docker compose ps
# All containers should show "Up"

# 5. Ingest sample library data (see Data Ingestion section)
#    Place your CSV file at ./data/hkpl_faq_clean.csv
docker run --rm -it \
  --network agentic-rag_default \
  -v $(pwd)/scripts:/scripts \
  -v $(pwd)/data:/data \
  -e DB_PASSWORD=postgres \
  -e DATA_PATH=/data/hkpl_faq_clean.csv \
  python:3.11-slim bash -c "
    pip install aiohttp psycopg2-binary langchain-core langchain-text-splitters sqlalchemy python-dotenv &&
    python /scripts/ingest.py
  "

# 6. Ask a question
curl -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"input_string": "How do I renew a book online?"}'
```

You should receive a streaming answer.

---

## 🔧 Detailed Component Guide

### PostgreSQL + pgvector
- Table: `document_chunks` (`id`, `text`, `embedding vector(1024)`, `metadata jsonb`)
- Initialised by `postgres-init/init.sql` (runs once on fresh volume)

### Embedding container
- llama.cpp with `--embedding`, model `Qwen3-Embedding-0.6B-GGUF:Q8_0`
- Endpoint: `http://embedding:8080/v1/embeddings` (OpenAI‑compatible)

### Reranker container
- llama.cpp with `--reranking`, model `Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0`
- Endpoint: `http://reranker:8080/reranking` (custom JSON format)

### LLM container
- llama.cpp with `Qwen3.5-9B-MTP-GGUF:Q6_K` (chat model)
- Endpoint: `http://llm:8080/v1/chat/completions` (OpenAI‑compatible)

### LangGraph Agent (inside `langgraph-agent` container)

#### Key files
| File | Purpose |
|------|---------|
| `main.py` | FastAPI entry point, `/chat/stream` endpoint |
| `src/state.py` | Typed state for LangGraph |
| `src/retrieval.py` | LlamaIndex index (in‑memory, loaded from DB), custom embedding, reranker postprocessor |
| `src/nodes.py` | LangGraph nodes (safety, intent, RAG, generate, filter) |
| `src/graph.py` | LangGraph workflow definition |

#### Logging in `nodes.py`
Each node logs its entry with `logger.info(f"[Node] ...")`. For detailed step‑by‑step logging (retrieval, reranking, generation), see the [Logging & Debugging](#logging--debugging) section.

---

## 📄 Data Ingestion & Test Dataset

### CSV format
The ingestion script expects a CSV with the following columns:
- `domain` – category (e.g., `reference_services`, `e_resources`, `collections`)
- `query` – the user question
- `expected_answer_text` – the full answer text
- `expected_bib_ids` – (optional) IDs for book records
- `expected_context_snippet` – a short summary (first sentence or key phrase)

### Generating a test dataset from the HKPL FAQ page
We provide an extraction script (`extract_faq.py`) that scrapes the [HKPL Ask a Librarian FAQ](https://www.hkpl.gov.hk/en/ask-a-librarian/faq.html) and generates a clean CSV.

**1. Create the extraction script:**

```bash
cat > extract_faq.py << 'EOF'
#!/usr/bin/env python3
"""
Extract FAQs from the HKPL "Ask a Librarian" page and save as a clean CSV.
Usage: python extract_faq.py
Output: hkpl_faq_clean.csv
"""

import requests
from bs4 import BeautifulSoup
import csv
import re

# ----------------------------------------------------------------------
# Domain classifier based on keywords in the question
# ----------------------------------------------------------------------
def classify_domain(question: str) -> str:
    q_lower = question.lower()
    if any(word in q_lower for word in ['e-resource', 'e-book', 'e-magazine', 'mobile app', 'digital', 'online']):
        return 'e_resources'
    elif any(word in q_lower for word in ['annual report', 'law', 'gazette', 'standard', 'collection']):
        return 'collections'
    else:
        return 'reference_services'

# ----------------------------------------------------------------------
# Main extraction
# ----------------------------------------------------------------------
def main():
    url = "https://www.hkpl.gov.hk/en/ask-a-librarian/faq.html"
    print(f"Fetching {url} ...")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all <li> that contain both .quest and .answ
    items = soup.find_all('li')
    data = []
    skipped = 0

    for item in items:
        question_elem = item.select_one('p.quest')
        answer_elem = item.select_one('div.answ')
        if question_elem and answer_elem:
            question = question_elem.text.strip()
            answer = answer_elem.text.strip()

            # Clean answer: replace multiple spaces/newlines with a single space
            answer = re.sub(r'\s+', ' ', answer).strip()
            # Remove invisible non-breaking spaces (common in web pages)
            answer = answer.replace('\u00a0', ' ').strip()

            # Context snippet: first sentence (up to first period)
            if '.' in answer:
                snippet = answer.split('.')[0] + '.'
            else:
                snippet = answer[:100] + '...' if len(answer) > 100 else answer

            domain = classify_domain(question)
            data.append([domain, question, answer, '', snippet])
        else:
            skipped += 1

    print(f"Extracted {len(data)} FAQ items (skipped {skipped} items without Q/A).")

    # Write to CSV
    output_file = 'hkpl_faq_clean.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'query', 'expected_answer_text', 'expected_bib_ids', 'expected_context_snippet'])
        writer.writerows(data)

    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
EOF
```

**2. Run the extraction script:**

```bash
docker run --rm -it \
  -v $(pwd):/data \
  -w /data \
  python:3.11-slim \
  bash -c "
    pip install --quiet requests beautifulsoup4 &&
    python /data/extract_faq.py
  "
```

This will create `hkpl_faq_clean.csv` in your current directory.

### Ingestion
After you have your CSV (either from the extraction script or your own dataset), run the ingestion container as shown in Quick Start. The script will:
- Read the CSV, create `Document` objects.
- Split each answer into chunks (500 characters, 50 overlap).
- Call the embedding container to generate vectors.
- Insert chunks and vectors into the `document_chunks` table.

---

## 📜 Logging & Debugging

The agent uses Python’s `logging` module. All nodes log their start, and key steps (retrieval count, reranking scores, LLM calls) are recorded. Logs are printed to stdout and can be viewed with `docker compose logs`.

### Enable detailed DEBUG logs
In `src/nodes.py`, set:
```python
logger.setLevel(logging.DEBUG)
```
Then restart the agent (no rebuild needed if using volume mounts).

### What logs will show
- `[Node] RAG Pipeline (LlamaIndex + built‑in reranking)` – start of retrieval
- `DEBUG:src.nodes:Query: ...` – the user query
- `INFO:src.nodes:Retrieval took ... seconds` – performance metric
- `DEBUG:src.nodes:Retrieved 10 nodes. Scores: [...]` – reranker scores
- `DEBUG:src.nodes:Top chunk text: ...` – the most relevant chunk
- `[Node] Generate Answer` – start of generation
- `DEBUG:src.nodes:Context length: ...` – prompt size
- `DEBUG:src.nodes:System prompt (first 500 chars): ...` – prompt preview

### View logs
```bash
# Follow all logs
docker compose logs -f langgraph-agent

# Show last 100 lines
docker compose logs langgraph-agent --tail 100

# Save logs to a file
docker compose logs langgraph-agent > agent_logs.txt
```

---

## 💻 API Usage

**Endpoint:** `POST /chat/stream`  
**Headers:** `Content-Type: application/json`  
**Body:** `{"input_string": "Your question here"}`  
**Response:** Server‑Sent Events (SSE)

Events:
- `event: node` – the current LangGraph node executing (`safety`, `intent_router`, `rag_pipeline`, `generate_answer`, `output_safety_filter`)
- `event: answer` – the final answer text
- `event: end` – marks completion

### Example with `curl`
```bash
curl -N -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"input_string": "What time does Shatin Library close?"}'
```

### Example with Python
```python
import requests
import json

resp = requests.post("http://localhost:8001/chat/stream", 
                     json={"input_string": "Library hours?"}, 
                     stream=True)
for line in resp.iter_lines():
    if line:
        line = line.decode("utf-8")
        if line.startswith("data: "):
            data = json.loads(line[6:])
            print(data)
```

---

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: llama_index` | Ensure `llama-index` and `llama-index-vector-stores-postgres` are in `requirements-agent.txt`. Rebuild. |
| `retriever` not defined | Check that `src/retrieval.py` exists and exports `retriever`. |
| Agent restarts with `ValueError` on `embedding_url` | Update `retrieval.py` to declare `embedding_url` as a `Field` and pass it to `super().__init__`. |
| Reranker returns 404 | Ensure the `reranker` service uses `--reranking` and the URL is `http://reranker:8080/reranking`. |
| GPU out of memory | Reduce `--n-gpu-layers` for LLM (e.g., from 99 to 50). |
| Retrieval returns 0 nodes | Check that `document_chunks` has non‑null embeddings. Re‑ingest if necessary. |
| Logs not showing details | Set `logger.setLevel(logging.DEBUG)` in `nodes.py` and restart. |
| `FileNotFoundError` during ingestion | Ensure `DATA_PATH` points to the correct file path inside the container (e.g., `/data/hkpl_faq_clean.csv`). |
| Empty answer from agent | Check logs for `RAG retrieved X chunks`. If 0, the database may be empty or embeddings are missing. |

---

## 📁 Project Structure

```
agentic-RAG/
├── compose.yaml
├── Dockerfile.agent
├── requirements-agent.txt
├── main.py
├── .env (optional)
├── postgres-init/
│   └── init.sql
├── src/
│   ├── retrieval.py          # LlamaIndex index, embedding, reranker postprocessor
│   ├── nodes.py              # LangGraph nodes with logging
│   ├── graph.py              # LangGraph workflow
│   └── state.py
├── scripts/
│   ├── ingest.py             # Data ingestion script
│   └── extract_faq.py        # Script to scrape HKPL FAQ and generate CSV
├── data/
│   └── (your CSV files)
└── README.md
```

**Note:** `utils.py` is not required – the `get_current_datetime` function is defined directly in `nodes.py` and `main.py` imports it from there. The `.env` file is optional; if not present, the default password `postgres` is used.

---

## 🚀 Future Improvements

- **Hybrid search** – add BM25 to the LlamaIndex retriever (`vector_store_query_mode="hybrid"`).
- **Benchmarking** – use `RetrieverEvaluator` to measure hit rate and MRR.
- **Metadata filtering** – store more metadata and add filters to the retriever.
- **Conversation memory** – extend LangGraph state to maintain chat history.
- **Faithfulness checks** – add a post‑generation step to verify the answer is grounded in the context.
- **LLM-based grading** – re‑enable `grade_docs_node` for self‑reflective retrieval.
- **Web interface** – add a simple Streamlit or Gradio UI.

---

## 📄 License

(Add your license here.)

---

## 🙋 Support

If you encounter issues, check the logs first (`docker compose logs langgraph-agent`). For detailed debugging, enable DEBUG logging as described above and share the output when opening an issue on GitHub.

