
# Agentic RAG System for Library Q&A

A production‑ready, containerised **RAG (Retrieval‑Augmented Generation)** system that answers questions about library services (opening hours, borrowing rules, book search, etc.). It uses:

- **PostgreSQL + pgvector** – vector database for document chunks
- **llama.cpp** – serves embedding, reranking, and LLM models (Qwen3 family)
- **LangGraph agent** – orchestrates retrieval, reranking, and answer generation
- **Docker Compose** – one‑command deployment with GPU support

All heavy models run inside Docker containers. Your local machine only needs Docker and the NVIDIA Container Toolkit (for GPU acceleration).

---

## 📦 Prerequisites

- **Linux server** (or any machine) with:
  - Docker Engine ≥ 20.10
  - Docker Compose V2 (included with Docker)
  - NVIDIA drivers + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  - At least **one NVIDIA GPU** (tested on RTX 2080 Ti, 11 GB VRAM)  
    *For CPU‑only, remove `runtime: nvidia` lines – but expect slow performance.*
- **Git** – to clone the repository
- **Python 3.11+** (optional – only needed for local ingestion script; you can also run ingestion inside a Docker container)

---

## 🚀 Quick Start (5 minutes)

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
# All containers should show "Up" (not "Restarting")

# 5. Ingest sample library data
#    Place your CSV file at ./data/test_dataset.csv (see format below)
#    Then run the ingestion script (from host or inside a container)

# Option A: Run ingestion on host
pip install aiohttp psycopg2-binary langchain-core langchain-text-splitters sqlalchemy python-dotenv
python scripts/ingest.py

# Option B: Run ingestion inside a temporary container (recommended)
docker run --rm -it \
  --network agentic-rag_default \
  -v $(pwd)/scripts:/scripts \
  -v $(pwd)/data:/data \
  -e DB_PASSWORD=postgres \
  -e DATA_PATH=/data/test_dataset.csv \
  python:3.11-slim bash -c "
    pip install aiohttp psycopg2-binary langchain-core langchain-text-splitters sqlalchemy python-dotenv &&
    python /scripts/ingest.py
  "

# 6. Ask a question
curl -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"input_string": "What time does Shatin Library close?"}'
```

You should receive a streaming answer like:  
`Shatin Public Library closes at 8:00 PM on weekdays and 6:00 PM on weekends.`

---

## 🧱 Architecture Overview

| Container | Purpose | Model (GGUF) | Internal Port | Host Port |
|-----------|---------|--------------|---------------|------------|
| `postgres` | Vector database (pgvector) | – | 5432 | 5433 |
| `embedding` | Text → vector (1024 dim) | `Qwen3-Embedding-0.6B-GGUF:Q8_0` | 8080 | 8003 |
| `reranker` | Query‑document scoring | `Qwen3-Reranker-0.6B-Q8_0-GGUF` | 8080 | 8004 |
| `llm` | Answer generation | `Qwen3.5-9B-MTP-Q6_K` | 8080 | 8081 |
| `langgraph-agent` | Orchestration + API (FastAPI) | – | 8001 | 8001 |

All containers communicate via Docker’s internal network using service names (`embedding`, `reranker`, `llm`, `postgres`).

---

## ⚙️ Configuration

### Environment variables (`.env` file)

Create a `.env` file in the root directory:

```env
DB_PASSWORD=postgres          # password for PostgreSQL (default is "postgres")
```

If not set, `postgres` is used automatically.

### Adjusting GPU allocation

By default:
- `embedding` and `reranker` use GPU 0 (device 0)
- `llm` uses GPU 1

If you have only one GPU, change `NVIDIA_VISIBLE_DEVICES=1` to `0` for the `llm` service and remove the `deploy.reservations.devices` section from `reranker` (or keep both on GPU 0 – works for small models).

### Changing models

- **Embedding**: edit `embedding` service’s `command` – change `-hf` model name
- **Reranker**: edit `reranker` service’s `command` – change `-hf` model name
- **LLM**: edit `llm` service’s `command` – change `-hf` model name

After changing, restart the affected service:
```bash
docker compose up -d --force-recreate <service>
```

---

## 🗄️ Database Schema

The first time you start, `postgres-init/init.sql` runs automatically and creates:

- `document_chunks` table with columns:
  - `id` (SERIAL PRIMARY KEY)
  - `text` (TEXT) – the chunk content
  - `embedding` (VECTOR(1024)) – 1024‑dim embedding
  - `metadata` (JSONB) – optional additional info (domain, query, etc.)

- pgvector extension (`vector`)
- IVFFlat index on `embedding` (optimises similarity search)

If you need to reset the database (delete all data):
```bash
docker compose down -v
docker compose up -d
```

---

## 📄 Ingesting Documents

You must populate the database with your own library documents. The repository includes an ingestion script (`scripts/ingest.py`) that reads a CSV file, splits text into chunks, creates embeddings, and stores them.

### CSV format expected

| column | description |
|--------|-------------|
| `domain` | category (e.g., "policy", "book") |
| `query` | example user question |
| `expected_answer_text` | answer text |
| `expected_bib_ids` | (optional) |
| `expected_context_snippet` | relevant context / source text |

**Example row:**
```csv
policy,What time does Shatin Library close?,Shatin Public Library closes at 8:00 PM on weekdays and 6:00 PM on weekends.,,Shatin Public Library hours: 9am-8pm weekdays, 10am-6pm weekends.
```

### Run ingestion

#### Option 1 – On the host (requires Python 3.11+)
```bash
pip install aiohttp psycopg2-binary langchain-core langchain-text-splitters sqlalchemy python-dotenv
python scripts/ingest.py
```
Make sure the `DATA_PATH` variable in `ingest.py` points to your CSV, or set the `DATA_PATH` environment variable.

#### Option 2 – Inside a temporary Docker container (recommended, no host Python needed)
```bash
docker run --rm -it \
  --network agentic-rag_default \
  -v $(pwd)/scripts:/scripts \
  -v $(pwd)/data:/data \
  -e DB_PASSWORD=postgres \
  -e DATA_PATH=/data/your_dataset.csv \
  python:3.11-slim bash -c "
    pip install aiohttp psycopg2-binary langchain-core langchain-text-splitters sqlalchemy python-dotenv &&
    python /scripts/ingest.py
  "
```

After ingestion, verify that data was inserted:
```bash
docker compose exec postgres psql -U postgres -d hkpl_vector_db -c "SELECT COUNT(*) FROM document_chunks;"
```

---

## 💬 Using the Agent API

The agent exposes a **streaming chat endpoint** at `http://localhost:8001/chat/stream`.

### Request format (JSON)
```json
{
  "input_string": "Your question here"
}
```

### Example with `curl`
```bash
curl -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"input_string": "How do I apply for a library card?"}'
```

### Streaming response

The endpoint returns **Server‑Sent Events (SSE)**. Each event is a JSON object containing the current step (`safety`, `intent_router`, `rag_pipeline`, `rerank`, `generate_answer`, etc.). The final answer appears in an `answer` event.

Example Python client:
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

## 🛠️ Useful Commands

| Action | Command |
|--------|---------|
| Start all containers | `docker compose up -d` |
| Stop all containers | `docker compose down` |
| Restart a single service | `docker compose restart <service>` |
| View logs (all) | `docker compose logs -f` |
| View logs (one service) | `docker compose logs -f langgraph-agent` |
| Rebuild agent after code change | `docker compose up -d --build langgraph-agent` |
| Reset database (wipe volumes) | `docker compose down -v` |
| Execute SQL inside Postgres | `docker compose exec postgres psql -U postgres -d hkpl_vector_db` |
| Check embedding endpoint | `curl http://localhost:8003/v1/embeddings -H "Content-Type: application/json" -d '{"input": ["test"]}'` |
| Check reranker endpoint | `curl http://localhost:8004/reranking -H "Content-Type: application/json" -d '{"query": "hours", "documents": ["open 9-5"]}'` |

---

## 🔍 Troubleshooting

### Containers crash or fail to start

- **Out of GPU memory**: Reduce `--n-gpu-layers` (e.g., from 99 to 50) for `embedding` or `reranker`.  
- **Port conflicts**: Change host ports in `compose.yaml` (e.g., `"8005:8080"`).  
- **Permission denied for Docker**: Add your user to the `docker` group: `sudo usermod -aG docker $USER` (log out and in).

### Embedding or reranker returns 404

- Ensure you are using the correct endpoint:  
  - Embedding → `/v1/embeddings`  
  - Reranker → `/reranking`  
- Check logs: `docker compose logs embedding` or `docker compose logs reranker`.

### Agent returns `{"detail":"Not Found"}`

- Use `/chat/stream` (not `/`).  
- Check that the agent started without errors: `docker compose logs langgraph-agent`.

### Database table `document_chunks` missing

Run the init script manually:
```bash
docker compose exec postgres psql -U postgres -d hkpl_vector_db -f /docker-entrypoint-initdb.d/init.sql
```

### Ingestion fails with “column metadata does not exist”

Add the column (should already be in `init.sql`):
```bash
docker compose exec postgres psql -U postgres -d hkpl_vector_db -c "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS metadata JSONB;"
```

### Empty answers or loops

- Ensure vectors are stored: `docker compose exec postgres psql -U postgres -d hkpl_vector_db -c "SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL;"` should return >0.
- If all rows have `NULL` embeddings, re‑run ingestion (make sure the `INSERT` uses `:embedding` without a cast).
- Check agent logs for retrieval count – it should be >0. If 0, check similarity distances manually.

### Slow performance on CPU

- Remove `runtime: nvidia` and all GPU environment variables from all services.  
- Set `--n-gpu-layers 0` (or remove that flag).  
- Use a smaller LLM (e.g., 1B‑3B parameters) or reduce batch sizes.

---

## 📁 Repository Structure

```
agentic-RAG/
├── compose.yaml                  # main Docker Compose file
├── Dockerfile.agent              # builds the LangGraph agent
├── requirements-agent.txt        # Python dependencies for agent
├── main.py                       # FastAPI entry point
├── hkpl_mcp_server.py            # MCP tools (optional)
├── src/                          # agent logic (LangGraph)
│   ├── graph.py
│   ├── nodes.py                  # HTTP calls to embedding/reranker/llm
│   ├── state.py
│   └── utils.py
├── postgres-init/                # SQL init scripts
│   └── init.sql
├── scripts/                      # utility scripts
│   └── ingest.py                 # document ingestion script
├── data/                         # place your CSV files here
│   └── test_dataset.csv          # example dataset
├── .env.example                  # environment variables template
└── README.md                     # this file
```

---

## 🤝 Contributing / Customising

- **Adding new tools**: Extend `hkpl_mcp_server.py` and modify `intent_router_node` in `src/nodes.py`.  
- **Using a different LLM**: Change the `-hf` model in the `llm` service. Ensure the context size (`--ctx-size`) matches the model’s capabilities.  
- **Using CPU only**: Remove GPU‑related lines and reduce batch sizes.

---
