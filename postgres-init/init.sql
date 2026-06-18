-- Enable pgvector extension for vector search support
CREATE EXTENSION IF NOT EXISTS vector;

-- Store short-term conversation history for session-based memory.
-- This table is used by src/memory.py.
CREATE TABLE IF NOT EXISTS conversation_history (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for quickly loading the latest turns from one session.
CREATE INDEX IF NOT EXISTS idx_conversation_history_session_time
ON conversation_history(session_id, created_at);

-- NOTE:
-- We no longer manually create document_chunks.
-- LlamaIndex PGVectorStore will automatically create/manage:
--
--     data_hkpl_faq
--
-- when scripts/ingest_pgvector_llamaindex.py runs.