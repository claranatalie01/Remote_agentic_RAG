-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the document chunks table (1024 dimensions for Qwen3-Embedding-0.6B)
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding VECTOR(1024),
    metadata JSONB
);

-- Optional: add an index for faster similarity search
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
ON document_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);