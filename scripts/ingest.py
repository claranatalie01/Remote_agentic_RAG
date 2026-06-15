import os
import json
import urllib.parse
import csv
import asyncio
import aiohttp
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, text

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "/home/cnatalie/agentic-rag/data/test_dataset.csv")

DB_PASSWORD = os.getenv("DB_PASSWORD")
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD not set in .env")

encoded_password = urllib.parse.quote(DB_PASSWORD, safe='')
CONNECTION_STRING = f"postgresql+psycopg2://postgres:{encoded_password}@postgres:5432/hkpl_vector_db"
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://embedding:8080/v1/embeddings")

async def embed_texts(texts):
    async with aiohttp.ClientSession() as session:
        async with session.post(EMBEDDING_URL, json={"input": texts}) as resp:
            if resp.status != 200:
                raise Exception(await resp.text())
            data = await resp.json()
            return [item["embedding"] for item in data["data"]]

async def main():
    # 1. Read CSV
    documents = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = f"Question: {row['query']}\nAnswer: {row['expected_answer_text']}\nDetails: {row['expected_context_snippet']}"
            metadata = {
                "domain": row["domain"],
                "query": row["query"],
                "expected_bib_ids": row["expected_bib_ids"],
                "expected_answer": row["expected_answer_text"]
            }
            documents.append(Document(page_content=content, metadata=metadata))

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} rows, split into {len(chunks)} chunks.")

    # 3. Embed all chunks
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = await embed_texts(chunk_texts)

    # 4. Connect to Postgres and insert
    engine = create_engine(CONNECTION_STRING)
    
    # Truncate table (clear old data)
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE document_chunks;"))
        conn.commit()

    # Insert chunks one by one
    for chunk, vector in zip(chunks, embeddings):
        vector_str = "[" + ",".join(str(x) for x in vector) + "]"
        metadata_json = json.dumps(chunk.metadata)
        with engine.connect() as conn:
            conn.execute(
                text("INSERT INTO document_chunks (text, embedding, metadata) VALUES (:text, :embedding, :metadata)"),
                {"text": chunk.page_content, "embedding": vector_str, "metadata": metadata_json}
            )
            conn.commit()
        print(f"Ingested: {chunk.page_content[:60]}...")

    print(f"Ingestion complete. {len(chunks)} chunks inserted.")

if __name__ == "__main__":
    asyncio.run(main())