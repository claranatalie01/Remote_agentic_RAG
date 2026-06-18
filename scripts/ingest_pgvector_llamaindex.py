#!/usr/bin/env python3

import os
import csv
import asyncio
from typing import List

import aiohttp
from dotenv import load_dotenv
from pydantic import Field

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore


# Load environment variables from .env if available
load_dotenv()


class LlamaCppEmbedding(BaseEmbedding):
    # Store the embedding API URL as a Pydantic field required by LlamaIndex
    embedding_url: str = Field(description="llama.cpp embedding endpoint")

    def __init__(self, embedding_url: str, **kwargs):
        # Pass embedding_url to BaseEmbedding so LlamaIndex can use it
        super().__init__(embedding_url=embedding_url, **kwargs)

    async def _get_embedding(self, text: str) -> List[float]:
        # Send text to the llama.cpp embedding server
        async with aiohttp.ClientSession() as session:
            async with session.post(self.embedding_url, json={"input": [text]}) as resp:
                # Stop ingestion if embedding server returns an error
                if resp.status != 200:
                    raise RuntimeError(f"Embedding error {resp.status}: {await resp.text()}")

                # Parse OpenAI-compatible embedding response
                data = await resp.json()
                return data["data"][0]["embedding"]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        # Async query embedding used during retrieval
        return await self._get_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        # Async text embedding used during document ingestion
        return await self._get_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        # Sync wrapper required by LlamaIndex
        return asyncio.run(self._aget_query_embedding(query))

    def _get_text_embedding(self, text: str) -> List[float]:
        # Sync wrapper required by LlamaIndex
        return asyncio.run(self._aget_text_embedding(text))


def load_documents(csv_path: str) -> list[Document]:
    # Store all converted CSV rows as LlamaIndex Document objects
    docs: list[Document] = []

    # Open the FAQ CSV file
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Convert each CSV row into a Document
        for i, row in enumerate(reader):
            question = row.get("query", "").strip()
            answer = row.get("expected_answer_text", "").strip()
            domain = row.get("domain", "").strip()
            snippet = row.get("expected_context_snippet", "").strip()

            source_title = row.get("source_title", "HKPL Ask a Librarian FAQ").strip()
            source_url = row.get("source_url", "https://www.hkpl.gov.hk/en/ask-a-librarian/faq.html").strip()
            source_type = row.get("source_type", "official_website").strip()
            source_row_id = row.get("source_row_id", str(i)).strip()

            if not question or not answer:
                continue

            docs.append(
                Document(
                    text=f"Question: {question}\nAnswer: {answer}",
                    metadata={
                        "source_title": source_title,
                        "source_url": source_url,
                        "source": source_title,
                        "url": source_url,
                        "source_type": source_type,
                        "domain": domain,
                        "question": question,
                        "snippet": snippet,
                        "row_id": source_row_id,
                    },
                )
            )

    return docs


def main():
    # Read configuration from environment variables
    db_password = os.getenv("DB_PASSWORD", "postgres")
    data_path = os.getenv("DATA_PATH", "/app/data/hkpl_faq_clean.csv")
    embedding_url = os.getenv("EMBEDDING_URL", "http://embedding:8080/v1/embeddings")

    # Create the embedding model used by LlamaIndex
    embed_model = LlamaCppEmbedding(embedding_url=embedding_url)
    Settings.embed_model = embed_model

    # Connect LlamaIndex to PostgreSQL pgvector
    vector_store = PGVectorStore.from_params(
        database="hkpl_vector_db",
        user="postgres",
        password=db_password,
        host="postgres",
        port=5432,
        table_name="hkpl_faq",
        embed_dim=1024,
    )

    # Remove old rows before re-ingesting to avoid duplicate chunks
    vector_store.clear()

    # Tell LlamaIndex to store vectors in PostgreSQL
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load FAQ rows from CSV
    documents = load_documents(data_path)
    print(f"Loaded {len(documents)} documents")

    # Split long documents into smaller chunks
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} nodes")

    # Embed nodes and store them in the PGVectorStore table
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print("✅ Ingested into LlamaIndex PGVectorStore table: data_hkpl_faq")


if __name__ == "__main__":
    main()