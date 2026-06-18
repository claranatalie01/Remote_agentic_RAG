import os
import asyncio
import logging
from typing import List

import aiohttp
from dotenv import load_dotenv
from pydantic import Field

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.vector_stores.postgres import PGVectorStore


# Load environment variables
load_dotenv()

# Create logger for this file
logger = logging.getLogger(__name__)


class LlamaCppEmbedding(BaseEmbedding):
    # Embedding endpoint URL used by the local llama.cpp embedding server
    embedding_url: str = Field(description="llama.cpp embedding endpoint")

    def __init__(self, embedding_url: str, **kwargs):
        # Initialise the parent LlamaIndex embedding class
        super().__init__(embedding_url=embedding_url, **kwargs)

    async def _get_embedding(self, text: str) -> List[float]:
        # Call the embedding server with input text
        async with aiohttp.ClientSession() as session:
            async with session.post(self.embedding_url, json={"input": [text]}) as resp:
                # Raise error if embedding server fails
                if resp.status != 200:
                    raise RuntimeError(f"Embedding service error {resp.status}: {await resp.text()}")

                # Return embedding vector from response
                data = await resp.json()
                return data["data"][0]["embedding"]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        # Async method used by LlamaIndex for query embedding
        return await self._get_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        # Async method used by LlamaIndex for document embedding
        return await self._get_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        # Sync wrapper required by LlamaIndex
        return asyncio.run(self._aget_query_embedding(query))

    def _get_text_embedding(self, text: str) -> List[float]:
        # Sync wrapper required by LlamaIndex
        return asyncio.run(self._aget_text_embedding(text))


class HTTPReranker(BaseNodePostprocessor):
    # URL of the local reranker server
    reranker_url: str = Field(description="reranker endpoint")

    # Number of nodes to keep after reranking
    top_n: int = Field(3, description="top nodes to keep")

    def __init__(self, reranker_url: str, top_n: int = 3, **kwargs):
        # Initialise LlamaIndex postprocessor
        super().__init__(reranker_url=reranker_url, top_n=top_n, **kwargs)

    async def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        # If retrieval returned no nodes, return immediately
        if not nodes:
            return nodes

        # Get the original user query
        query = query_bundle.query_str

        # Extract text from retrieved nodes for reranking
        documents = [node.node.text for node in nodes]

        # Send query and retrieved documents to reranker
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.reranker_url,
                json={"query": query, "documents": documents},
            ) as resp:
                # If reranker fails, fall back to original vector search order
                if resp.status != 200:
                    logger.warning("Reranker failed; returning vector results.")
                    return nodes[: self.top_n]

                data = await resp.json()

        # Get reranker scores from response
        results = data.get("results", [])

        # If reranker response is empty, fall back to vector search order
        if not results:
            return nodes[: self.top_n]

        # Attach reranker scores to nodes
        for node, item in zip(nodes, results):
            node.score = item.get("relevance_score", 0.0)

        # Sort nodes by reranker score from highest to lowest
        nodes.sort(key=lambda x: x.score or 0.0, reverse=True)

        # Return only the best top_n nodes
        return nodes[: self.top_n]


# Read service URLs and database password from environment variables
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://embedding:8080/v1/embeddings")
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8080/reranking")

# Create embedding model used for query embeddings
embed_model = LlamaCppEmbedding(embedding_url=EMBEDDING_URL)

# Register embedding model globally for LlamaIndex
Settings.embed_model = embed_model

# Connect to the LlamaIndex-managed pgvector table
vector_store = PGVectorStore.from_params(
    database="hkpl_vector_db",
    user="postgres",
    password=DB_PASSWORD,
    host="postgres",
    port=5432,
    table_name="hkpl_faq",
    embed_dim=1024,
)

# Build a LlamaIndex object backed by PostgreSQL pgvector
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)

# Create reranker postprocessor
reranker = HTTPReranker(reranker_url=RERANKER_URL, top_n=3)

# Create retriever:
# 1. Retrieve top 10 candidates from pgvector
# 2. Rerank them
# 3. Keep top 3
retriever = index.as_retriever(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)

logger.info("✅ Retrieval configured with LlamaIndex PGVectorStore: data_hkpl_faq")