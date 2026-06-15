import os
import asyncio
from typing import List
import aiohttp
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.vector_stores.pgvector import PGVectorStore

load_dotenv()

# ----------------------------------------------------------------------
# Custom embedding using your llama.cpp embedding container
# ----------------------------------------------------------------------
class LlamaCppEmbedding(BaseEmbedding):
    def __init__(self, embedding_url: str, **kwargs):
        super().__init__(**kwargs)
        self.embedding_url = embedding_url

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._get_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._get_embedding(text)

    async def _get_embedding(self, text: str) -> List[float]:
        async with aiohttp.ClientSession() as session:
            async with session.post(self.embedding_url, json={"input": [text]}) as resp:
                data = await resp.json()
                return data["data"][0]["embedding"]

    def _get_query_embedding(self, query: str) -> List[float]:
        return asyncio.run(self._aget_query_embedding(query))

    def _get_text_embedding(self, text: str) -> List[float]:
        return asyncio.run(self._aget_text_embedding(text))

# ----------------------------------------------------------------------
# Custom reranker that calls your llama.cpp reranker container
# ----------------------------------------------------------------------
class HTTPReranker(BaseNodePostprocessor):
    def __init__(self, reranker_url: str, top_n: int = 3):
        self.reranker_url = reranker_url
        self.top_n = top_n

    async def _apostprocess(self, nodes: List[NodeWithScore], query_bundle: QueryBundle) -> List[NodeWithScore]:
        if not nodes:
            return nodes
        query = query_bundle.query_str
        documents = [node.node.text for node in nodes]
        async with aiohttp.ClientSession() as session:
            async with session.post(self.reranker_url, json={"query": query, "documents": documents}) as resp:
                data = await resp.json()
                scores = [item["relevance_score"] for item in data["results"]]
        for node, score in zip(nodes, scores):
            node.score = score
        nodes.sort(key=lambda x: x.score, reverse=True)
        return nodes[:self.top_n]

    def _postprocess(self, nodes: List[NodeWithScore], query_bundle: QueryBundle) -> List[NodeWithScore]:
        return asyncio.run(self._apostprocess(nodes, query_bundle))

# ----------------------------------------------------------------------
# Connect to pgvector
# ----------------------------------------------------------------------
DB_PASSWORD = os.getenv("DB_PASSWORD")
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD not set in .env")

vector_store = PGVectorStore.from_params(
    database="hkpl_vector_db",
    user="postgres",
    password=DB_PASSWORD,
    host="postgres",
    port=5432,
    table_name="document_chunks",
    embed_dim=1024,                # must match your stored vectors
)

# ----------------------------------------------------------------------
# Embedding model (uses your llama.cpp embedding service)
# ----------------------------------------------------------------------
embedding_url = os.getenv("EMBEDDING_URL", "http://embedding:8080/v1/embeddings")
embed_model = LlamaCppEmbedding(embedding_url=embedding_url)
Settings.embed_model = embed_model

# ----------------------------------------------------------------------
# Build index and retriever with reranking
# ----------------------------------------------------------------------
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Create reranker postprocessor
reranker = HTTPReranker(reranker_url=os.getenv("RERANKER_URL", "http://reranker:8080/reranking"), top_n=3)

# Retriever: first fetch 10 candidates, then rerank to top 3
retriever = index.as_retriever(
    similarity_top_k=10,
    node_postprocessors=[reranker]
)