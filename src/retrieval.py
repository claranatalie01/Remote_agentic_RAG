import os
import asyncio
import json
import logging
from typing import List
import aiohttp
from dotenv import load_dotenv
from pydantic import Field
from sqlalchemy import create_engine, text

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.postprocessor.types import BaseNodePostprocessor

load_dotenv()

# ----------------------------------------------------------------------
# Custom embedding (calls your llama.cpp embedding container)
# ----------------------------------------------------------------------
class LlamaCppEmbedding(BaseEmbedding):
    embedding_url: str = Field(description="URL of the llama.cpp embedding service")

    def __init__(self, embedding_url: str, **kwargs):
        super().__init__(embedding_url=embedding_url, **kwargs)

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
# Custom reranker (calls your llama.cpp reranker container)
# ----------------------------------------------------------------------
class HTTPReranker(BaseNodePostprocessor):
    reranker_url: str = Field(description="URL of the reranker service")
    top_n: int = Field(3, description="Number of top nodes to keep after reranking")

    def __init__(self, reranker_url: str, top_n: int = 3, **kwargs):
        super().__init__(reranker_url=reranker_url, top_n=top_n, **kwargs)

    async def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: QueryBundle) -> List[NodeWithScore]:
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

# ----------------------------------------------------------------------
# Load data from PostgreSQL directly into an in‑memory index
# ----------------------------------------------------------------------
DB_PASSWORD = os.getenv("DB_PASSWORD")
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD not set in .env")

# Database connection
db_url = f"postgresql://postgres:{DB_PASSWORD}@postgres:5432/hkpl_vector_db"
engine = create_engine(db_url)
# Fetch all rows
with engine.connect() as conn:
    rows = conn.execute(text("SELECT text, embedding FROM document_chunks")).fetchall()

# Build nodes with existing embeddings
nodes = []
for text, embedding in rows:
    # Convert the string representation of the list back to a Python list
    if isinstance(embedding, str):
        embedding_list = json.loads(embedding)
    else:
        embedding_list = embedding  # In case it's already a list
    node = TextNode(text=text, embedding=embedding_list)
    nodes.append(node)

logger = logging.getLogger(__name__)
logger.info("🔍 LOADED_INDEX: {} nodes".format(len(nodes)))
# ----------------------------------------------------------------------
# Embedding model (for query embedding)
# ----------------------------------------------------------------------
embedding_url = os.getenv("EMBEDDING_URL", "http://embedding:8080/v1/embeddings")
embed_model = LlamaCppEmbedding(embedding_url=embedding_url)
Settings.embed_model = embed_model

# Create an in‑memory index (no PGVectorStore needed)
index = VectorStoreIndex(nodes, embed_model=embed_model)

# ----------------------------------------------------------------------
# Create reranker and retriever
# ----------------------------------------------------------------------
reranker = HTTPReranker(reranker_url=os.getenv("RERANKER_URL", "http://reranker:8080/reranking"), top_n=3)

retriever = index.as_retriever(
    similarity_top_k=10,
    node_postprocessors=[reranker]
)