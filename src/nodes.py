import os
import logging
import json
import asyncio
import urllib.parse
import re
from typing import Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from gliner2 import GLiNER2
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from .state import LibraryBotState
from .utils import get_current_datetime
import aiohttp

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MCP_SERVER_URL = "http://localhost:8000/mcp"

# ----------------------------------------------------------------------
# Helper mappings (keyword‑based intent routing)
# ----------------------------------------------------------------------
LIBRARY_NAME_TO_CODE = {
    "central": "HKCL",
    "hong kong central": "HKCL",
    "shatin": "STPL",
    "sha tin": "STPL",
    "tsim sha tsui": "TSTPL",
    "kowloon": "KLNPL",
}

KEYWORD_TO_TOOL = {
    "workstation|computer|pc|availability": "get_live_workstation_availability",
    "address|phone|email|details|about library": "get_library_details",
    "opening hours|open|hours": "get_library_opening_hours",
    "how many libraries|library count|total libraries": "get_library_count",
    "search for|find book|catalog|title|author": "search_library_catalog",
    "available|in stock|check if": "check_book_availability",
    "nearby|near me|close to|libraries in|district": "find_nearby_libraries",
}

# ----------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------
DB_PASSWORD = os.getenv("DB_PASSWORD")
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD not set in .env")
encoded_password = urllib.parse.quote(DB_PASSWORD, safe='')
CONNECTION_STRING = f"postgresql+psycopg2://postgres:{encoded_password}@postgres:5432/hkpl_vector_db"
engine = create_engine(CONNECTION_STRING, pool_size=10, max_overflow=20)

executor = ThreadPoolExecutor(max_workers=10)

async def run_db_query(query_text: str, params: dict):
    loop = asyncio.get_running_loop()
    with engine.connect() as conn:
        result = await loop.run_in_executor(
            executor,
            lambda: conn.execute(text(query_text), params).fetchall()
        )
    return result

# ----------------------------------------------------------------------
# HTTP endpoints for external services (set via environment variables)
# ----------------------------------------------------------------------
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://embedding:8080/v1/embeddings")
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8080/reranking")  # ✅ correct
LLM_URL = os.getenv("LLM_URL", "http://llm:8080/v1/chat/completions")

async def http_embed(texts: list[str]) -> list[list[float]]:
    async with aiohttp.ClientSession() as session:
        payload = {"input": texts, "model": "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"}
        async with session.post(EMBEDDING_URL, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Embedding service error {resp.status}: {text}")
            data = await resp.json()
            return [item["embedding"] for item in data["data"]]   # <-- must use data["data"]

async def http_rerank(query: str, documents: list[str]) -> list[float]:
    async with aiohttp.ClientSession() as session:
        payload = {"query": query, "documents": documents}
        async with session.post(RERANKER_URL, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Reranker service error {resp.status}: {text}")
            data = await resp.json()   # <-- no extra spaces before this line
            scores = [0.0] * len(documents)
            for item in data["results"]:
                scores[item["index"]] = item["relevance_score"]
            return scores

async def http_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "qwen3.5-9b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(LLM_URL, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"LLM service error {resp.status}: {text}")
            data = await resp.json()
            response = data["choices"][0]["message"]["content"]
            if not response or len(response.strip()) == 0:
                response = "I'm sorry, I couldn't generate a proper answer. Please try again."
            return response

# ----------------------------------------------------------------------
# Nodes (original, unchanged except replaced with HTTP)
# ----------------------------------------------------------------------
async def voice_to_text_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Voice-to-Text Conversion (mock)")
    transcribed = "請問沙田圖書館幾點關門？"
    return {"messages": [HumanMessage(content=transcribed)]}

# ----------------------------------------------------------------------
# GLiGuard safety classifier (unchanged)
# ----------------------------------------------------------------------
safety_model = None

def get_safety_model():
    global safety_model
    if safety_model is None:
        safety_model = GLiNER2.from_pretrained("fastino/gliguard-LLMGuardrails-300M")
        safety_model.to("cpu")
    return safety_model

async def safety_and_intent_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Safety Classifier (GLiGuard)")
    user_input = state["messages"][-1].content
    model = get_safety_model()

    toxicity_labels = [
        "violence_and_weapons", "non_violent_crime", "sexual_content",
        "hate_and_discrimination", "self_harm_and_suicide", "pii_exposure",
        "misinformation", "copyright_violation", "child_safety",
        "political_manipulation", "unethical_conduct", "regulated_advice",
        "privacy_violation", "other", "benign",
    ]
    toxicity_task = {
        "labels": toxicity_labels,
        "multi_label": True,
        "cls_threshold": 0.4,
    }
    jailbreak_labels = [
        "prompt_injection", "jailbreak_attempt", "policy_evasion",
        "instruction_override", "system_prompt_exfiltration", "data_exfiltration",
        "roleplay_bypass", "hypothetical_bypass", "obfuscated_attack",
        "multi_step_attack", "social_engineering", "benign",
    ]
    jailbreak_task = {
        "labels": jailbreak_labels,
        "multi_label": True,
        "cls_threshold": 0.4,
    }
    schema = {
        "prompt_safety": ["safe", "unsafe"],
        "prompt_toxicity": toxicity_task,
        "jailbreak_detection": jailbreak_task,
    }

    try:
        result = model.classify_text(user_input, schema, threshold=0.5)
        safety = result.get("prompt_safety", "safe")
        is_unsafe = (safety == "unsafe")
        detected_categories = []
        toxicity = result.get("prompt_toxicity", [])
        if isinstance(toxicity, list):
            detected_categories.extend(toxicity)
        jailbreak = result.get("jailbreak_detection", [])
        if isinstance(jailbreak, list):
            detected_categories.extend(jailbreak)
    except Exception as e:
        logger.error(f"GLiGuard classification error: {e}")
        is_unsafe = False
        detected_categories = []

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input[:500],
        "safety": "unsafe" if is_unsafe else "safe",
        "categories": detected_categories
    }

    if is_unsafe:
        with open("safety_intent_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.warning(f"Unsafe input blocked: {user_input[:100]} | categories: {detected_categories}")
        if any(cat in ["self_harm_and_suicide", "self_harm", "suicide"] for cat in detected_categories):
            safe_msg = (
                "I'm really sorry you're feeling this way. Please know that you're not alone. "
                "If you are in distress, please reach out to the Samaritans Hong Kong 24‑hour hotline at 2896 0000, "
                "or contact a mental health professional. Your well‑being is very important."
            )
        elif "political_manipulation" in detected_categories:
            safe_msg = (
                "I'm here to help with library services, book information, and general library questions. "
                "I can't discuss political topics. Is there something about the library I can help you with?"
            )
        elif any(cat in ["prompt_injection", "jailbreak_attempt", "instruction_override"] for cat in detected_categories):
            safe_msg = (
                "I can only follow instructions related to library services. "
                "Please ask a genuine question about library hours, book searches, or library policies."
            )
        elif "violence_and_weapons" in detected_categories:
            safe_msg = (
                "I cannot provide information that promotes or glorifies violence. "
                "If you need help with library resources, I'm happy to assist."
            )
        elif "hate_and_discrimination" in detected_categories:
            safe_msg = (
                "I strive to be respectful and helpful to everyone. "
                "Please ask a library‑related question without using offensive language."
            )
        else:
            safe_msg = (
                "I'm unable to process that request. Please ask a library‑related question, such as "
                "library hours, book availability, or how to borrow materials."
            )
        return {
            "messages": [AIMessage(content=safe_msg)],
            "is_output_safe": True,
            "end_conversation": True
        }

    return {}

# ----------------------------------------------------------------------
# Intent router (keyword‑based)
# ----------------------------------------------------------------------
async def intent_router_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Intent Router (keyword‑based)")
    user_input = state["messages"][-1].content.lower()

    for pattern, tool_name in KEYWORD_TO_TOOL.items():
        if re.search(pattern, user_input):
            args = {}
            if tool_name in ("get_library_details", "get_library_opening_hours", "get_live_workstation_availability"):
                lib_code = None
                for name, code in LIBRARY_NAME_TO_CODE.items():
                    if name in user_input:
                        lib_code = code
                        break
                if lib_code:
                    args["library_code"] = lib_code
                elif tool_name != "get_live_workstation_availability":
                    args["library_code"] = "HKCL"
            elif tool_name == "search_library_catalog":
                query_clean = re.sub(r"^(search for|find book|catalog for|find|search)\s+", "", user_input)
                args["query"] = query_clean
                args["limit"] = 5
            elif tool_name == "check_book_availability":
                args["title"] = user_input
                args["author"] = None
            elif tool_name == "find_nearby_libraries":
                match = re.search(r"in (\w+(?: \w+)?) district", user_input)
                args["district"] = match.group(1) if match else "Sha Tin"
            return {
                "request_type": "mcp_tool",
                "tool_name": tool_name,
                "tool_args": args
            }

    if any(word in user_input for word in ["hello", "hi", "hey", "greeting"]):
        return {"request_type": "normal_info"}
    else:
        return {"request_type": "rag_search"}

# ----------------------------------------------------------------------
# RAG pipeline (retrieve top‑5 raw chunks)
# ----------------------------------------------------------------------
async def rag_pipeline_node(state: LibraryBotState) -> dict:
    logger.info("[Node] RAG Pipeline")
    query = state["messages"][-1].content
    embedding_list = await http_embed([query])
    query_vector = embedding_list[0]
    query_vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"
    rows = await run_db_query(
        "SELECT text FROM document_chunks ORDER BY embedding <=> CAST(:embedding AS vector) LIMIT 5",
        {"embedding": query_vector_str}
    )
    chunk_texts = [row[0] for row in rows]
    context = "\n\n".join(chunk_texts) if chunk_texts else "No relevant documents found."
    logger.info(f"RAG retrieved {len(chunk_texts)} chunks")
    logger.info(f"Retrieved chunks: {chunk_texts}")
    return {
        "retrieved_chunks": chunk_texts,
        "retrieved_context": context
    }

# ----------------------------------------------------------------------
# Reranker via HTTP
# ----------------------------------------------------------------------
async def rerank_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Reranking retrieved chunks via HTTP")
    query = state["messages"][-1].content
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"retrieved_context": state.get("retrieved_context", ""), "is_relevant": False}
    chunks = chunks[:5]
    scores = await http_rerank(query, chunks)
    scored = list(zip(scores, chunks))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:3]]
    context_with_sources = "\n\n".join([f"[Source {i+1}] {chunk}" for i, chunk in enumerate(top_chunks)])
    return {"retrieved_context": context_with_sources, "is_relevant": len(top_chunks) > 0}

# ----------------------------------------------------------------------
# Grade and rewrite (using HTTP LLM)
# ----------------------------------------------------------------------
async def grade_docs_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Grade Documents")
    context = state.get("retrieved_context", "")
    question = state["messages"][-1].content
    if not context or context == "No relevant documents found.":
        return {"is_relevant": False}
    prompt = f"""Determine if the provided context contains any useful information that could help answer the user's question, even if not complete. Answer "YES" or "NO".

Context: {context}
Question: {question}"""
    try:
        result = await http_llm(prompt, temperature=0.0)
        is_relevant = result.strip().upper() == "YES"
    except Exception as e:
        logger.error(f"Grade error: {e}")
        is_relevant = True
    return {"is_relevant": is_relevant}

async def rewrite_query_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Rewrite Query")
    original = state["messages"][-1].content
    rewrite_count = state.get("rewrite_count", 0) + 1
    prompt = f"""Rewrite the following user question to be more specific, detailed, and effective for a vector similarity search in a library knowledge base.
Keep the original intent but improve clarity and keywords.

Original: {original}
Rewritten:"""
    rewritten = await http_llm(prompt, temperature=0.0)
    new_messages = state["messages"][:-1] + [HumanMessage(content=rewritten)]
    return {"messages": new_messages, "rewrite_count": rewrite_count}

# ----------------------------------------------------------------------
# Legacy API node (optional)
# ----------------------------------------------------------------------
async def utility_api_node(state: LibraryBotState) -> dict:
    logger.info("[Node] External API Call (HKPL)")
    import aiohttp
    api_url = "https://sls.hkpl.gov.hk/api/cfm-admin-service/open-api/library/selectLibraryPageInfoForPSI"
    params = {"language": "en-US", "sizePerPage": "9999"}
    timeout = aiohttp.ClientTimeout(total=5.0)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(api_url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        total_wkt = 0
                        for lib in data:
                            for sess in lib.get("sessionList", []):
                                for grp in sess.get("workstationGroup", []):
                                    total_wkt += grp.get("availableWktNumber", 0)
                        context = f"Live HKPL data: {len(data)} branches open. Total workstations available: {total_wkt}."
                    else:
                        context = "Live data format unexpected. Using fallback information."
                    return {"retrieved_context": context}
                else:
                    logger.error(f"API error {resp.status}")
                    return {"retrieved_context": "Live service temporarily unavailable. Using static hours."}
    except Exception as e:
        logger.error(f"API timeout: {e}")
        return {"retrieved_context": "Network error. Please try again later."}

# ----------------------------------------------------------------------
# Generate answer (using HTTP LLM)
# ----------------------------------------------------------------------
async def generate_answer_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Generate Answer")
    request_type = state.get("request_type", "normal_info")
    question = state["messages"][-1].content

    # --- Normal chat (no RAG) ---
    if request_type == "normal_info":
        system_msg = "You are a helpful library assistant. Keep answers short and friendly. Do not answer library‑specific factual questions – just chat."
        full_prompt = f"{system_msg}\n\nUser: {question}"
        response = await http_llm(full_prompt, temperature=0.7)
        return {"messages": [AIMessage(content=response)]}

    # --- RAG path ---
    context = state.get("retrieved_context", "")
    # If no context or empty, return fallback immediately
    if not context or context == "No relevant documents found.":
        fallback = "I'm sorry, I couldn't find that information. Could you rephrase or ask about a specific library branch (e.g., Shatin Library)?"
        return {"messages": [AIMessage(content=fallback)]}

    # Build hints (location, date, memory)
    library_name = state.get("current_library_name")
    library_code = state.get("current_library_code")
    current_time = state.get("current_datetime") or get_current_datetime()
    user_memory = state.get("user_memory", {})

    location_hint = ""
    if library_name:
        location_hint = f"The user is currently at or near **{library_name}** (code: {library_code}). " \
                        f"If they ask about 'the library' or a branch without naming it, assume they mean this library.\n"
    date_hint = f"Current date and time: {current_time}. " \
                "If the user asks for today's hours or events, use this information.\n"
    memory_hint = ""
    if user_memory:
        memory_hint = f"User context: {json.dumps(user_memory, indent=2)}\n"

    # System prompt with clear instructions
    system_prompt = f"""You are the official HKPL (Hong Kong Public Libraries) assistant.  

{date_hint}{location_hint}{memory_hint}

**Instructions:**
- Answer based **only** on the provided context. Do not invent facts.
- If the context contains the exact answer, state it clearly first, then add any relevant details.
- If the context has **partial** information, provide what is available and clearly state what is missing.
- If the context is completely empty or irrelevant, say: "I'm sorry, I couldn't find that information. Could you rephrase or ask about a specific library branch (e.g., Shatin Library)?"
- Keep answers concise (1-3 sentences), but include the most important facts.
- If the user asks for a list (e.g., multiple libraries), present it as a bullet list.

**Context:**
{context}

**Question:** {question}
**Answer:**"""

    response = await http_llm(system_prompt, temperature=0.0)
    return {"messages": [AIMessage(content=response)]}

# ----------------------------------------------------------------------
# Output safety filter & MCP tool node (unchanged)
# ----------------------------------------------------------------------
async def output_safety_filter_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Output Safety Filter")
    answer = state["messages"][-1].content
    blocked_phrases = ["unapproved", "harmful", "illegal", "self-harm", "suicide", "kill yourself"]
    if any(phrase in answer.lower() for phrase in blocked_phrases):
        return {"is_output_safe": False, "messages": [AIMessage(content="I cannot provide that answer. Please contact library staff or call the Samaritans at 2896 0000 for immediate help.")]}
    return {"is_output_safe": True}

async def mcp_tool_node(state: LibraryBotState) -> dict:
    logger.info(f"[Node] MCP Tool Node")
    tool_name = state.get("tool_name")
    tool_args = state.get("tool_args", {})
    if not tool_name:
        logger.error("mcp_tool_node called without tool_name")
        return {"retrieved_context": "No tool specified."}
    logger.info(f"[Node] MCP Tool Call: {tool_name} with args {tool_args}")
    try:
        async with streamablehttp_client(MCP_SERVER_URL) as transport:
            async with ClientSession(transport[0], transport[1]) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=tool_args)
                answer = result.content[0].text if result.content else "No output"
                return {"retrieved_context": answer}
    except Exception as e:
        logger.error(f"MCP call failed: {e}")
        return {"retrieved_context": f"Sorry, the tool '{tool_name}' is currently unavailable."}
