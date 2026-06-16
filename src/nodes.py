import os
import logging
import json
import asyncio
import re
import time
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from gliner2 import GLiNER2
from .state import LibraryBotState
from .retrieval import retriever   # LlamaIndex retriever (includes reranking)
import aiohttp

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)          # ✅ moved after logger definition

def get_current_datetime():
    """Returns the current date and time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ----------------------------------------------------------------------
# HTTP endpoint for LLM (answer generation)
# ----------------------------------------------------------------------
LLM_URL = os.getenv("LLM_URL", "http://llm:8080/v1/chat/completions")

async def http_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 4096) -> str:
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
            logger.debug(f"LLM raw response: {data}")
            response = data["choices"][0]["message"]["content"]
            if not response or len(response.strip()) == 0:
                response = "I'm sorry, I couldn't generate a proper answer. Please try again."
            return response

# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------
async def voice_to_text_node(state: LibraryBotState) -> dict:
    """Mock voice transcription node (kept as placeholder)."""
    logger.info("[Node] Voice-to-Text Conversion (mock)")
    transcribed = "請問沙田圖書館幾點關門？"
    return {"messages": [HumanMessage(content=transcribed)]}

# ----------------------------------------------------------------------
# GLiGuard safety classifier
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
# Intent router (only distinguishes greetings from everything else)
# ----------------------------------------------------------------------
async def intent_router_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Intent Router (keyword‑based)")
    user_input = state["messages"][-1].content.lower()

    if any(word in user_input for word in ["hello", "hi", "hey", "greeting"]):
        return {"request_type": "normal_info"}
    else:
        return {"request_type": "rag_search"}

# ----------------------------------------------------------------------
# RAG pipeline using LlamaIndex retriever (includes reranking)
# ----------------------------------------------------------------------
async def rag_pipeline_node(state: LibraryBotState) -> dict:
    logger.info("[Node] RAG Pipeline (LlamaIndex + built‑in reranking)")
    query = state["messages"][-1].content
    logger.debug(f"Query: {query}")

    start = time.time()
    nodes = await retriever.aretrieve(query)
    elapsed = time.time() - start
    logger.info(f"Retrieval took {elapsed:.3f} seconds")

    chunk_texts = [node.node.text for node in nodes]
    scores = [node.score for node in nodes]
    context = "\n\n".join(chunk_texts) if chunk_texts else "No relevant documents found."

    logger.debug(f"Retrieved {len(nodes)} nodes. Scores: {scores}")
    if nodes:
        logger.debug(f"Top chunk text: {nodes[0].node.text[:200]}")
    else:
        logger.debug("No chunks retrieved.")

    return {
        "retrieved_chunks": chunk_texts,
        "retrieved_context": context,
        "retrieved_scores": scores,
    }

# ----------------------------------------------------------------------
# Generate answer (using HTTP LLM)
# ----------------------------------------------------------------------
async def generate_answer_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Generate Answer")
    request_type = state.get("request_type", "normal_info")
    question = state["messages"][-1].content

    if request_type == "normal_info":
        system_msg = "You are a helpful library assistant. Keep answers short and friendly."
        full_prompt = f"{system_msg}\n\nUser: {question}"
        response = await http_llm(full_prompt, temperature=0.7)
        return {"messages": [AIMessage(content=response)]}

    context = state.get("retrieved_context", "")
    # Truncate context to avoid potential token overflow
    MAX_CONTEXT_CHARS = 300  # start conservative
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "..."
    if not context or context == "No relevant documents found.":
        fallback = "I'm sorry, I couldn't find that information. Could you rephrase or ask about a specific library branch (e.g., Shatin Library)?"
        return {"messages": [AIMessage(content=fallback)]}

    scores = state.get("retrieved_scores", [])
    
    if not scores or max(scores) < 0.50:  # adjust threshold as needed
        fallback = "I don't have enough confidence to answer that. Could you rephrase your question or ask about a specific library service?"
        return {"messages": [AIMessage(content=fallback)]}

    library_name = state.get("current_library_name")
    library_code = state.get("current_library_code")
    current_time = get_current_datetime()
    user_memory = state.get("user_memory", {})

    location_hint = f"The user is currently at or near **{library_name}** (code: {library_code}). " if library_name else ""
    date_hint = f"Current date and time: {current_time}. " if current_time else ""
    memory_hint = f"User context: {json.dumps(user_memory)}" if user_memory else ""

    # ✅ Build the system prompt first
    system_prompt = f"""You are the official HKPL (Hong Kong Public Libraries) assistant.  
    **IMPORTANT:** Do NOT include any reasoning, thinking, or analysis in your response. Output only the final answer.
    {date_hint}{location_hint}{memory_hint}

    **Instructions:**
    1. Answer based **only** on the provided context. Do not invent facts.
    2. **Generalise across phrasing:** The user's question may use different words, but if the intent matches information in the context, answer it. Treat paraphrases as equivalent to the original question in the context.
    3. **Extract fully:** If the context contains the exact answer, state it clearly and completely.
    4. **Handle partial information:** If the context provides only part of the answer, give what is available and politely note what is missing.
    5. **Handle empty or irrelevant context:** If the context is empty or does not address the question at all, say: "I don't have that information in my knowledge base. Please try rephrasing or ask about a specific library service."
    6. **Be concise:** Keep answers short (1-3 sentences), but include all essential facts.
    7. **Lists:** If the question asks for a list, present it in bullet points.

    **Context:**
    {context}

    **Question:** {question}
    **Answer:**"""


    logger.debug(f"Context length: {len(context)} characters")
    logger.debug(f"System prompt (first 500 chars): {system_prompt[:500]}...")

    response = await http_llm(system_prompt, temperature=0.0)
    return {"messages": [AIMessage(content=response)]}

# ----------------------------------------------------------------------
# Output safety filter
# ----------------------------------------------------------------------
async def output_safety_filter_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Output Safety Filter")
    answer = state["messages"][-1].content
    blocked_phrases = ["self-harm", "suicide", "kill yourself"]
    if any(phrase in answer.lower() for phrase in blocked_phrases):
        return {"is_output_safe": False, "messages": [AIMessage(content="I cannot provide that answer. Please contact library staff or call the Samaritans at 2896 0000 for immediate help.")]}
    return {"is_output_safe": True}