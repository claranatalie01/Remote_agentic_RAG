from typing import Literal
from langgraph.graph import StateGraph, START, END
from .state import LibraryBotState
from .nodes import (
    voice_to_text_node,
    safety_and_intent_node,
    intent_router_node,
    rag_pipeline_node,
    generate_answer_node,
    output_safety_filter_node,
)

# ----------------------------------------------------------------------
# Routing logic
# ----------------------------------------------------------------------
def route_by_input_type(state: LibraryBotState) -> Literal["voice_path", "direct_to_safety"]:
    if state["input_type"] == "voice" and state.get("stt_confidence", 1.0) < 0.85:
        return "voice_path"
    return "direct_to_safety"

def after_voice(state: LibraryBotState) -> Literal["safety"]:
    return "safety"

def after_safety(state: LibraryBotState) -> Literal["end", "continue"]:
    if state.get("end_conversation", False):
        return "end"
    return "continue"

def after_intent(state: LibraryBotState) -> Literal["rag_path", "direct_path"]:
    req = state.get("request_type", "normal_info")
    if req == "rag_search":
        return "rag_path"
    return "direct_path"

def route_safety_decision(state: LibraryBotState) -> Literal["show", "block"]:
    return "show" if state.get("is_output_safe", True) else "block"

# ----------------------------------------------------------------------
# Build graph
# ----------------------------------------------------------------------
builder = StateGraph(LibraryBotState)

# Add all necessary nodes (no extra nodes)
builder.add_node("voice_to_text", voice_to_text_node)
builder.add_node("safety", safety_and_intent_node)
builder.add_node("intent_router", intent_router_node)
builder.add_node("rag_pipeline", rag_pipeline_node)
builder.add_node("generate_answer", generate_answer_node)
builder.add_node("output_safety_filter", output_safety_filter_node)

# Start edges
builder.add_conditional_edges(START, route_by_input_type, {
    "voice_path": "voice_to_text",
    "direct_to_safety": "safety"
})
builder.add_conditional_edges("voice_to_text", after_voice, {"safety": "safety"})

# After safety
builder.add_conditional_edges("safety", after_safety, {
    "end": END,
    "continue": "intent_router"
})

# After intent router
builder.add_conditional_edges("intent_router", after_intent, {
    "rag_path": "rag_pipeline",
    "direct_path": "generate_answer"
})

# RAG directly to answer (reranking already inside retriever)
builder.add_edge("rag_pipeline", "generate_answer")

# Output safety filter
builder.add_edge("generate_answer", "output_safety_filter")
builder.add_conditional_edges("output_safety_filter", route_safety_decision, {
    "show": END,
    "block": END
})

compiled_workflow = builder.compile()