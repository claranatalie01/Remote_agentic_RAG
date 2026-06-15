from typing import Literal
from langgraph.graph import StateGraph, START, END
from .state import LibraryBotState
from .nodes import (
    voice_to_text_node,
    safety_and_intent_node,
    intent_router_node,
    rag_pipeline_node,
    rerank_node,                     # added
    grade_docs_node,
    rewrite_query_node,
    utility_api_node,
    generate_answer_node,
    output_safety_filter_node,
    mcp_tool_node
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

def after_intent(state: LibraryBotState) -> Literal["rag_path", "tool_path", "direct_path"]:
    req = state.get("request_type", "normal_info")
    if req == "rag_search":
        return "rag_path"
    if req == "mcp_tool" or req == "tool_use":
        return "tool_path"
    return "direct_path"

def route_after_grade(state: LibraryBotState) -> Literal["generate_answer", "rewrite_query", "fallback_generate"]:
    if state.get("is_relevant", False):
        return "generate_answer"
    else:
        if state.get("rewrite_count", 0) >= 2:
            return "fallback_generate"
        return "rewrite_query"

def route_safety_decision(state: LibraryBotState) -> Literal["show", "block"]:
    return "show" if state.get("is_output_safe", True) else "block"

# ----------------------------------------------------------------------
# Build graph
# ----------------------------------------------------------------------
builder = StateGraph(LibraryBotState)

# Add all nodes
builder.add_node("voice_to_text", voice_to_text_node)
builder.add_node("safety", safety_and_intent_node)
builder.add_node("intent_router", intent_router_node)
builder.add_node("rag_pipeline", rag_pipeline_node)
builder.add_node("rerank", rerank_node)
builder.add_node("grade_docs", grade_docs_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("utility_api", utility_api_node)
builder.add_node("generate_answer", generate_answer_node)
builder.add_node("output_safety_filter", output_safety_filter_node)
builder.add_node("mcp_tool", mcp_tool_node)

# Start: either go through voice transcription or directly to safety
builder.add_conditional_edges(START, route_by_input_type, {
    "voice_path": "voice_to_text",
    "direct_to_safety": "safety"
})
builder.add_conditional_edges("voice_to_text", after_voice, {
    "safety": "safety"
})

# After safety: block or continue to intent router
builder.add_conditional_edges("safety", after_safety, {
    "end": END,
    "continue": "intent_router"
})

# After intent router: route to RAG, tool, or direct answer
builder.add_conditional_edges("intent_router", after_intent, {
    "rag_path": "rag_pipeline",
    "tool_path": "mcp_tool",
    "direct_path": "generate_answer"
})

# RAG pipeline: retrieval → rerank → (skip grading) → generate_answer
builder.add_edge("rag_pipeline", "rerank")
# builder.add_edge("rerank", "grade_docs")
builder.add_edge("rerank", "generate_answer")   # <-- new

# builder.add_conditional_edges("grade_docs", route_after_grade, {
#     "generate_answer": "generate_answer",
#     "rewrite_query": "rewrite_query",
#     "fallback_generate": "generate_answer"
# })
builder.add_edge("rewrite_query", "rag_pipeline")

# Tools and final answer
builder.add_edge("mcp_tool", "generate_answer")
builder.add_edge("utility_api", "generate_answer")   # optional

# Output safety filter
builder.add_edge("generate_answer", "output_safety_filter")
builder.add_conditional_edges("output_safety_filter", route_safety_decision, {
    "show": END,
    "block": END
})

compiled_workflow = builder.compile()