from typing import Literal, TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class LibraryBotState(TypedDict):
    messages: List[BaseMessage]
    input_type: Literal["text", "voice"]
    stt_confidence: float
    request_type: Literal["sensitive_reject", "normal_info", "tool_use", "rag_search", "mcp_tool"]
    retrieved_context: str
    retrieved_chunks: List[str]           # raw chunk texts before reranking
    retrieved_scores: List[float]          # <-- ADD THIS LINE
    is_relevant: bool
    rewrite_count: int
    is_output_safe: bool
    tool_name: str
    tool_args: Dict[str, Any]
    end_conversation: bool
    current_library_code: Optional[str]
    current_library_name: Optional[str]
    current_datetime: str
    user_memory: Dict[str, Any]