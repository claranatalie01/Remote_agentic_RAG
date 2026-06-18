from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from src.graph import compiled_workflow
from src.nodes import get_current_datetime
from src.memory import load_conversation_history

app = FastAPI(title="HKPL Agentic RAG Service")

class UserRequest(BaseModel):
    input_string: str
    session_id: str
    is_voice: bool = False
    stt_confidence: float = 1.0
    library_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    user_memory: Optional[dict] = None
    

@app.post("/chat/stream")
async def chat_stream(payload: UserRequest):
    current_library = None
    if payload.library_code:
        name_map = {"HKCL": "Hong Kong Central Library", "STPL": "Shatin Public Library"}
        current_library = {
            "code": payload.library_code,
            "name": name_map.get(payload.library_code, payload.library_code)
        }
    elif payload.latitude and payload.longitude:
        current_library = await resolve_current_library(payload.latitude, payload.longitude)

    history = load_conversation_history(payload.session_id)

    initial_state = {
        "messages": [HumanMessage(content=payload.input_string)],

        "session_id": payload.session_id,
        "conversation_history": history,

        "original_query": payload.input_string,
        "rewritten_query": payload.input_string,

        "input_type": "voice" if payload.is_voice else "text",
        "stt_confidence": payload.stt_confidence,

        "request_type": "normal_info",

        "retrieved_context": "",
        "retrieved_chunks": [],
        "retrieved_scores": [],
        "retrieved_sources": [],
        "generated_answer": "",
        "faithfulness_passed": True,
        "faithfulness_reason": "",
        "is_relevant": False,
        "rewrite_count": 0,
        "is_output_safe": True,
        "end_conversation": False,

        "tool_name": "",
        "tool_args": {},

        "current_library_code": current_library["code"] if current_library else None,
        "current_library_name": current_library["name"] if current_library else None,
        "current_datetime": get_current_datetime(),
        "user_memory": payload.user_memory or {},
    }

    async def event_generator():
        async for chunk in compiled_workflow.astream(initial_state, stream_mode="updates"):
            for node_name, updated in chunk.items():
                yield f"event: node\ndata: {node_name}\n\n"

                # Stream safety-block messages and final normal answers.
                if node_name in ["safety", "add_citations"]:
                    if isinstance(updated, dict) and "messages" in updated:
                        for msg in updated["messages"]:
                            if isinstance(msg, AIMessage):
                                yield f"event: answer\ndata: {msg.content}\n\n"

        yield "event: end\ndata: \n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)