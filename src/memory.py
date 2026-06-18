import os
from typing import List, Dict

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://postgres:postgres@postgres:5432/hkpl_vector_db"
)

engine = create_engine(DB_URL)


def load_conversation_history(session_id: str, limit: int = 6) -> List[Dict[str, str]]:
    # Load the latest conversation turns for this session
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT role, content
                FROM conversation_history
                WHERE session_id = :session_id
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {
                "session_id": session_id,
                "limit": limit,
            },
        ).fetchall()

    # Return oldest-to-newest order for prompt readability
    return [
        {"role": row.role, "content": row.content}
        for row in reversed(rows)
    ]


def save_conversation_turn(session_id: str, role: str, content: str) -> None:
    # Save one message into conversation history
    if not session_id or not content:
        return

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO conversation_history (session_id, role, content)
                VALUES (:session_id, :role, :content)
            """),
            {
                "session_id": session_id,
                "role": role,
                "content": content,
            },
        )