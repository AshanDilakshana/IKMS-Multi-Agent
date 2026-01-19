import uuid
from typing import Dict, List, Optional
from datetime import datetime
from ..core.database import get_db_connection

class SessionService:

    @classmethod
    def get_history(cls, session_id: str) -> List[dict]:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC", 
                (session_id,)
            ).fetchall()
            
            return [
                {
                    "question": row["content"] if row["role"] == "user" else "",
                    "answer": row["content"] if row["role"] == "assistant" else "",
                    "context": row["context"],
                    "timestamp": row["timestamp"]
                }
                # Note: The original format expected merged turns (Q & A together).
                # However, our DB stores them separately. 
                # We need to re-construct the 'turn' format expected by the frontend/graph.
                # Let's simple return the raw messages for now, or adapt the graph to handle raw messages.
                # Looking at original code: 
                # turn = { "question": ..., "answer": ..., "context": ... }
                # The graph probably expects this. Let's reconstruct it.
            ]

    @classmethod
    def get_history_formatted(cls, session_id: str) -> List[dict]:
        """
        Reconstructs turns from flat message list. 
        Assumes strictly alternating User -> Assistant structure for simplicity, 
        or groups them by time proximity.
        """
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", 
                (session_id,)
            ).fetchall()

        history = []
        current_turn = {}
        turn_index = 1
        
        for row in rows:
            if row["role"] == "user":
                current_turn = {
                    "question": row["content"],
                    "timestamp": row["timestamp"],
                    "turn": turn_index
                }
            elif row["role"] == "assistant":
                if "question" in current_turn:
                    current_turn["answer"] = row["content"]
                    current_turn["context"] = row["context"]
                    # Ensure all fields are present
                    if "turn" not in current_turn:
                         current_turn["turn"] = turn_index
                    history.append(current_turn)
                    current_turn = {}
                    turn_index += 1
                else:
                    # Assistant message without preceding user message? 
                    pass
        
        return history

    @classmethod
    def add_turn(cls, session_id: str, question: str, answer: str, context: Optional[str] = None):
        with get_db_connection() as conn:
            # 1. Ensure session exists (redundant if create_session called, but safe)
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, title) VALUES (?, ?)", 
                (session_id, "New Chat")
            )
            
            # 2. Update title if it's currently "New Chat" (meaning first turn)
            # using first 50 chars of the question
            conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ? AND title = 'New Chat'",
                (question[:50], session_id)
            )

            # 2. Add User Message
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "user", question)
            )

            # 3. Add Assistant Message
            conn.execute(
                "INSERT INTO messages (session_id, role, content, context) VALUES (?, ?, ?, ?)",
                (session_id, "assistant", answer, context)
            )
            
            conn.commit()
    
    @classmethod
    def create_session(cls) -> str:
        session_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title) VALUES (?, ?)", 
                (session_id, "New Chat")
            )
            conn.commit()
        return session_id

    @classmethod
    def list_sessions(cls) -> List[dict]:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    @classmethod
    def delete_session(cls, session_id: str):
        with get_db_connection() as conn:
            # Delete messages first (foreign key constraint usually handles this if ON DELETE CASCADE, 
            # but explicit is safe)
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()

    @classmethod
    def cleanup_old_sessions(cls, days: int = 7):
        with get_db_connection() as conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE created_at < datetime('now', ?))", 
                (f'-{days} days',)
            )
            conn.execute(
                "DELETE FROM sessions WHERE created_at < datetime('now', ?)", 
                (f'-{days} days',)
            )
            conn.commit()

    # Clean up references to old get_history if needed. 
    # The original get_history returned objects with question/answer/context keys.
    # So we should alias get_history to get_history_formatted.
    get_history = get_history_formatted
