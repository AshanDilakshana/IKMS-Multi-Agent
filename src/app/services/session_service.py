import uuid
from typing import Dict, List, Optional
from datetime import datetime
from ..core.database import get_db

class SessionService:

    @classmethod
    def get_history(cls, session_id: str) -> List[dict]:
        db = get_db()
        # Find messages for the session, sorted by timestamp
        messages = list(db.messages.find({"session_id": session_id}).sort("timestamp", 1))
        
        return [
            {
                "question": msg.get("content", "") if msg.get("role") == "user" else "",
                "answer": msg.get("content", "") if msg.get("role") == "assistant" else "",
                "context": msg.get("context"),
                "timestamp": msg.get("timestamp").isoformat() if msg.get("timestamp") else None
            }
            for msg in messages
        ]

    @classmethod
    def get_history_formatted(cls, session_id: str) -> List[dict]:
        """
        Reconstructs turns from flat message list. 
        """
        db = get_db()
        # Ensure we sort by a consistent order. MongoDB's default _id is chronological roughly, but explicit timestamp is better.
        messages = list(db.messages.find({"session_id": session_id}).sort("timestamp", 1))

        history = []
        current_turn = {}
        turn_index = 1
        
        for msg in messages:
            if msg.get("role") == "user":
                current_turn = {
                    "question": msg.get("content"),
                    "timestamp": msg.get("timestamp").isoformat() if msg.get("timestamp") else None,
                    "turn": turn_index
                }
            elif msg.get("role") == "assistant":
                if "question" in current_turn:
                    current_turn["answer"] = msg.get("content")
                    current_turn["context"] = msg.get("context")
                    # Ensure all fields are present
                    if "turn" not in current_turn:
                         current_turn["turn"] = turn_index
                    history.append(current_turn)
                    current_turn = {}
                    turn_index += 1
                else:
                    # Assistant message without preceding user message
                    pass
        
        return history

    @classmethod
    def add_turn(cls, session_id: str, question: str, answer: str, context: Optional[str] = None):
        db = get_db()
        timestamp = datetime.now()

        # 1. Update session (upsert to ensure it exists)
        # We also want to update the title if it's the default "New Chat" and we have a question
        
        # Check if session exists first to decide on title update
        session = db.sessions.find_one({"id": session_id})
        
        if not session:
            # Create new session if not exists (should be rare if create_session called, but good fallback)
            db.sessions.insert_one({
                "id": session_id,
                "title": question[:50], # Set title from first question
                "created_at": timestamp
            })
        elif session.get("title") == "New Chat":
            # Update title
            db.sessions.update_one(
                {"id": session_id},
                {"$set": {"title": question[:50]}}
            )

        # 2. Add User Message
        db.messages.insert_one({
            "session_id": session_id,
            "role": "user",
            "content": question,
            "timestamp": timestamp
        })

        # 3. Add Assistant Message
        db.messages.insert_one({
            "session_id": session_id,
            "role": "assistant",
            "content": answer,
            "context": context,
            "timestamp": timestamp
        })
    
    @classmethod
    def create_session(cls) -> str:
        session_id = str(uuid.uuid4())
        db = get_db()
        db.sessions.insert_one({
            "id": session_id,
            "title": "New Chat",
            "created_at": datetime.now()
        })
        return session_id

    @classmethod
    def list_sessions(cls) -> List[dict]:
        db = get_db()
        # Return all sessions sorted by created_at descending
        sessions = list(db.sessions.find({}, {"_id": 0}).sort("created_at", -1))
        # No serialization needed as pymongo returns dicts (but we might need to handle datetime if fastAPI doesn't)
        return sessions

    @classmethod
    def delete_session(cls, session_id: str):
        db = get_db()
        db.messages.delete_many({"session_id": session_id})
        db.sessions.delete_one({"id": session_id})

    @classmethod
    def cleanup_old_sessions(cls, days: int = 7):
        # This is a bit more complex in Mongo without direct SQL intervals, 
        # but we can calculate the cutoff date in Python.
        pass
        # Skipping implementation for now as it wasn't strictly requested and might need index TTL.
        # If needed:
        # cutoff = datetime.now() - timedelta(days=days)
        # db.sessions.delete_many({"created_at": {"$lt": cutoff}})
        # And cleanup messages...

    # Clean up references to old get_history if needed. 
    # The original get_history returned objects with question/answer/context keys.
    # So we should alias get_history to get_history_formatted.
    get_history = get_history_formatted
