from src.app.services.session_service import SessionService
from src.app.core.database import get_db
import uuid

def test_mongo_connection():
    try:
        db = get_db()
        # Ping command to check connection
        db.command('ping')
        print("SUCCESS: Connected to MongoDB.")
    except Exception as e:
        print(f"FAILURE: Could not connect to MongoDB. Error: {e}")
        return

def test_session_operations():
    try:
        print("\nTesting Session Operations...")
        
        # 1. Create Session
        session_id = SessionService.create_session()
        print(f"Created session: {session_id}")
        
        # 2. Add Turn
        question = "Hello, how are you?"
        answer = "I am a bot, functioning within normal parameters."
        SessionService.add_turn(session_id, question, answer)
        print("Added a turn to the session.")
        
        # 3. Get History
        history = SessionService.get_history(session_id)
        print(f"Retrieved history: {len(history)} items")
        if len(history) > 0:
            print(f"Sample item: {history[0]}")
        else:
            print("FAILURE: History is empty.")
            
        # 4. Cleanup
        SessionService.delete_session(session_id)
        print("Deleted session.")
        
        # Verify deletion
        history_after = SessionService.get_history(session_id)
        if len(history_after) == 0:
            print("SUCCESS: Session verified deleted.")
        else:
            print("FAILURE: Session data still exists.")

    except Exception as e:
        print(f"FAILURE: Error during session operations: {e}")

if __name__ == "__main__":
    test_mongo_connection()
    test_session_operations()
