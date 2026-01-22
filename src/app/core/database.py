from pymongo import MongoClient
from pymongo.database import Database
from .config import get_settings

def get_db_client() -> MongoClient:
    """Get a MongoDB client instance."""
    settings = get_settings()
    return MongoClient(settings.database_url)

def get_db() -> Database:
    """Get the MongoDB database instance."""
    client = get_db_client()
    # The database name is usually part of the connection string or we can pick a default
    # For MongoDB Atlas, the default database is often in the path, but let's assume 'chat_history' if not specified
    # or get it from the connection itself if possible. 
    # However, pymongo's get_default_database() is useful if the URI has a db name.
    
    try:
        return client.get_default_database()
    except Exception:
        # Fallback if no database specified in URI
        return client["chat_history"]
