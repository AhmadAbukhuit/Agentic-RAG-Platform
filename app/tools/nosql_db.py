import uuid
from typing import Any

from core.config import settings
from pymongo import MongoClient


def get_mongo_collection():
    """Returns the MongoDB collection for storing raw document text."""
    client = MongoClient(settings.mongodb_url, serverSelectionTimeoutMS=2000)
    db = client[settings.mongodb_database]
    return db[settings.mongodb_collection]


def insert_into_mongodb(raw_text: str, metadata: dict[str, Any]) -> str:
    """Saves raw unchunked document text and metadata to MongoDB."""
    try:
        collection = get_mongo_collection()
        doc = {
            "raw_text": raw_text,
            "metadata": metadata,
            "inserted_at": uuid.uuid4().hex
        }
        result = collection.insert_one(doc)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Warning: MongoDB insertion failed ({e}). Generating fallback document ID.")
        return f"fallback-doc-{uuid.uuid4().hex[:12]}"
