import os
from qdrant_client import QdrantClient
from app.db import COLLECTION_NAME, init_db, client

def clean():
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' deleted.")
    except Exception as e:
        print(f"Could not delete collection: {e}")
    
    # Re-initialize
    init_db()
    print(f"Collection '{COLLECTION_NAME}' created.")

if __name__ == "__main__":
    clean()
