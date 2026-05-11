import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import uuid
import hashlib

# Initialize Qdrant Client (Docker connection)
client = QdrantClient(url="http://localhost:6333")

# Initialize SentenceTransformer
MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME)
EMBEDDING_SIZE = embedding_model.get_sentence_embedding_dimension()

COLLECTION_NAME = "ats_resumes"

def init_db():
    """Create the collection if it doesn't exist."""
    collections = client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
        )

def get_embedding(text: str) -> List[float]:
    """Generate embedding for the given text."""
    return embedding_model.encode(text).tolist()

def upsert_resume(resume_data_dict: Dict[str, Any], raw_text: str = ""):
    """Insert or update a resume in Qdrant."""
    # Create a dense representation for highly specific semantic search
    projects = resume_data_dict.get('Projects', [])
    experiences = resume_data_dict.get('Experiences', [])
    
    projects_text = " ".join([f"{p.get('Name', '')} using {', '.join(p.get('Technologies_Used', []))}." for p in projects])
    experiences_text = " ".join([f"{e.get('Role', '')} at {e.get('Company', '')}." for e in experiences])
    
    text_to_embed = (
        f"Role: {resume_data_dict.get('Primary_Role', '')}. "
        f"Languages: {', '.join(resume_data_dict.get('Programming_Languages', []))}. "
        f"Frameworks/Tools: {', '.join(resume_data_dict.get('Frameworks_Tools', []))}. "
        f"Experience: {resume_data_dict.get('Years_of_Experience', 0)} years. "
        f"Work History: {experiences_text} "
        f"Projects: {projects_text} "
        f"Total Projects Count: {resume_data_dict.get('Total_Projects_Count', len(projects))}."
    )
    
    vector = get_embedding(text_to_embed)
    
    # Generate a deterministic UUID based on Email, Name, or a hash of the raw text
    unique_string = resume_data_dict.get('Email') or resume_data_dict.get('Name')
    if not unique_string:
        unique_string = hashlib.sha256(raw_text.encode('utf-8')).hexdigest()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, unique_string))
    
    payload = resume_data_dict.copy()
    payload["Raw_Text"] = raw_text
    
    point = PointStruct(
        id=point_id,
        vector=vector,
        payload=payload
    )
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )
    return point_id

def search_resumes(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for matching resumes."""
    query_vector = get_embedding(query)
    
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    
    results = []
    for hit in search_result.points:
        payload = hit.payload.copy() if hit.payload else {}
        payload["_score"] = hit.score
        results.append(payload)
        
    return results
