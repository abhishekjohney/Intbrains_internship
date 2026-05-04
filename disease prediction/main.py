from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid
import uvicorn
import os

app = FastAPI(title="Local RAG: Medical Diagnostic Assistant")

# ==========================================
# 1. VECTOR DATABASE SETUP
# ==========================================
print("Connecting to Qdrant Docker Server...")
qdrant = QdrantClient(url="http://localhost:6333") 

# Using a dedicated medical collection
COLLECTION_NAME = "medical_knowledge"

try:
    qdrant.get_collection(COLLECTION_NAME)
except Exception:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

# ==========================================
# 2. PYDANTIC SHIELDS
# ==========================================
class FilePathPayload(BaseModel):
    file_path: str

class SymptomPayload(BaseModel):
    symptoms: str

# ==========================================
# 3. ROUTE: INGEST DISEASE DATA
# ==========================================
@app.post("/api/learn-disease")
def learn_disease(payload: FilePathPayload):
    if not os.path.exists(payload.file_path):
        raise HTTPException(status_code=404, detail=f"Could not find file at: {payload.file_path}")

    # Enforce .txt files for clean formatting
    if not payload.file_path.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Please provide a .txt file formatted for diseases.")

    try:
        with open(payload.file_path, "r", encoding="utf-8") as file:
            full_text = file.read().strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if len(full_text) < 10:
        raise HTTPException(status_code=400, detail="File is too short. Please include disease name and symptoms.")

    # Vectorize the ENTIRE file as one single chunk
    response = ollama.embeddings(model="nomic-embed-text", prompt=full_text)
    vector_math = response["embedding"]

    # --- ENTERPRISE FIX: DETERMINISTIC IDs ---
    # Generate a consistent ID based on the file's name to prevent duplicates!
    filename = os.path.basename(payload.file_path)
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))
    
    # Save to Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id=doc_id, 
            vector=vector_math, 
            payload={
                "source_file": payload.file_path,
                "disease_data": full_text
            }
        )]
    )

    return {
        "status": "success", 
        "disease_memorized": filename,
        "database_id": doc_id
    }

# ==========================================
# 4. ROUTE: DIAGNOSE SYMPTOMS
# ==========================================
@app.post("/api/diagnose")
def diagnose_symptoms(payload: SymptomPayload):
    # 1. Convert user's symptoms to a math vector
    response = ollama.embeddings(model="nomic-embed-text", prompt=payload.symptoms)
    query_vector = response["embedding"]

    # 2. Find the top 2 closest matching diseases in the database
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=2
    )

    # 3. Extract the disease text
    retrieved_facts = [hit.payload["disease_data"] for hit in search_results.points]
    context_string = "\n---\n".join(retrieved_facts)

    # 4. The Specialized Medical Prompt
    system_prompt = f"""
    You are a highly analytical Medical AI Assistant. 
    Your goal is to match the user's reported symptoms to the most likely diseases using ONLY the provided medical database context.

    Follow these rules strictly:
    1. List the potential diseases that match the user's symptoms.
    2. Explain WHY you chose those diseases based on the matching symptoms.
    3. If the symptoms do not clearly match anything in the context, say "I cannot find a matching disease in my database."
    4. ALWAYS end your response with this exact disclaimer: "Disclaimer: I am an AI, not a doctor. Please consult a healthcare professional for actual medical advice."
    
    Database Context:
    {context_string}
    
    User's Symptoms: {payload.symptoms}
    """

    # 5. Generate Diagnosis via Llama 3
    llm_response = ollama.generate(model="llama3", prompt=system_prompt)

    return {
        "reported_symptoms": payload.symptoms,
        "ai_diagnosis": llm_response["response"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)