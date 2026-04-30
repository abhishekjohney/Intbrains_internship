from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid
import uvicorn
import os

# --- UNIVERSAL FILE PARSERS ---
import PyPDF2
import docx
import pandas as pd

app = FastAPI(title="Local RAG: Docker & Universal Enterprise Engine")

# ==========================================
# 1. VECTOR DATABASE SETUP (Docker)
# ==========================================
print("Connecting to Qdrant Docker Server...")
# Connects to the Qdrant instance running in your Docker container
qdrant = QdrantClient(url="http://localhost:6333") 
COLLECTION_NAME = "private_knowledge"

try:
    qdrant.get_collection(COLLECTION_NAME)
except Exception:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

# ==========================================
# 2. UNIVERSAL TEXT EXTRACTOR
# ==========================================
def extract_text_from_file(file_path: str) -> str:
    """Reads a file path and extracts raw text based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                
        elif ext == ".pdf":
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n\n"
                        
        elif ext == ".docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n\n"
                    
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
            text = df.to_string(index=False)
            
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)
            
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")

    return text

# ==========================================
# 3. PYDANTIC SHIELDS
# ==========================================
class FilePathPayload(BaseModel):
    file_path: str

class ChatPayload(BaseModel):
    question: str

# ==========================================
# 4. ROUTE: INGEST ANY FILE
# ==========================================
@app.post("/api/learn-from-path")
def learn_from_local_path(payload: FilePathPayload):
    if not os.path.exists(payload.file_path):
        raise HTTPException(status_code=404, detail=f"Could not find file at: {payload.file_path}")

    try:
        full_text = extract_text_from_file(payload.file_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in the file.")

    # Chunking: Split by paragraphs
    paragraphs = full_text.split("\n\n")
    saved_chunks = 0

    for chunk in paragraphs:
        clean_chunk = chunk.strip()
        if len(clean_chunk) > 10: 
            
            # Vectorize using Nomic
            response = ollama.embeddings(model="nomic-embed-text", prompt=clean_chunk)
            vector_math = response["embedding"]

            # Save to Qdrant
            doc_id = str(uuid.uuid4())
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(
                    id=doc_id, 
                    vector=vector_math, 
                    payload={
                        "source_file": payload.file_path,
                        "source_text": clean_chunk
                    }
                )]
            )
            saved_chunks += 1

    return {
        "status": "success", 
        "file_processed": payload.file_path,
        "chunks_memorized": saved_chunks
    }

# ==========================================
# 5. ROUTE: CHAT WITH THE DATA
# ==========================================
@app.post("/api/chat")
def ask_ai(payload: ChatPayload):
    # 1. Convert question to vector
    response = ollama.embeddings(model="nomic-embed-text", prompt=payload.question)
    query_vector = response["embedding"]

    # 2. Semantic Search (Qdrant 1.16+ syntax)
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3
    )

    # 3. Extract text from search_results.points
    retrieved_facts = [hit.payload["source_text"] for hit in search_results.points]
    context_string = "\n---\n".join(retrieved_facts)

    # 4. Build prompt
    system_prompt = f"""
    You are an internal company assistant. Answer using ONLY the provided context. 
    If the context doesn't contain the answer, say "I do not know."
    
    Context:
    {context_string}
    
    Question: {payload.question}
    """

    # 5. Generate Answer via Llama 3
    llm_response = ollama.generate(model="llama3", prompt=system_prompt)

    # Clean response: No raw database chunks shown
    return {
        "user_question": payload.question,
        "ai_answer": llm_response["response"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)