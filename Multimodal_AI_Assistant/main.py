import os
import sys

# Disable oneDNN and PIR executor to fix PaddleOCR compatibility issue
os.environ['PADDLE_WITH_ONEDNN'] = '0'
os.environ['PADDLE_NO_ONEDNN'] = '1'
os.environ['PADDLE_SKIP_OPINFO_DESC'] = '1'
os.environ['PADDLE_DEVICE'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['FLAGS_enable_pir_api'] = '0'  # Disable PIR executor to fix ConvertPirAttribute error

# Import paddle early and configure it
try:
    import paddle
    paddle.device.set_device('cpu')
except:
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import ollama

# Windows-specific fix for PyTorch shm.dll loading issue
whisper_model = None
try:
    import torch
    torch.multiprocessing.set_sharing_strategy('file_system')
    import whisper
    whisper_available = True
except Exception as e:
    print(f"Warning: Whisper/PyTorch not available: {e}")
    whisper_available = False
import uuid
import os
import uvicorn
import base64

import fitz  
import pandas as pd 
import docx
from PIL import Image 

# Try to load PaddleOCR, but make it optional as it requires PyTorch
paddle_ocr = None
try:
    from paddleocr import PaddleOCR
    paddle_available = True
except Exception as e:
    print(f"Warning: PaddleOCR not available: {e}")
    paddle_available = False

app = FastAPI(title="Multimodal RAG Assistant (Audio, PDF, Excel, Word, Text)")

print("Connecting to Qdrant Vector Database...")
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "multimodal_brain"

try:
    qdrant.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' already exists.")
except Exception as e:
    print(f"Creating collection '{COLLECTION_NAME}'...")
    try:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as create_err:
        print(f"Error creating collection: {create_err}")

print("Waking up Whisper AI Model...")
if whisper_available:
    try:
        whisper_model = whisper.load_model("base")
        print("✓ Whisper model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load Whisper model: {e}")
        whisper_available = False
else:
    print("✗ Whisper not available (PyTorch issue)")

print("Waking up PaddleOCR Model...")
if paddle_available:
    try:
        paddle_ocr = PaddleOCR(use_textline_orientation=True, lang="en")
        print("✓ PaddleOCR model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load PaddleOCR model: {e}")
        paddle_available = False
        paddle_ocr = None
else:
    print("✗ PaddleOCR not available (PyTorch issue)")


class FilePayload(BaseModel):
    file_path: str

class QuestionPayload(BaseModel):
    question: str

class AudioQuestionPayload(BaseModel):
    audio_file_path: str

# 3. THE TEXT EXTRACTION ROUTER
def extract_text_from_file(file_path: str) -> str:
    """Routes the file to the correct parser based on its extension."""
    ext = file_path.lower().split('.')[-1]

    try:
        if ext in ['mp3', 'wav', 'm4a']:
            if not whisper_available:
                raise ValueError(f"Audio transcription not available (Whisper/PyTorch issue)")
            print(f"Listening to audio file: {file_path}")
            return whisper_model.transcribe(file_path)["text"].strip()

        elif ext == 'pdf':
            print(f"Reading PDF: {file_path}")
            text = ""
            with fitz.open(file_path) as pdf_doc:
                for page in pdf_doc:
                    text += page.get_text() + "\n"
            return text.strip()

        elif ext == 'docx':
            print(f"Reading Word document: {file_path}")
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs]).strip()

        elif ext in ['xlsx', 'xls', 'csv']:
            print(f"Analyzing Spreadsheet: {file_path}")
            # Converts the spreadsheet grid into a readable string table
            df = pd.read_excel(file_path) if ext != 'csv' else pd.read_csv(file_path)
            return df.to_string()

        elif ext == 'txt':
            print(f"Reading Text file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()

        else:
            raise ValueError(f"Unsupported file type: .{ext}")

    except Exception as e:
        raise Exception(f"Failed to extract text from .{ext} file: {str(e)}")

# 4. CHUNK LARGE TEXTS
def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200):
    """Split text into overlapping chunks (generator to save memory)."""
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        yield text[start:end]
        
        start = end - overlap
        
        if end == text_len:
            break
    
    # If text is empty, yield it anyway
    if text_len == 0:
        yield text

# 4A. PADDLE OCR FUNCTION
def extract_text_from_image_ocr(image_path: str) -> str:
    """Extract text from image using PaddleOCR (strong for invoices and documents)."""
    try:
        if not paddle_available or paddle_ocr is None:
            print("PaddleOCR is unavailable, skipping OCR and using vision fallback.")
            return ""

        print(f"Extracting text with PaddleOCR: {image_path}")
        image = Image.open(image_path)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Try using ocr() without cls parameter (simpler inference)
        try:
            ocr_result = paddle_ocr.ocr(image_path)
        except Exception as e:
            print(f"ocr() failed: {e}, trying with use_angle_cls=False...")
            return ""

        # Flatten results into a readable block of text
        lines = []
        for page in ocr_result:
            for item in page:
                text = item[1][0]
                if text:
                    lines.append(text)

        extracted_text = "\n".join(lines).strip()
        print(f"OCR extracted {len(extracted_text)} characters")
        return extracted_text

    except Exception as e:
        print(f"PaddleOCR failed: {str(e)}")
        return ""

# 4A2. STRUCTURED DATA EXTRACTION FUNCTION
def extract_structured_data_from_image(image_path: str) -> str:
    """Extract structured data from document images with intelligent processing."""
    try:
        print(f"Extracting structured data from: {image_path}")
        
        # Step 1: Try Tesseract OCR
        ocr_text = extract_text_from_image_ocr(image_path)
        
        if len(ocr_text) > 100:
            print("✓ OCR successfully extracted structured text")
            return ocr_text
        
        # Step 2: If OCR insufficient, use LLaVA with optimized prompt for structured extraction
        print("✓ Using LLaVA for structured data extraction")
        with open(image_path, "rb") as image_file:
            image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")
        
        # Specialized prompt for extracting structured data from invoices and financial documents
        structured_prompt = """Extract all text and data from this document. Include:
- Document type and headers
- All company/person names and addresses
- All dates and numbers with their context
- All line items with descriptions, quantities, prices, and totals
- Any tables - show all rows and columns exactly as they appear
- Totals, subtotals, taxes, amounts due
- Any other visible information

Do NOT summarize or use templates. Copy exact text and numbers as shown."""
        
        response = ollama.generate(
            model="llava",
            prompt=structured_prompt,
            images=[image_data],
            stream=False
        )
        
        extracted_text = response["response"].strip()
        print(f"LLaVA extracted {len(extracted_text)} characters")
        return extracted_text
        
    except Exception as e:
        raise Exception(f"Failed to extract structured data: {str(e)}")

# 4B. VISION HELPER FUNCTION (HYBRID APPROACH WITH STRUCTURED EXTRACTION)
def analyze_image_with_vision(image_path: str, question: str = "") -> str:
    """
    Hybrid approach for image analysis:
    1. Try Tesseract OCR with preprocessing (best for text-heavy documents)
    2. Extract structured data using LLaVA (for complex/visual documents)
    3. Returns precise text instead of generic descriptions
    """
    try:
        # Use structured data extraction instead of generic vision analysis
        extracted_text = extract_structured_data_from_image(image_path)
        
        if extracted_text:
            return extracted_text
        else:
            raise Exception("Failed to extract any data from image")
    
    except Exception as e:
        raise Exception(f"Failed to analyze image: {str(e)}")

# 5. ROUTE: LEARN ANY DOCUMENT OR IMAGE
@app.post("/api/learn")
def learn_document(payload: FilePayload):
    if not os.path.exists(payload.file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    ext = payload.file_path.lower().split('.')[-1]
    
    # Handle image files with vision model
    if ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'gif']:
        try:
            print(f"Analyzing image with vision model: {payload.file_path}")
            extracted_text = analyze_image_with_vision(
                payload.file_path,
                ""  # Using structured extraction, prompt not needed
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Handle other file types (PDF, Word, Audio, etc.)
        try:
            extracted_text = extract_text_from_file(payload.file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if len(extracted_text) < 5:
        raise HTTPException(status_code=400, detail="File was empty or unreadable.")

    # 2. Process chunks one at a time to save memory
    print("Chunking and processing document...")
    filename = os.path.basename(payload.file_path)
    points_to_upsert = []
    chunk_count = 0

    print("Converting data to math vectors...")
    for i, chunk in enumerate(chunk_text(extracted_text, chunk_size=2000, overlap=200)):
        response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
        vector_math = response["embedding"]
        
        # Create unique ID for each chunk
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_chunk_{i}"))
        
        points_to_upsert.append(PointStruct(
            id=chunk_id,
            vector=vector_math,
            payload={
                "source_file": filename,
                "content": chunk,
                "chunk_index": i,
                "type": ext
            }
        ))
        chunk_count += 1
        
        # Upsert in batches to avoid memory buildup
        if len(points_to_upsert) >= 10:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points_to_upsert
            )
            points_to_upsert = []
    
    # Upsert remaining chunks
    if points_to_upsert:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upsert
        )

    # Return image description if it's an image file
    response = {"status": "success", "memorized": filename, "chunks": chunk_count, "file_type": ext}
    if ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'gif']:
        response["image_description"] = extracted_text
    
    return response

# 6. ROUTE: ASK THE MULTIMODAL BRAIN
@app.post("/api/ask")
def ask_assistant(payload: QuestionPayload):
    # 1. Vectorize the User's Question
    response = ollama.embeddings(model="nomic-embed-text", prompt=payload.question)
    query_vector = response["embedding"]

    # 2. Search Qdrant for the top 3 most relevant documents
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3
    )

    # 3. Build the Context String (Include the file names so the AI knows its sources!)
    context_chunks = []
    for hit in search_results.points:
        source = hit.payload["source_file"]
        content = hit.payload["content"]
        context_chunks.append(f"[Source: {source}]\n{content}\n")
    
    context_string = "\n---\n".join(context_chunks)

    # 4. Generate the Answer with Llama 3
    system_prompt = f"""
    You are a highly intelligent corporate AI assistant. 
    Answer the user's question using ONLY the provided database context.
    If the context contains the answer, explicitly state which [Source] file you got the information from.
    If the answer is not in the context, say "I do not have enough information to answer that."

    Database Context:
    {context_string}
    
    User Question: {payload.question}
    """

    llm_response = ollama.generate(model="llama3", prompt=system_prompt)

    return {
        "question": payload.question,
        "answer": llm_response["response"]
    }

# 7. ROUTE: ASK THE MULTIMODAL BRAIN WITH AUDIO
@app.post("/api/ask-audio")
def ask_assistant_audio(payload: AudioQuestionPayload):
    """Ask a question using audio input. Audio is transcribed to text first."""
    
    # 1. Check if audio file exists
    if not os.path.exists(payload.audio_file_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")
    
    # 1.5. Check if Whisper is available
    if not whisper_available:
        raise HTTPException(status_code=503, detail="Audio transcription is not available due to PyTorch/Whisper issues.")
    
    # 2. Transcribe audio to text using Whisper
    try:
        print(f"Transcribing audio: {payload.audio_file_path}")
        transcribed_text = whisper_model.transcribe(payload.audio_file_path)["text"].strip()
        print(f"Transcribed question: {transcribed_text}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to transcribe audio: {str(e)}")
    
    # 3. Vectorize the transcribed question
    response = ollama.embeddings(model="nomic-embed-text", prompt=transcribed_text)
    query_vector = response["embedding"]

    # 4. Search Qdrant for the top 3 most relevant documents
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3
    )

    # 5. Build the Context String
    context_chunks = []
    for hit in search_results.points:
        source = hit.payload["source_file"]
        content = hit.payload["content"]
        context_chunks.append(f"[Source: {source}]\n{content}\n")
    
    context_string = "\n---\n".join(context_chunks)

    # 6. Generate the Answer with Llama 3
    system_prompt = f"""
    You are a highly intelligent corporate AI assistant. 
    Answer the user's question using ONLY the provided database context.
    If the context contains the answer, explicitly state which [Source] file you got the information from.
    If the answer is not in the context, say "I do not have enough information to answer that."

    Database Context:
    {context_string}
    
    User Question: {transcribed_text}
    """

    llm_response = ollama.generate(model="llama3", prompt=system_prompt)

    return {
        "original_audio_file": os.path.basename(payload.audio_file_path),
        "transcribed_question": transcribed_text,
        "answer": llm_response["response"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)