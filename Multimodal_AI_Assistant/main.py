import os

# Fix Windows terminal encoding — prevents UnicodeEncodeError on progress bars
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only mode

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import ollama
import uuid
import uvicorn
import base64
from pathlib import Path

# ── Faster-Whisper (audio transcription) ────────────────────────────────────
# Uses CTranslate2 backend: runs `small` model at same CPU speed as openai-
# whisper `base`, giving much better accuracy for accents and background noise.
whisper_model = None
try:
    from faster_whisper import WhisperModel
    whisper_available = True
except Exception as e:
    print(f"Warning: faster-whisper not available: {e}")
    whisper_available = False

# ── Docling (document + image extraction) ────────────────────────────────────
docling_converter = None
try:
    from docling.document_converter import DocumentConverter
    docling_available = True
except Exception as e:
    print(f"Warning: Docling not available: {e}")
    docling_available = False

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Multimodal RAG Assistant — Powered by Docling")

# ── Qdrant vector database ────────────────────────────────────────────────────
print("Connecting to Qdrant Vector Database...")
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "multimodal_brain"

try:
    qdrant.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' already exists.")
except Exception:
    print(f"Creating collection '{COLLECTION_NAME}'...")
    try:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as create_err:
        print(f"Error creating collection: {create_err}")

# ── Load Faster-Whisper model ────────────────────────────────────────────────
# Model options (accuracy vs speed tradeoff on CPU):
#   tiny   ~39M  — fastest, lower accuracy
#   base   ~74M  — previous setting
#   small  ~244M — recommended: best accuracy/speed balance (default)
#   medium ~769M — very accurate, slower (~3-4 min per audio on CPU)
WHISPER_MODEL_SIZE = "small"  # ← change this to tune accuracy vs speed
print(f"Loading faster-whisper ({WHISPER_MODEL_SIZE} model)...")
if whisper_available:
    try:
        # compute_type="int8" quantizes weights → 3-4x faster on CPU, same accuracy
        whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        print(f"[OK] faster-whisper ({WHISPER_MODEL_SIZE}) loaded successfully")
    except Exception as e:
        print(f"[FAIL] faster-whisper failed to load: {e}")
        whisper_available = False
else:
    print("[FAIL] faster-whisper not available")

# ── Load Docling converter ────────────────────────────────────────────────────
print("Loading Docling DocumentConverter...")
if docling_available:
    try:
        docling_converter = DocumentConverter()
        print("[OK] Docling converter ready")
    except Exception as e:
        print(f"[FAIL] Docling failed to initialize: {e}")
        docling_available = False
else:
    print("[FAIL] Docling not available")

# ── Supported file types ──────────────────────────────────────────────────────
IMAGE_EXTS   = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'gif'}
AUDIO_EXTS   = {'mp3', 'wav', 'm4a'}
DOCLING_EXTS = {'pdf', 'docx', 'pptx', 'xlsx', 'xls', 'csv', 'html', 'txt'} | IMAGE_EXTS


# ── Request schemas ───────────────────────────────────────────────────────────
class FilePayload(BaseModel):
    file_path: str

class QuestionPayload(BaseModel):
    question: str

class AudioQuestionPayload(BaseModel):
    audio_file_path: str


# ─────────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_with_docling(file_path: str) -> str:
    """
    Use Docling to extract text from any supported file.
    Returns clean Markdown — preserves tables, headings, lists, and reading order.
    Supports: PDF, DOCX, PPTX, XLSX, HTML, TXT, JPG, PNG, BMP, TIFF, WEBP.
    """
    if not docling_available or docling_converter is None:
        raise RuntimeError("Docling is not available.")

    print(f"[Docling] Converting: {file_path}")
    result = docling_converter.convert(file_path)
    markdown = result.document.export_to_markdown()
    print(f"[Docling] Extracted {len(markdown)} characters")
    return markdown.strip()


def extract_with_llava_fallback(image_path: str) -> str:
    """
    LLaVA vision fallback for images where Docling finds little/no text.
    Used for person photos, artwork, charts — anything that is visual rather
    than a text document.
    """
    print(f"[LLaVA] Falling back to vision model for: {image_path}")
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    prompt = """Describe this image in detail. If it contains any text, numbers, 
tables or structured data, copy them exactly as shown. 
If it is a photo of a person or scene, describe what you see clearly."""

    response = ollama.generate(
        model="llava",
        prompt=prompt,
        images=[image_data],
        stream=False,
    )
    text = response["response"].strip()
    print(f"[LLaVA] Extracted {len(text)} characters")
    return text


def extract_text(file_path: str) -> str:
    """
    Main extraction router.
    - Audio  → Whisper speech-to-text
    - Everything else → Docling (PDF, DOCX, PPTX, XLSX, TXT, images)
    - Images with sparse Docling output → LLaVA vision fallback
    """
    ext = Path(file_path).suffix.lower().lstrip('.')

    # ── Audio: transcribe with faster-whisper ─────────────────────────────────
    if ext in AUDIO_EXTS:
        if not whisper_available or whisper_model is None:
            raise ValueError("Audio transcription unavailable (faster-whisper not loaded).")
        print(f"[Whisper] Transcribing: {file_path}")
        # vad_filter=True  → Silero VAD removes silence & background noise automatically
        # beam_size=5      → wider beam search = more accurate (default is 5)
        # language='en'    → skip language detection, assume English (faster + more accurate)
        segments, info = whisper_model.transcribe(
            file_path,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        print(f"[Whisper] Transcribed {len(text)} chars | detected lang: {info.language} ({info.language_probability:.0%})")
        return text

    # ── All documents and images: try Docling first ───────────────────────────
    if ext in DOCLING_EXTS:
        try:
            docling_text = extract_with_docling(file_path)

            # For image files: if Docling returns sparse content (person photo,
            # chart, artwork) fall back to LLaVA for a visual description.
            if ext in IMAGE_EXTS and len(docling_text) < 100:
                print(f"[Info] Docling found < 100 chars in image — switching to LLaVA.")
                return extract_with_llava_fallback(file_path)

            return docling_text

        except Exception as e:
            # If Docling fails on an image, try LLaVA before giving up
            if ext in IMAGE_EXTS:
                print(f"[Warning] Docling failed ({e}), trying LLaVA fallback.")
                return extract_with_llava_fallback(file_path)
            raise

    raise ValueError(f"Unsupported file type: .{ext}")


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200):
    """
    Split text into overlapping chunks (generator — memory efficient).
    The 200-char overlap ensures no sentence is lost at a chunk boundary.
    """
    start = 0
    text_len = len(text)

    if text_len == 0:
        yield text
        return

    while start < text_len:
        end = min(start + chunk_size, text_len)
        yield text[start:end]
        if end == text_len:
            break
        start = end - overlap


# ─────────────────────────────────────────────────────────────────────────────
# SHARED RAG HELPER — embed question + search + generate answer
# ─────────────────────────────────────────────────────────────────────────────

def rag_answer(question: str) -> str:
    """Embed question → search Qdrant → build context → Llama 3 answer."""

    # 1. Vectorize the question
    embed_resp = ollama.embeddings(model="nomic-embed-text", prompt=question)
    query_vector = embed_resp["embedding"]

    # 2. Find top-3 most semantically similar stored chunks
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3,
    )

    # 3. Build context string with source labels
    context_parts = [
        f"[Source: {hit.payload['source_file']}]\n{hit.payload['content']}"
        for hit in search_results.points
    ]
    context_string = "\n---\n".join(context_parts)

    # 4. Ask Llama 3 — constrain it to ONLY the retrieved context
    system_prompt = f"""You are a highly intelligent corporate AI assistant.
Answer the user's question using ONLY the provided database context.
Always state which [Source] file the answer comes from.
If the answer is not in the context, say: "I do not have enough information to answer that."

Database Context:
{context_string}

User Question: {question}"""

    llm_response = ollama.generate(model="llama3", prompt=system_prompt)
    return llm_response["response"]


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/learn")
def learn_document(payload: FilePayload):
    """Ingest any file — extract text, chunk, embed, store in Qdrant."""

    if not os.path.exists(payload.file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    ext = Path(payload.file_path).suffix.lower().lstrip('.')
    filename = os.path.basename(payload.file_path)

    # Extract text
    try:
        extracted_text = extract_text(payload.file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(extracted_text) < 5:
        raise HTTPException(status_code=400, detail="File was empty or unreadable.")

    # Chunk → embed → store in Qdrant
    print(f"Chunking and embedding '{filename}'...")
    points_to_upsert = []
    chunk_count = 0

    for i, chunk in enumerate(chunk_text(extracted_text)):
        embed_resp = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_chunk_{i}"))

        points_to_upsert.append(PointStruct(
            id=chunk_id,
            vector=embed_resp["embedding"],
            payload={
                "source_file": filename,
                "content": chunk,
                "chunk_index": i,
                "type": ext,
            }
        ))
        chunk_count += 1

        # Batch-write every 10 chunks to avoid memory buildup
        if len(points_to_upsert) >= 10:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points_to_upsert)
            points_to_upsert = []

    if points_to_upsert:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points_to_upsert)

    print(f"[OK] '{filename}' memorized in {chunk_count} chunks.")

    result = {
        "status": "success",
        "memorized": filename,
        "chunks": chunk_count,
        "file_type": ext,
    }
    # For image files include the extracted content so caller can verify
    if ext in IMAGE_EXTS:
        result["extracted_content"] = extracted_text

    return result


@app.post("/api/ask")
def ask_assistant(payload: QuestionPayload):
    """Answer a text question from stored knowledge."""
    answer = rag_answer(payload.question)
    return {"question": payload.question, "answer": answer}


@app.post("/api/ask-audio")
def ask_assistant_audio(payload: AudioQuestionPayload):
    """Transcribe audio question with Whisper, then answer from stored knowledge."""

    if not os.path.exists(payload.audio_file_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    if not whisper_available or whisper_model is None:
        raise HTTPException(
            status_code=503,
            detail="Audio transcription unavailable (Whisper not loaded).",
        )

    try:
        print(f"[Whisper] Transcribing: {payload.audio_file_path}")
        segments, info = whisper_model.transcribe(
            payload.audio_file_path,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        question = " ".join(seg.text.strip() for seg in segments).strip()
        print(f"[Whisper] Heard: {question} | lang: {info.language} ({info.language_probability:.0%})")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcription failed: {e}")

    answer = rag_answer(question)

    return {
        "original_audio_file": os.path.basename(payload.audio_file_path),
        "transcribed_question": question,
        "answer": answer,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
