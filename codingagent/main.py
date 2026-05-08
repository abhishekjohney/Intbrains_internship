import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json

app = FastAPI(title="Coding Agent API")

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]

MODELS = [
    {"id": "qwen2.5-coder", "name": "Qwen 2.5 Coder"},
    {"id": "deepseek-coder-v2", "name": "DeepSeek Coder V2"},
    {"id": "codellama", "name": "Code Llama"}
]

@app.get("/models")
async def get_models():
    return MODELS

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Check if the model is valid
        if request.model not in [m["id"] for m in MODELS]:
            raise HTTPException(status_code=400, detail="Invalid model selected")

        # System prompt for a coding agent
        system_message = {
            "role": "system",
            "content": "You are an expert programming assistant. You write clean, concise, and well-documented code. Always prioritize writing correct code and explaining it simply."
        }
        
        # Prepend system message
        messages = [system_message] + request.messages

        # Streaming response from Ollama
        def generate():
            stream = ollama.chat(
                model=request.model,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")
