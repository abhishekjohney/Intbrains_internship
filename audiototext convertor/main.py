from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import whisper
import os
import uvicorn

app = FastAPI(title="Universal Audio Transcriber")

# 1. LOAD THE AI MODEL 
print("Loading Whisper AI... (This may take a few seconds)")
# Using the "base" model for a great mix of speed and accuracy
model = whisper.load_model("base") 


class AudioRequest(BaseModel):
    file_path: str

# 3. THE TRANSCRIPTION ROUTE
@app.post("/api/transcribe")
def transcribe_audio(payload: AudioRequest):
    if not os.path.exists(payload.file_path):
        raise HTTPException(status_code=404, detail=f"File not found at: {payload.file_path}")

    valid_extensions = (".mp3", ".wav", ".m4a")
    if not payload.file_path.lower().endswith(valid_extensions):
        raise HTTPException(status_code=400, detail=f"File must be one of: {valid_extensions}")

    print(f"Transcribing file: {payload.file_path}...")

    try:
        result = model.transcribe(payload.file_path)
        extracted_text = result["text"]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    return {
        "status": "success",
        "file": os.path.basename(payload.file_path),
        "text": extracted_text.strip()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)