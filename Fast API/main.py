from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import uvicorn

# ==========================================
# 1. AI ENGINE INITIALIZATION
# ==========================================
print("Booting up Dual NLP Engines...")

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: spaCy model not found. Run: python -m spacy download en_core_web_sm")
    exit()

# Load NLTK Data
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# ==========================================
# 2. SERVER & PYDANTIC SETUP
# ==========================================
app = FastAPI(title="Dual-Engine NLP Preprocessor")

class TextPayload(BaseModel):
    raw_text: str

# ==========================================
# 3. THE API ROUTE
# ==========================================
@app.post("/api/preprocess")
def preprocess_text(payload: TextPayload):
    text = payload.raw_text
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # --- ENGINE 1: NLTK (Stemming) ---
    nltk_text = text.lower()
    tokens = word_tokenize(nltk_text)
    stop_words = set(stopwords.words('english'))
    
    nltk_clean = [
        word for word in tokens 
        if word not in string.punctuation and word not in stop_words
    ]
    
    stemmer = PorterStemmer()
    nltk_result = [stemmer.stem(word) for word in nltk_clean]

    # --- ENGINE 2: spaCy (Lemmatization) ---
    doc = nlp(text)
    spacy_result = [
        token.lemma_.lower() for token in doc 
        if not token.is_punct and not token.is_stop and token.text.strip()
    ]

    # --- COMPILE FINAL JSON RESPONSE ---
    return {
        "status": "success",
        "original_text": text,
        "engine_results": {
            "nltk": {
                "method": "Stemming",
                "tokens": nltk_result
            },
            "spacy": {
                "method": "Lemmatization",
                "tokens": spacy_result
            }
        }
    }

# ==========================================
# 4. SERVER BOOT COMMAND
# ==========================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)