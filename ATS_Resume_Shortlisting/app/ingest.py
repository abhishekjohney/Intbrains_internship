import fitz  # PyMuPDF
import requests
import json
import re
from app.models import ResumeData

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder"

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a single-column PDF resume in memory."""
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def is_ats_friendly_rules(markdown_text: str) -> dict:
    """
    Evaluates parsed resume text using strict heuristic rules.
    Returns a dict with 'passed' boolean and the 'reason' if it failed.
    """
    
    if len(markdown_text.strip()) < 200:
        return {"passed": False, "reason": "Insufficient text (Likely image-based)"}

    standard_headers = ['experience', 'education', 'skills', 'employment', 'history']
    text_lower = markdown_text.lower()
    
    has_headers = any(header in text_lower for header in standard_headers)
    if not has_headers:
        return {"passed": False, "reason": "Missing standard resume headers"}

    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    if not re.search(email_pattern, markdown_text):
        return {"passed": False, "reason": "Could not detect a valid email address structure"}

    normal_char_count = sum(c.isalnum() or c.isspace() for c in markdown_text)
    character_ratio = normal_char_count / len(markdown_text) if len(markdown_text) > 0 else 0
    
    if character_ratio < 0.85:
        return {"passed": False, "reason": f"Font encoding failure (Low alphanumeric ratio: {character_ratio:.2f})"}

    return {"passed": True, "reason": "Clean parse"}

def parse_resume_with_llm(text: str) -> ResumeData:
    """Send resume text to Ollama and extract structured JSON."""
    schema = ResumeData.model_json_schema()
    
    prompt = f"""
    You are an expert Applicant Tracking System (ATS).
    Your task is to extract highly structured data from the resume text provided below.
    
    Strict extraction rules:
    1. Separate actual 'Programming_Languages' (like Java, Python, C++) from 'Frameworks_Tools' (like React, Spring Boot, Docker).
    2. Count the exact number of projects listed and set 'Total_Projects_Count'.
    3. For 'Projects', extract the project Name, a brief Description, and a list of specific 'Technologies_Used' (languages/frameworks).
    4. For 'Experiences', list the Role, Company, and calculate Duration_Months if possible.
    5. Ensure you return ONLY valid JSON matching this schema:
    {json.dumps(schema, indent=2)}
    
    Resume Text:
    {text}
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    
    data = response.json()
    result_text = data.get("response", "{}")
    
    # Clean the result text in case of markdown wrappers
    cleaned_text = result_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()
    
    try:
        parsed_json = json.loads(cleaned_text)
        # Validate against the Pydantic model
        resume_data = ResumeData(**parsed_json)
        return resume_data
    except Exception as e:
        print(f"Failed to parse LLM response: {result_text}")
        raise e
