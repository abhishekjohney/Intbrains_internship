from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Any, Dict
import os
import shutil
import uuid

from app.models import ResumeData
from app.ingest import extract_text_from_pdf, parse_resume_with_llm, is_ats_friendly_rules
from app.db import init_db, upsert_resume, search_resumes

from fastapi.responses import RedirectResponse

app = FastAPI(title="Local ATS Backend API")

@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(url='/docs')

RESUMES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resumes")
os.makedirs(RESUMES_DIR, exist_ok=True)

@app.on_event("startup")
def startup_event():
    init_db()

@app.post("/ingest", response_model=Dict[str, Any])
def ingest_resume(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # Read file into memory
        file_bytes = file.file.read()
        
        # Extract text from memory
        text = extract_text_from_pdf(file_bytes)
        
        # Check if resume is ATS-friendly using strict programmatic rules
        ats_check = is_ats_friendly_rules(text)
        if not ats_check["passed"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Resume is not ATS-friendly: {ats_check['reason']}. Please upload a standard text-based PDF."
            )
        
        # Parse with LLM
        resume_data = parse_resume_with_llm(text)
        
        # Store in Qdrant
        point_id = upsert_resume(resume_data.model_dump(), raw_text=text)
        
        return {
            "status": "success",
            "message": "Resume ingested successfully",
            "id": point_id,
            "data": resume_data.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SearchRequest(BaseModel):
    query: str

class JobRequirements(BaseModel):
    query: str
    required_skills: List[str] = []
    min_experience_years: int = 0

@app.post("/ats-rank", response_model=List[Dict[str, Any]])
def rank_candidates(job: JobRequirements):
    try:
        # 1. Get base semantic matches from Qdrant
        results = search_resumes(job.query, top_k=20)
        
        # 2. Calculate ATS score for each candidate
        ranked_results = []
        for candidate in results:
            ats_score = 0.0
            
            # Semantic search score (maps to max 40 points)
            base_score = candidate.get("_score", 0) * 40
            ats_score += base_score
            
            # Skill Match (max 40 points)
            candidate_skills = [s.lower() for s in candidate.get("Programming_Languages", []) + candidate.get("Frameworks_Tools", [])]
            if job.required_skills:
                matched_skills = sum(1 for skill in job.required_skills if skill.lower() in candidate_skills)
                skill_score = (matched_skills / len(job.required_skills)) * 40
                ats_score += skill_score
            else:
                ats_score += 40
                
            # Experience Match (max 20 points)
            cand_exp = candidate.get("Years_of_Experience") or 0
            if cand_exp >= job.min_experience_years:
                ats_score += 20
            else:
                if job.min_experience_years > 0:
                    ats_score += (cand_exp / job.min_experience_years) * 20
                    
            candidate["ATS_Match_Percentage"] = round(min(ats_score, 100.0), 2)
            ranked_results.append(candidate)
            
        # Sort by ATS Match Percentage descending
        ranked_results.sort(key=lambda x: x["ATS_Match_Percentage"], reverse=True)
        return ranked_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[Dict[str, Any]])
def search(request: SearchRequest):
    try:
        results = search_resumes(request.query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
