"""
VAR-ify Backend API (Lightweight Version for Railway)
ML processing is done locally, this API handles uploads and serves results
"""

import os
import re
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(
    title="VAR-ify API",
    description="Video Assistant Referee Analysis System",
    version="1.0.0"
)

# CORS - allow all origins for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
RESULTS_DIR = BASE_DIR / "data" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Max video duration
MAX_VIDEO_DURATION = 15


def sanitize_filename(filename: str) -> str:
    """Remove special characters from filename"""
    name = Path(filename).stem
    ext = Path(filename).suffix
    clean_name = re.sub(r'[^\w\-_]', '_', name)
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    if not clean_name:
        clean_name = f"video_{uuid.uuid4().hex[:8]}"
    return f"{clean_name}{ext}"


class AnalysisResponse(BaseModel):
    success: bool
    message: str
    video_id: Optional[str] = None
    filename: Optional[str] = None
    results: Optional[dict] = None
    video_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    mode: str


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "mode": "lite"
    }


@app.post("/api/upload", response_model=AnalysisResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload video for analysis"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid video format")
        
        # Sanitize filename
        safe_filename = sanitize_filename(file.filename)
        video_id = uuid.uuid4().hex[:12]
        
        # Save uploaded file
        upload_path = UPLOAD_DIR / f"{video_id}_{safe_filename}"
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "success": True,
            "message": f"Video uploaded successfully. ML processing requires local setup.",
            "video_id": video_id,
            "filename": safe_filename,
            "results": {
                "status": "uploaded",
                "note": "For full VAR analysis with ML detection, run the backend locally with full dependencies.",
                "handball_events": [],
                "offside_events": []
            },
            "video_url": None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/{filename}")
async def get_result_video(filename: str):
    """Serve result video with proper headers"""
    file_path = RESULTS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Content-Disposition": f"inline; filename={filename}"
        }
    )


@app.get("/results/{filename}")
async def get_result_video_alt(filename: str):
    """Alternative route for result videos"""
    return await get_result_video(filename)


@app.get("/api/analyses")
async def list_analyses():
    """List all uploaded videos"""
    uploads = []
    for f in UPLOAD_DIR.glob("*"):
        if f.is_file():
            uploads.append({
                "filename": f.name,
                "size": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })
    return {"analyses": uploads}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": "VAR-ify API",
        "version": "1.0.0",
        "mode": "lite",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
