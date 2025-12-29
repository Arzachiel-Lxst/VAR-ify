"""
VAR API Server
FastAPI backend for VAR-ify Analysis System
"""
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import uvicorn

from run_var import VARSystem
from database import init_db, get_db, Analysis

MAX_VIDEO_DURATION = 15  # seconds

app = FastAPI(
    title="VAR-ify API", 
    version="1.0.0",
    description="Video Assistant Referee Analysis System - Deteksi Handball & Offside"
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("data/uploads")
RESULTS_DIR = Path("data/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Don't use StaticFiles for results - use custom endpoint with CORS headers


class AnalyzeRequest(BaseModel):
    filename: str


@app.get("/")
async def root():
    return {"message": "VAR Analysis API", "version": "1.0.0"}


def get_video_duration(file_path: str) -> float:
    """Get video duration using ffprobe"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
             '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except:
        return 0


def trim_video(input_path: str, output_path: str, duration: int):
    """Trim video to specified duration using ffmpeg"""
    subprocess.run([
        'ffmpeg', '-y', '-i', input_path, 
        '-t', str(duration), 
        '-c', 'copy', output_path
    ], capture_output=True)


import re
import uuid

def sanitize_filename(filename: str) -> str:
    """Sanitize filename - remove special chars, emoji, spaces"""
    # Get extension
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem
    
    # Remove emoji and special chars, keep only alphanumeric and underscore
    clean = re.sub(r'[^\w\s-]', '', stem)
    clean = re.sub(r'[\s-]+', '_', clean).strip('_')
    
    # If empty after cleaning, use UUID
    if not clean:
        clean = f"video_{uuid.uuid4().hex[:8]}"
    
    return f"{clean}{ext}"


@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...)):
    """Upload video file for analysis"""
    if not video.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    ext = Path(video.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}")
    
    # Sanitize filename
    safe_filename = sanitize_filename(video.filename)
    
    # Save file temporarily
    temp_path = UPLOAD_DIR / f"temp_{safe_filename}"
    file_path = UPLOAD_DIR / safe_filename
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Check duration and trim if needed
        duration = get_video_duration(str(temp_path))
        was_trimmed = False
        
        if duration > MAX_VIDEO_DURATION:
            # Trim video to 15 seconds
            trim_video(str(temp_path), str(file_path), MAX_VIDEO_DURATION)
            os.remove(temp_path)
            was_trimmed = True
        else:
            # Just rename temp to final
            shutil.move(str(temp_path), str(file_path))
        
        return {
            "filename": safe_filename,
            "size": file_path.stat().st_size,
            "original_duration": round(duration, 2),
            "trimmed": was_trimmed,
            "message": f"Upload successful{' (trimmed to 15s)' if was_trimmed else ''}"
        }
        
    except Exception as e:
        # Cleanup temp file if exists
        if temp_path.exists():
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


@app.post("/api/analyze")
async def analyze_video(request: AnalyzeRequest, db: Session = Depends(get_db)):
    """Analyze uploaded video for handball and offside"""
    video_path = UPLOAD_DIR / request.filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Create analysis record
    analysis = Analysis(
        video_name=request.filename,
        video_size=video_path.stat().st_size,
        status="processing"
    )
    db.add(analysis)
    db.commit()
    
    try:
        # Run VAR analysis
        var_system = VARSystem()
        result = var_system.analyze(str(video_path))
        
        # Get result video URL - always set if violations found
        result_video_name = f"{video_path.stem}_VAR.mp4"
        result_video_path = RESULTS_DIR / result_video_name
        
        # Always return video URL if there are violations (video should exist)
        total_violations = len(result.get("handball", [])) + len(result.get("offside", []))
        video_url = f"/results/{result_video_name}" if total_violations > 0 else None
        
        # Update analysis record
        analysis.handball_count = len(result.get("handball", []))
        analysis.offside_count = len(result.get("offside", []))
        analysis.total_violations = result.get("summary", {}).get("total_violations", 0)
        analysis.result_json = result
        analysis.result_video = video_url
        analysis.status = "completed"
        analysis.completed_at = datetime.utcnow()
        db.commit()
        
        return {
            "id": analysis.id,
            "video": result.get("video"),
            "analyzed_at": result.get("analyzed_at"),
            "handball": result.get("handball", []),
            "offside": result.get("offside", []),
            "summary": result.get("summary", {}),
            "video_url": video_url
        }
        
    except Exception as e:
        analysis.status = "failed"
        analysis.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/results/{filename}")
async def get_result_video(filename: str):
    """Get result video file with CORS headers"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        file_path, 
        media_type="video/mp4",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Content-Disposition": f"inline; filename={filename}"
        }
    )


@app.get("/api/results/{filename}")
async def get_result_video_api(filename: str):
    """Get result video file (API path) with CORS headers"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        file_path, 
        media_type="video/mp4",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Content-Disposition": f"inline; filename={filename}"
        }
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "VAR-ify API"}


@app.get("/api/history")
async def get_history(limit: int = 10, db: Session = Depends(get_db)):
    """Get analysis history"""
    analyses = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(limit).all()
    return [
        {
            "id": a.id,
            "video_name": a.video_name,
            "handball_count": a.handball_count,
            "offside_count": a.offside_count,
            "total_violations": a.total_violations,
            "status": a.status,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "result_video": a.result_video
        }
        for a in analyses
    ]


@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """Get specific analysis result"""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {
        "id": analysis.id,
        "video_name": analysis.video_name,
        "result": analysis.result_json,
        "result_video": analysis.result_video,
        "status": analysis.status,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
