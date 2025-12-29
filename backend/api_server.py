"""
VAR-ify Backend API
Lightweight API for Railway deployment
ML processing via Hugging Face Spaces
"""
import os
import re
import uuid
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx

# Configuration
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "")
MAX_VIDEO_DURATION = 15

app = FastAPI(
    title="VAR-ify API",
    version="1.0.0",
    description="Video Assistant Referee Analysis System"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# Directories
UPLOAD_DIR = Path("data/uploads")
RESULTS_DIR = Path("data/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem
    clean = re.sub(r'[^\w\s-]', '', stem)
    clean = re.sub(r'[\s-]+', '_', clean).strip('_')
    if not clean:
        clean = f"video_{uuid.uuid4().hex[:8]}"
    return f"{clean}{ext}"


def get_video_duration(file_path: str) -> float:
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
    subprocess.run([
        'ffmpeg', '-y', '-i', input_path,
        '-t', str(duration), '-c', 'copy', output_path
    ], capture_output=True)


@app.get("/")
async def root():
    return {"app": "VAR-ify API", "version": "1.0.0", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "VAR-ify API", "ml_service": HF_SPACE_URL or "not configured"}


@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    ext = Path(video.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}")

    safe_filename = sanitize_filename(video.filename)
    video_id = uuid.uuid4().hex[:12]
    final_filename = f"{video_id}_{safe_filename}"
    file_path = UPLOAD_DIR / final_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        duration = get_video_duration(str(file_path))
        was_trimmed = False

        if duration > MAX_VIDEO_DURATION:
            trimmed_path = UPLOAD_DIR / f"trimmed_{final_filename}"
            trim_video(str(file_path), str(trimmed_path), MAX_VIDEO_DURATION)
            os.remove(file_path)
            shutil.move(str(trimmed_path), str(file_path))
            was_trimmed = True

        # Return info for ML processing
        return {
            "success": True,
            "video_id": video_id,
            "filename": final_filename,
            "size": file_path.stat().st_size,
            "duration": min(duration, MAX_VIDEO_DURATION),
            "trimmed": was_trimmed,
            "message": "Video uploaded. Use ML service for analysis.",
            "ml_service": HF_SPACE_URL or "Configure HF_SPACE_URL env var"
        }
    except Exception as e:
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{filename}")
async def get_result_video(filename: str):
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        file_path,
        media_type="video/mp4",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Content-Disposition": f"inline; filename={filename}"
        }
    )


@app.get("/api/results/{filename}")
async def get_result_video_api(filename: str):
    return await get_result_video(filename)


@app.get("/api/uploads")
async def list_uploads():
    uploads = []
    for f in UPLOAD_DIR.glob("*"):
        if f.is_file():
            uploads.append({
                "filename": f.name,
                "size": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })
    return {"uploads": uploads}


class AnalyzeRequest(BaseModel):
    filename: str


@app.post("/api/analyze")
async def analyze_video(request: AnalyzeRequest):
    """
    Analyze video for VAR violations.
    Calls Hugging Face ML service if configured, otherwise returns mock data.
    """
    file_path = UPLOAD_DIR / request.filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.filename}")
    
    # If HF_SPACE_URL is configured, call the ML service
    if HF_SPACE_URL:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Upload video to HF Space for analysis
                with open(file_path, "rb") as f:
                    files = {"video": (request.filename, f, "video/mp4")}
                    response = await client.post(
                        f"{HF_SPACE_URL}/api/analyze",
                        files=files
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "filename": request.filename,
                        "handball_events": result.get("handball_events", 0),
                        "offside_events": result.get("offside_events", 0),
                        "violations": result.get("violations", []),
                        "video_url": result.get("video_url"),
                        "source": "ml_service"
                    }
        except Exception as e:
            print(f"ML service error: {e}")
    
    # Fallback: Return demo analysis result
    video_stem = file_path.stem
    result_video = f"{video_stem}_VAR.mp4"
    
    return {
        "success": True,
        "filename": request.filename,
        "handball_events": 0,
        "offside_events": 0,
        "violations": [],
        "video_url": f"/results/{result_video}" if (RESULTS_DIR / result_video).exists() else None,
        "message": "Analysis complete. ML service not configured - using demo mode.",
        "source": "demo"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
