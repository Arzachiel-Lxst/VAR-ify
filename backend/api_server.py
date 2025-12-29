"""
VAR-ify Backend API
Combined API with integrated ML processing
"""
import os
import re
import sys
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

# Add ml folder to path for imports
ML_PATH = Path(__file__).parent.parent / "ml"
sys.path.insert(0, str(ML_PATH))

# Import VAR system from ml folder
try:
    from run_var import VARSystem
    ML_AVAILABLE = True
    print("[Backend] ML system loaded successfully")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"[Backend] ML system not available: {e}")

# Configuration
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
    return {"status": "healthy", "service": "VAR-ify API", "ml_available": ML_AVAILABLE}


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
            "duration": min(duration, MAX_VIDEO_DURATION) if duration > 0 else 15,
            "trimmed": was_trimmed,
            "message": "Video uploaded successfully",
            "ml_available": ML_AVAILABLE
        }
    except Exception as e:
        import traceback
        print(f"[Upload Error] {traceback.format_exc()}")
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


# VAR system instance
var_system = None

def get_var_system():
    global var_system
    if var_system is None and ML_AVAILABLE:
        var_system = VARSystem(output_dir=str(RESULTS_DIR))
    return var_system


@app.post("/api/analyze")
async def analyze_video(request: AnalyzeRequest):
    """
    Analyze video for VAR violations using integrated ML.
    """
    file_path = UPLOAD_DIR / request.filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.filename}")
    
    # Use integrated ML
    if ML_AVAILABLE:
        try:
            print(f"[VAR] Analyzing: {file_path}")
            var = get_var_system()
            results = var.analyze(str(file_path), create_video=True)
            print(f"[VAR] Results: {results}")
            
            handball_events = results.get("handball", [])
            offside_events = results.get("offside", [])
            summary = results.get("summary", {})
            
            # Find result video
            video_stem = file_path.stem
            result_video = f"{video_stem}_VAR.mp4"
            video_url = None
            
            if (RESULTS_DIR / result_video).exists():
                video_url = f"/results/{result_video}"
                print(f"[VAR] Video found: {video_url}")
            
            return {
                "success": True,
                "filename": request.filename,
                "handball_events": len(handball_events),
                "offside_events": len(offside_events),
                "handball": handball_events,
                "offside": offside_events,
                "video_url": video_url,
                "summary": summary,
                "source": "integrated_ml"
            }
        except Exception as e:
            import traceback
            print(f"[VAR] Error: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Fallback: ML not available
    return {
        "success": False,
        "filename": request.filename,
        "handball_events": 0,
        "offside_events": 0,
        "handball": [],
        "offside": [],
        "video_url": None,
        "summary": {"total_violations": 0},
        "message": "ML system not available",
        "source": "none"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
