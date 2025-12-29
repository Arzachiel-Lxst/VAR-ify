"""
Upload API Module
Handles video upload and VAR analysis endpoints
"""
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..core.config import settings
from ..services.var_pipeline import VARPipeline, PipelineConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["VAR Analysis"])


# Request/Response Models
class AnalysisRequest(BaseModel):
    """Request model for analysis"""
    clip_id: Optional[str] = None
    analyze_offside: bool = True
    analyze_handball: bool = True


class AnalysisResponse(BaseModel):
    """Response model for analysis"""
    clip_id: str
    status: str
    message: str
    result: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str


# In-memory job storage (use Redis in production)
analysis_jobs = {}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION
    )


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    clip_id: Optional[str] = None
):
    """
    Upload video for VAR analysis.
    Returns job_id for tracking.
    """
    # Validate file type
    allowed_types = ["video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {allowed_types}"
        )
    
    # Generate IDs
    job_id = str(uuid.uuid4())
    clip_id = clip_id or f"clip_{job_id[:8]}"
    
    # Save file
    file_ext = Path(file.filename).suffix or ".mp4"
    save_path = settings.UPLOAD_DIR / f"{clip_id}{file_ext}"
    
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Video uploaded: {save_path}")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to save video")
    
    # Store job info
    analysis_jobs[job_id] = {
        "clip_id": clip_id,
        "video_path": str(save_path),
        "status": "uploaded",
        "result": None
    }
    
    return {
        "job_id": job_id,
        "clip_id": clip_id,
        "status": "uploaded",
        "message": "Video uploaded. Call /analyze to start analysis."
    }


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks = None
):
    """
    Start VAR analysis on uploaded video.
    
    For synchronous analysis (small videos):
    - Returns result directly
    
    For async analysis (large videos):
    - Returns job_id and processes in background
    """
    clip_id = request.clip_id
    
    if not clip_id:
        raise HTTPException(status_code=400, detail="clip_id required")
    
    # Find video file
    video_path = None
    for ext in [".mp4", ".avi", ".mov"]:
        path = settings.UPLOAD_DIR / f"{clip_id}{ext}"
        if path.exists():
            video_path = path
            break
    
    if video_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Video not found for clip_id: {clip_id}"
        )
    
    # Configure pipeline
    config = PipelineConfig(
        analyze_offside=request.analyze_offside,
        analyze_handball=request.analyze_handball
    )
    
    try:
        # Run analysis
        pipeline = VARPipeline(config)
        result = pipeline.analyze(str(video_path), clip_id)
        
        # Save result
        result_dict = result.to_dict()
        
        return AnalysisResponse(
            clip_id=clip_id,
            status="completed",
            message=f"Analysis complete. {len(result.decisions)} events detected.",
            result=result_dict
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/analyze/file")
async def analyze_uploaded_file(
    file: UploadFile = File(...),
    analyze_offside: bool = True,
    analyze_handball: bool = True
):
    """
    Upload and analyze video in one request.
    For smaller videos / quick testing.
    """
    # Validate
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Generate clip_id
    clip_id = f"clip_{uuid.uuid4().hex[:8]}"
    
    # Save temporarily
    file_ext = Path(file.filename).suffix or ".mp4"
    save_path = settings.UPLOAD_DIR / f"{clip_id}{file_ext}"
    
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save: {e}")
    
    # Analyze
    config = PipelineConfig(
        analyze_offside=analyze_offside,
        analyze_handball=analyze_handball
    )
    
    try:
        pipeline = VARPipeline(config)
        result = pipeline.analyze(str(save_path), clip_id)
        
        return {
            "clip_id": clip_id,
            "status": "completed",
            "frames_processed": result.frames_processed,
            "frames_eligible": result.frames_eligible,
            "processing_time": round(result.processing_time, 2),
            "events": [d.to_dict() for d in result.decisions]
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{clip_id}")
async def get_result(clip_id: str):
    """Get analysis result for a clip"""
    # Check saved results
    result_path = settings.RESULTS_DIR / f"{clip_id}.json"
    
    if result_path.exists():
        import json
        with open(result_path) as f:
            return json.load(f)
    
    # Check in-memory jobs
    for job_id, job in analysis_jobs.items():
        if job["clip_id"] == clip_id and job["result"]:
            return job["result"]
    
    raise HTTPException(status_code=404, detail="Result not found")


@router.get("/clips")
async def list_clips():
    """List all uploaded clips"""
    clips = []
    
    for ext in ["*.mp4", "*.avi", "*.mov"]:
        for path in settings.UPLOAD_DIR.glob(ext):
            clips.append({
                "clip_id": path.stem,
                "filename": path.name,
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2)
            })
    
    return {"clips": clips}


@router.delete("/clips/{clip_id}")
async def delete_clip(clip_id: str):
    """Delete a clip and its results"""
    deleted = []
    
    # Delete video files
    for ext in [".mp4", ".avi", ".mov"]:
        path = settings.UPLOAD_DIR / f"{clip_id}{ext}"
        if path.exists():
            path.unlink()
            deleted.append(str(path))
    
    # Delete result
    result_path = settings.RESULTS_DIR / f"{clip_id}.json"
    if result_path.exists():
        result_path.unlink()
        deleted.append(str(result_path))
    
    # Delete frames
    frames_dir = settings.FRAMES_DIR / clip_id
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
        deleted.append(str(frames_dir))
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    return {"message": "Deleted", "files": deleted}
