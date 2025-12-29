"""
VAR Backend - Main Application Entry Point
FastAPI application for Video Assistant Referee analysis
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import settings
from .api.upload import router as upload_router

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"Models directory: {settings.MODELS_DIR}")
    
    # Ensure directories exist
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down VAR Backend")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## VAR Backend API
    
    Video Assistant Referee system for soccer highlight analysis.
    
    ### Features
    - **Video Upload**: Upload soccer highlight clips for analysis
    - **Offside Detection**: Analyze potential offside situations
    - **Handball Detection**: Detect handball violations
    - **Confidence Scoring**: All decisions include confidence levels
    
    ### Output Format
    ```json
    {
        "clip_id": "match_001_clip_07",
        "events": [
            {
                "type": "offside",
                "decision": "PROBABLE",
                "confidence": 0.87,
                "reason": "Camera stable, field visible",
                "frame_index": 1245
            }
        ]
    }
    ```
    
    ### Decision Types
    - **YES**: Definitive violation detected
    - **NO**: Definitive no violation
    - **PROBABLE**: Likely violation but uncertainty exists
    - **NOT_DECIDABLE**: Cannot determine due to poor data quality
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router)


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "VAR Backend API for soccer highlight analysis",
        "endpoints": {
            "health": "/api/v1/health",
            "upload": "/api/v1/upload",
            "analyze": "/api/v1/analyze",
            "analyze_file": "/api/v1/analyze/file",
            "clips": "/api/v1/clips",
            "docs": "/docs"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


# Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
