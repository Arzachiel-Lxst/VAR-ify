"""
VAR Backend Configuration
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # App Info
    APP_NAME: str = "VAR Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    FRAMES_DIR: Path = BASE_DIR / "data" / "frames"
    RESULTS_DIR: Path = BASE_DIR / "data" / "results"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Video Processing
    TARGET_FPS: int = 30
    FRAME_WIDTH: int = 1920
    FRAME_HEIGHT: int = 1080
    
    # Scene Detection
    SCENE_CUT_THRESHOLD: float = 0.4
    HISTOGRAM_BINS: int = 256
    
    # Frame Eligibility
    GRASS_WEIGHT: float = 0.30
    LINES_WEIGHT: float = 0.25
    PLAYERS_WEIGHT: float = 0.25
    STABILITY_WEIGHT: float = 0.20
    ELIGIBILITY_THRESHOLD: float = 0.70
    
    # YOLO Detection
    YOLO_MODEL: str = "yolov8n.pt"  # Can be custom trained: yolo_soccer.pt
    YOLO_CONFIDENCE: float = 0.5
    YOLO_IOU: float = 0.45
    
    # Tracking
    TRACK_BUFFER: int = 30
    MATCH_THRESHOLD: float = 0.8
    
    # Ball Tracking (Kalman Filter)
    BALL_VELOCITY_THRESHOLD: float = 15.0  # pixels/frame for pass detection
    
    # Offside Detection
    OFFSIDE_TOLERANCE_CM: float = 5.0  # FIFA tolerance
    OFFSIDE_MULTI_HYPOTHESIS: int = 3
    
    # Handball Detection
    HANDBALL_ZONE_THRESHOLD: float = 0.7
    ARM_EXTENDED_ANGLE: float = 45.0  # degrees
    
    # Decision Engine
    CONFIDENCE_HIGH: float = 0.85
    CONFIDENCE_MEDIUM: float = 0.70
    CONFIDENCE_LOW: float = 0.50
    
    # Scene Classes
    SCENE_CLASSES: List[str] = [
        "field_play",
        "crowd", 
        "bench",
        "replay",
        "close_up"
    ]
    
    # Detection Classes (for custom YOLO)
    DETECTION_CLASSES: List[str] = [
        "player",
        "goalkeeper",
        "ball",
        "goal",
        "sideline",
        "penalty_area"
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
