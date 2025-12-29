"""
Video Loader Module
Handles video ingestion, frame extraction using FFmpeg and OpenCV
"""
import cv2
import subprocess
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, List
import logging

from .config import settings

logger = logging.getLogger(__name__)


class VideoLoader:
    """Load and process video files for VAR analysis"""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 0
        self.frame_count: int = 0
        self.width: int = 0
        self.height: int = 0
        self.duration: float = 0
        
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load video metadata"""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        logger.info(f"Video loaded: {self.video_path.name}")
        logger.info(f"Resolution: {self.width}x{self.height}, FPS: {self.fps}, Duration: {self.duration:.2f}s")
    
    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Get specific frame by index"""
        if self.cap is None:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        
        return frame if ret else None
    
    def get_frames_range(self, start: int, end: int) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Get frames within a range"""
        if self.cap is None:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        for idx in range(start, min(end, self.frame_count)):
            ret, frame = self.cap.read()
            if not ret:
                break
            yield idx, frame
    
    def iterate_frames(self, skip: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterate through all frames with optional skip"""
        if self.cap is None:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % skip == 0:
                yield frame_idx, frame
            
            frame_idx += 1
    
    def extract_frames_ffmpeg(self, output_dir: Path, target_fps: int = None) -> List[Path]:
        """
        Extract frames using FFmpeg (more reliable for various codecs)
        Returns list of extracted frame paths
        """
        target_fps = target_fps or settings.TARGET_FPS
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique prefix from video name
        prefix = self.video_path.stem
        output_pattern = output_dir / f"{prefix}_%05d.jpg"
        
        cmd = [
            "ffmpeg",
            "-i", str(self.video_path),
            "-vf", f"fps={target_fps}",
            "-q:v", "2",  # High quality
            "-y",  # Overwrite
            str(output_pattern)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Frames extracted to {output_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise
        
        # Return list of extracted frames
        frames = sorted(output_dir.glob(f"{prefix}_*.jpg"))
        return frames
    
    def extract_keyframes(self, output_dir: Path) -> List[Path]:
        """Extract only keyframes (I-frames) for quick analysis"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = self.video_path.stem
        output_pattern = output_dir / f"{prefix}_key_%05d.jpg"
        
        cmd = [
            "ffmpeg",
            "-i", str(self.video_path),
            "-vf", "select=eq(pict_type\\,I)",
            "-vsync", "vfr",
            "-q:v", "2",
            "-y",
            str(output_pattern)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg keyframe extraction error: {e.stderr.decode()}")
            raise
        
        frames = sorted(output_dir.glob(f"{prefix}_key_*.jpg"))
        return frames
    
    def get_clip_info(self) -> dict:
        """Return clip metadata as dictionary"""
        return {
            "path": str(self.video_path),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": self.duration
        }
    
    def release(self) -> None:
        """Release video capture resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def __del__(self):
        self.release()


def load_frame_from_file(frame_path: Path) -> np.ndarray:
    """Load a single frame from image file"""
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise ValueError(f"Cannot load frame: {frame_path}")
    return frame


def load_frames_from_directory(frames_dir: Path, pattern: str = "*.jpg") -> Generator[Tuple[int, np.ndarray], None, None]:
    """Load frames from directory in sorted order"""
    frame_files = sorted(frames_dir.glob(pattern))
    
    for idx, frame_path in enumerate(frame_files):
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            yield idx, frame
