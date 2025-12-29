"""
Object Detection Module
YOLO-based detection for soccer elements
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single object detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]
    area: float
    
    # Additional attributes
    team: Optional[str] = None  # For players: "home", "away"
    jersey_color: Optional[Tuple[int, int, int]] = None
    
    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "center": self.center,
            "area": self.area,
            "team": self.team
        }


@dataclass
class FrameDetections:
    """All detections for a single frame"""
    frame_index: int
    detections: List[Detection] = field(default_factory=list)
    
    @property
    def players(self) -> List[Detection]:
        return [d for d in self.detections if d.class_name in ["player", "goalkeeper"]]
    
    @property
    def ball(self) -> Optional[Detection]:
        balls = [d for d in self.detections if d.class_name == "ball"]
        return balls[0] if balls else None
    
    @property
    def goalkeepers(self) -> List[Detection]:
        return [d for d in self.detections if d.class_name == "goalkeeper"]


class SoccerDetector:
    """
    YOLO-based object detector for soccer analysis.
    
    Detects:
    - player
    - goalkeeper
    - ball
    - goal
    - sideline
    - penalty_area
    """
    
    # Default YOLO COCO classes that map to soccer elements
    COCO_MAPPING = {
        0: "player",      # person -> player
        32: "ball",       # sports ball
    }
    
    # Custom soccer model classes
    SOCCER_CLASSES = {
        0: "player",
        1: "goalkeeper",
        2: "ball",
        3: "goal",
        4: "sideline",
        5: "penalty_area"
    }
    
    def __init__(
        self,
        model_path: str = None,
        confidence: float = None,
        iou_threshold: float = None,
        use_cuda: bool = True
    ):
        self.confidence = confidence or settings.YOLO_CONFIDENCE
        self.iou_threshold = iou_threshold or settings.YOLO_IOU
        self.model = None
        self.is_custom_model = False
        
        model_path = model_path or settings.YOLO_MODEL
        self._load_model(model_path, use_cuda)
    
    def _load_model(self, model_path: str, use_cuda: bool) -> None:
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            
            # Check if custom soccer model exists
            custom_path = settings.MODELS_DIR / "yolo_soccer.pt"
            
            if custom_path.exists():
                self.model = YOLO(str(custom_path))
                self.is_custom_model = True
                logger.info(f"Loaded custom soccer model: {custom_path}")
            else:
                # Use default YOLOv8 model
                self.model = YOLO(model_path)
                self.is_custom_model = False
                logger.info(f"Loaded default YOLO model: {model_path}")
            
            # Move to GPU if available
            if use_cuda:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    logger.info("YOLO running on CUDA")
                    
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, frame: np.ndarray, frame_index: int = 0) -> FrameDetections:
        """
        Run detection on a single frame.
        
        Args:
            frame: BGR image
            frame_index: Index of the frame
            
        Returns:
            FrameDetections object
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            bbox = tuple(box.xyxy[0].cpu().numpy().tolist())
            
            # Map class ID to name
            if self.is_custom_model:
                class_name = self.SOCCER_CLASSES.get(class_id, "unknown")
            else:
                class_name = self.COCO_MAPPING.get(class_id, None)
                if class_name is None:
                    continue  # Skip non-soccer classes
            
            # Calculate center and area
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            area = (x2 - x1) * (y2 - y1)
            
            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                center=center,
                area=area
            )
            
            # Try to identify team by jersey color
            if class_name in ["player", "goalkeeper"]:
                jersey_color = self._extract_jersey_color(frame, bbox)
                detection.jersey_color = jersey_color
            
            detections.append(detection)
        
        return FrameDetections(frame_index=frame_index, detections=detections)
    
    def detect_batch(
        self,
        frames: List[Tuple[int, np.ndarray]]
    ) -> List[FrameDetections]:
        """
        Run detection on batch of frames.
        """
        results = []
        
        for frame_idx, frame in frames:
            frame_detections = self.detect(frame, frame_idx)
            results.append(frame_detections)
        
        return results
    
    def _extract_jersey_color(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[int, int, int]:
        """Extract dominant jersey color from player bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract upper body region (jersey)
        height = y2 - y1
        jersey_y1 = y1 + int(height * 0.1)
        jersey_y2 = y1 + int(height * 0.4)
        
        jersey_region = frame[jersey_y1:jersey_y2, x1:x2]
        
        if jersey_region.size == 0:
            return (128, 128, 128)  # Gray default
        
        # Convert to RGB and find dominant color
        jersey_rgb = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2RGB)
        pixels = jersey_rgb.reshape(-1, 3)
        
        # Use k-means to find dominant color
        from scipy.cluster.vq import kmeans
        try:
            centroids, _ = kmeans(pixels.astype(float), 2)
            dominant = centroids[0].astype(int)
            return tuple(dominant)
        except Exception:
            # Fallback: use mean color
            mean_color = np.mean(pixels, axis=0).astype(int)
            return tuple(mean_color)
    
    def classify_teams(
        self,
        detections: FrameDetections,
        team_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> FrameDetections:
        """
        Classify players into teams based on jersey color.
        
        Args:
            detections: Frame detections
            team_colors: Optional dict {"home": (R,G,B), "away": (R,G,B)}
        """
        players = detections.players
        
        if len(players) < 2:
            return detections
        
        # Get all jersey colors
        colors = [p.jersey_color for p in players if p.jersey_color]
        
        if not colors:
            return detections
        
        colors_array = np.array(colors)
        
        if team_colors is None:
            # Cluster colors to identify teams
            from scipy.cluster.vq import kmeans, vq
            
            try:
                centroids, _ = kmeans(colors_array.astype(float), 2)
                labels, _ = vq(colors_array.astype(float), centroids)
                
                for i, player in enumerate(players):
                    if i < len(labels):
                        player.team = "home" if labels[i] == 0 else "away"
                        
            except Exception as e:
                logger.warning(f"Team classification failed: {e}")
        else:
            # Use provided team colors
            home_color = np.array(team_colors["home"])
            away_color = np.array(team_colors["away"])
            
            for player in players:
                if player.jersey_color:
                    color = np.array(player.jersey_color)
                    home_dist = np.linalg.norm(color - home_color)
                    away_dist = np.linalg.norm(color - away_color)
                    player.team = "home" if home_dist < away_dist else "away"
        
        return detections
    
    def get_feet_positions(
        self,
        detections: FrameDetections
    ) -> Dict[int, Tuple[float, float]]:
        """
        Get estimated feet positions for all players.
        Uses bottom center of bounding box.
        """
        feet_positions = {}
        
        for i, player in enumerate(detections.players):
            x1, y1, x2, y2 = player.bbox
            feet_x = (x1 + x2) / 2
            feet_y = y2  # Bottom of bbox
            feet_positions[i] = (feet_x, feet_y)
        
        return feet_positions


def create_detector(use_custom: bool = True) -> SoccerDetector:
    """Factory function to create detector"""
    if use_custom:
        custom_path = settings.MODELS_DIR / "yolo_soccer.pt"
        if custom_path.exists():
            return SoccerDetector(model_path=str(custom_path))
    
    return SoccerDetector()
