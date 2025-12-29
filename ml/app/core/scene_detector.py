"""
Scene Cut Detection Module
Detects scene transitions to handle replays and reset tracking
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class SceneCut:
    """Represents a detected scene cut"""
    frame_index: int
    confidence: float
    histogram_diff: float
    cut_type: str  # "hard_cut", "fade", "dissolve"


class SceneDetector:
    """
    Detect scene cuts in video using histogram analysis.
    Used to:
    - Identify replay segments
    - Reset object tracking after cuts
    - Filter out non-continuous footage
    """
    
    def __init__(
        self,
        threshold: float = None,
        histogram_bins: int = None
    ):
        self.threshold = threshold or settings.SCENE_CUT_THRESHOLD
        self.histogram_bins = histogram_bins or settings.HISTOGRAM_BINS
        self.previous_hist: Optional[np.ndarray] = None
        self.scene_cuts: List[SceneCut] = []
    
    def compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute color histogram for frame"""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Compute histogram for H and S channels
        hist_h = cv2.calcHist([hsv], [0], None, [self.histogram_bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [self.histogram_bins], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        
        # Concatenate histograms
        hist = np.concatenate([hist_h.flatten(), hist_s.flatten()])
        return hist
    
    def compute_histogram_difference(
        self,
        hist1: np.ndarray,
        hist2: np.ndarray,
        method: str = "correlation"
    ) -> float:
        """
        Compute difference between two histograms.
        Returns value between 0 (identical) and 1 (completely different)
        """
        if method == "correlation":
            # Correlation returns -1 to 1, we convert to 0-1 difference
            corr = cv2.compareHist(
                hist1.astype(np.float32),
                hist2.astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            return 1 - (corr + 1) / 2
        
        elif method == "chi_square":
            chi_sq = cv2.compareHist(
                hist1.astype(np.float32),
                hist2.astype(np.float32),
                cv2.HISTCMP_CHISQR
            )
            # Normalize chi-square to 0-1 range (approximate)
            return min(chi_sq / 100, 1.0)
        
        elif method == "bhattacharyya":
            return cv2.compareHist(
                hist1.astype(np.float32),
                hist2.astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def detect_cut(self, frame: np.ndarray, frame_index: int) -> Optional[SceneCut]:
        """
        Check if current frame represents a scene cut.
        Returns SceneCut if detected, None otherwise.
        """
        current_hist = self.compute_histogram(frame)
        
        if self.previous_hist is None:
            self.previous_hist = current_hist
            return None
        
        # Compute histogram difference
        hist_diff = self.compute_histogram_difference(
            self.previous_hist,
            current_hist,
            method="correlation"
        )
        
        # Update previous histogram
        self.previous_hist = current_hist
        
        # Check if scene cut detected
        if hist_diff > self.threshold:
            # Determine cut type based on difference magnitude
            if hist_diff > 0.8:
                cut_type = "hard_cut"
            elif hist_diff > 0.6:
                cut_type = "fade"
            else:
                cut_type = "dissolve"
            
            confidence = min(hist_diff / self.threshold, 1.0)
            
            scene_cut = SceneCut(
                frame_index=frame_index,
                confidence=confidence,
                histogram_diff=hist_diff,
                cut_type=cut_type
            )
            
            self.scene_cuts.append(scene_cut)
            logger.debug(f"Scene cut detected at frame {frame_index}: {cut_type}")
            
            return scene_cut
        
        return None
    
    def detect_cuts_batch(
        self,
        frames: List[Tuple[int, np.ndarray]]
    ) -> List[SceneCut]:
        """Process batch of frames and return all detected cuts"""
        self.reset()
        
        for frame_idx, frame in frames:
            self.detect_cut(frame, frame_idx)
        
        return self.scene_cuts.copy()
    
    def get_segments(self) -> List[Tuple[int, int]]:
        """
        Get continuous segments between scene cuts.
        Returns list of (start_frame, end_frame) tuples.
        """
        if not self.scene_cuts:
            return []
        
        segments = []
        cut_frames = [cut.frame_index for cut in self.scene_cuts]
        
        # Add start segment
        if cut_frames[0] > 0:
            segments.append((0, cut_frames[0] - 1))
        
        # Add segments between cuts
        for i in range(len(cut_frames) - 1):
            segments.append((cut_frames[i], cut_frames[i + 1] - 1))
        
        return segments
    
    def is_scene_cut_frame(self, frame_index: int) -> bool:
        """Check if a specific frame is a scene cut"""
        return any(cut.frame_index == frame_index for cut in self.scene_cuts)
    
    def reset(self) -> None:
        """Reset detector state"""
        self.previous_hist = None
        self.scene_cuts.clear()


class ContentClassifier:
    """
    Classify frame content type.
    Classes: field_play, crowd, bench, replay, close_up
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.classes = settings.SCENE_CLASSES
        
        # Load pre-trained model if provided
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> None:
        """Load scene classification model (MobileNet/EfficientNet)"""
        try:
            import torch
            self.model = torch.load(model_path, map_location='cpu')
            self.model.eval()
            logger.info(f"Scene classifier loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load scene classifier: {e}")
            self.model = None
    
    def classify(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Classify frame content type.
        Returns (class_name, confidence)
        """
        if self.model is None:
            # Fallback: use heuristic-based classification
            return self._heuristic_classify(frame)
        
        # Preprocess frame for model
        import torch
        import torchvision.transforms as T
        
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_rgb).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        return self.classes[predicted.item()], confidence.item()
    
    def _heuristic_classify(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Heuristic-based classification when no model is available.
        Uses color analysis and edge detection.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green detection for grass
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        
        # Edge detection for lines
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Skin detection for close-ups
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
        
        # Classification logic
        if green_ratio > 0.3 and edge_ratio > 0.02:
            return "field_play", 0.7 + 0.3 * green_ratio
        elif skin_ratio > 0.3:
            return "close_up", 0.6 + 0.4 * skin_ratio
        elif green_ratio < 0.1:
            return "crowd", 0.5
        else:
            return "replay", 0.4
    
    def is_field_play(self, frame: np.ndarray, threshold: float = 0.6) -> bool:
        """Check if frame shows field play (useful for offside detection)"""
        class_name, confidence = self.classify(frame)
        return class_name == "field_play" and confidence >= threshold
