"""
Camera Analyzer Module
Analyzes camera motion, zoom levels, and view stability
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import logging

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class CameraState:
    """Camera state for a frame"""
    frame_index: int
    zoom_level: float  # 1.0 = normal, >1 = zoomed in
    pan_velocity: Tuple[float, float]  # (dx, dy) pixels/frame
    tilt_velocity: float  # vertical motion
    is_stable: bool
    is_zoomed: bool
    transformation_matrix: Optional[np.ndarray]


class CameraAnalyzer:
    """
    Analyze camera motion and zoom to handle:
    - Camera pans/tilts during play
    - Zoom changes that affect calibration
    - Unstable footage
    """
    
    def __init__(self):
        self.previous_frame: Optional[np.ndarray] = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
        # History for smoothing
        self.zoom_history: List[float] = []
        self.motion_history: List[Tuple[float, float]] = []
        self.history_size = 10
        
        # Thresholds
        self.zoom_change_threshold = 0.1  # 10% zoom change
        self.motion_threshold = 5.0  # pixels/frame
    
    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_index: int
    ) -> CameraState:
        """Analyze camera state for current frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        
        if self.previous_frame is None or self.previous_descriptors is None:
            self.previous_frame = gray
            self.previous_keypoints = keypoints
            self.previous_descriptors = descriptors
            
            return CameraState(
                frame_index=frame_index,
                zoom_level=1.0,
                pan_velocity=(0, 0),
                tilt_velocity=0,
                is_stable=True,
                is_zoomed=False,
                transformation_matrix=None
            )
        
        # Match features
        if descriptors is None or len(descriptors) < 10:
            self._update_state(gray, keypoints, descriptors)
            return self._create_unknown_state(frame_index)
        
        try:
            matches = self.matcher.knnMatch(self.previous_descriptors, descriptors, k=2)
        except Exception:
            self._update_state(gray, keypoints, descriptors)
            return self._create_unknown_state(frame_index)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 10:
            self._update_state(gray, keypoints, descriptors)
            return self._create_unknown_state(frame_index)
        
        # Extract matched points
        src_pts = np.float32([
            self.previous_keypoints[m.queryIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)
        
        dst_pts = np.float32([
            keypoints[m.trainIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)
        
        # Estimate transformation
        try:
            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        except Exception:
            self._update_state(gray, keypoints, descriptors)
            return self._create_unknown_state(frame_index)
        
        if matrix is None:
            self._update_state(gray, keypoints, descriptors)
            return self._create_unknown_state(frame_index)
        
        # Extract camera motion parameters
        zoom_level, pan, tilt = self._decompose_transformation(matrix, frame.shape)
        
        # Update history
        self.zoom_history.append(zoom_level)
        self.motion_history.append(pan)
        
        if len(self.zoom_history) > self.history_size:
            self.zoom_history.pop(0)
            self.motion_history.pop(0)
        
        # Determine stability
        avg_motion = np.sqrt(pan[0]**2 + pan[1]**2)
        is_stable = avg_motion < self.motion_threshold
        
        # Detect zoom changes
        is_zoomed = abs(zoom_level - 1.0) > self.zoom_change_threshold
        
        # Update state
        self._update_state(gray, keypoints, descriptors)
        
        return CameraState(
            frame_index=frame_index,
            zoom_level=zoom_level,
            pan_velocity=pan,
            tilt_velocity=tilt,
            is_stable=is_stable,
            is_zoomed=is_zoomed,
            transformation_matrix=matrix
        )
    
    def _decompose_transformation(
        self,
        matrix: np.ndarray,
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[float, Tuple[float, float], float]:
        """
        Decompose homography matrix into camera motion parameters.
        Returns (zoom_level, (pan_x, pan_y), tilt)
        """
        h, w = frame_shape[:2]
        
        # Extract scale (zoom)
        sx = np.sqrt(matrix[0, 0]**2 + matrix[0, 1]**2)
        sy = np.sqrt(matrix[1, 0]**2 + matrix[1, 1]**2)
        zoom_level = (sx + sy) / 2
        
        # Extract translation (pan/tilt)
        pan_x = matrix[0, 2]
        pan_y = matrix[1, 2]
        
        # Tilt is primarily vertical motion
        tilt = pan_y
        
        return zoom_level, (pan_x, pan_y), tilt
    
    def _update_state(
        self,
        gray: np.ndarray,
        keypoints,
        descriptors
    ) -> None:
        """Update internal state"""
        self.previous_frame = gray
        self.previous_keypoints = keypoints
        self.previous_descriptors = descriptors
    
    def _create_unknown_state(self, frame_index: int) -> CameraState:
        """Create state when analysis fails"""
        return CameraState(
            frame_index=frame_index,
            zoom_level=1.0,
            pan_velocity=(0, 0),
            tilt_velocity=0,
            is_stable=False,  # Assume unstable when unknown
            is_zoomed=False,
            transformation_matrix=None
        )
    
    def get_smoothed_zoom(self) -> float:
        """Get smoothed zoom level from history"""
        if not self.zoom_history:
            return 1.0
        return np.median(self.zoom_history)
    
    def get_smoothed_motion(self) -> Tuple[float, float]:
        """Get smoothed motion from history"""
        if not self.motion_history:
            return (0, 0)
        motions = np.array(self.motion_history)
        return (float(np.median(motions[:, 0])), float(np.median(motions[:, 1])))
    
    def detect_camera_cut(
        self,
        prev_state: CameraState,
        curr_state: CameraState
    ) -> bool:
        """Detect if there was a camera cut between two states"""
        if prev_state is None or curr_state is None:
            return False
        
        # Large zoom change
        zoom_change = abs(curr_state.zoom_level - prev_state.zoom_level)
        if zoom_change > 0.3:
            return True
        
        # Large motion
        motion = np.sqrt(
            curr_state.pan_velocity[0]**2 + 
            curr_state.pan_velocity[1]**2
        )
        if motion > 50:  # Large sudden motion
            return True
        
        return False
    
    def is_suitable_for_var(self, state: CameraState) -> bool:
        """
        Check if camera state is suitable for VAR analysis.
        Requires stable camera and reasonable zoom.
        """
        if not state.is_stable:
            return False
        
        # Avoid extreme zoom levels
        if state.zoom_level < 0.5 or state.zoom_level > 2.0:
            return False
        
        return True
    
    def reset(self) -> None:
        """Reset analyzer state"""
        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.zoom_history.clear()
        self.motion_history.clear()


def compute_frame_sharpness(frame: np.ndarray) -> float:
    """
    Compute frame sharpness using Laplacian variance.
    Higher value = sharper image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def estimate_blur_amount(frame: np.ndarray) -> float:
    """
    Estimate motion blur in frame.
    Returns blur score 0-1 (0 = sharp, 1 = very blurry)
    """
    sharpness = compute_frame_sharpness(frame)
    
    # Normalize: typical range is 10-1000
    if sharpness > 500:
        return 0.0
    elif sharpness > 100:
        return 0.3
    elif sharpness > 50:
        return 0.5
    else:
        return min(1.0, 1.0 - sharpness / 50)
