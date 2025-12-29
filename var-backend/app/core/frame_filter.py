"""
Frame Eligibility Filtering Module
Scores frames based on visibility and usability for VAR analysis
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import logging

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class FrameScore:
    """Frame eligibility scoring result"""
    frame_index: int
    total_score: float
    grass_score: float
    lines_score: float
    players_score: float
    stability_score: float
    is_eligible: bool
    reason: str


class FrameFilter:
    """
    Filter and score frames for VAR eligibility.
    
    Scoring factors:
    - Grass visibility (green pixel ratio)
    - Field lines detection
    - Player count detection
    - Camera stability
    
    Score = 0.30*grass + 0.25*lines + 0.25*players + 0.20*stability
    Eligible if score >= 0.70
    """
    
    def __init__(
        self,
        grass_weight: float = None,
        lines_weight: float = None,
        players_weight: float = None,
        stability_weight: float = None,
        threshold: float = None
    ):
        self.grass_weight = grass_weight or settings.GRASS_WEIGHT
        self.lines_weight = lines_weight or settings.LINES_WEIGHT
        self.players_weight = players_weight or settings.PLAYERS_WEIGHT
        self.stability_weight = stability_weight or settings.STABILITY_WEIGHT
        self.threshold = threshold or settings.ELIGIBILITY_THRESHOLD
        
        # State for stability calculation
        self.previous_frame: Optional[np.ndarray] = None
        self.previous_keypoints = None
    
    def compute_grass_score(self, frame: np.ndarray) -> float:
        """
        Compute grass visibility score based on green pixel ratio.
        Returns score 0-1.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green color range for grass
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate ratio
        green_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        green_ratio = green_pixels / total_pixels
        
        # Score: expect 30-70% green for good field visibility
        if green_ratio < 0.2:
            return green_ratio / 0.2 * 0.5  # Low visibility
        elif green_ratio < 0.4:
            return 0.5 + (green_ratio - 0.2) / 0.2 * 0.5
        elif green_ratio <= 0.7:
            return 1.0  # Optimal range
        else:
            # Too much green might mean too close/zoomed
            return max(0.5, 1.0 - (green_ratio - 0.7) / 0.3)
    
    def compute_lines_score(self, frame: np.ndarray) -> float:
        """
        Detect field lines using Hough transform.
        Returns score 0-1 based on line detection quality.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        # Score based on number and length of lines
        num_lines = len(lines)
        
        # Calculate total line length
        total_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            total_length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # Normalize by frame diagonal
        frame_diagonal = np.sqrt(frame.shape[0] ** 2 + frame.shape[1] ** 2)
        normalized_length = total_length / frame_diagonal
        
        # Score: expect 5-20 significant lines
        line_score = min(num_lines / 15, 1.0)
        length_score = min(normalized_length / 5, 1.0)
        
        return 0.5 * line_score + 0.5 * length_score
    
    def compute_players_score(
        self,
        frame: np.ndarray,
        detections: Optional[List] = None
    ) -> float:
        """
        Score based on detected players.
        If detections provided, use them; otherwise use heuristic.
        """
        if detections is not None:
            # Count player detections
            player_count = sum(1 for d in detections if d.get('class') in ['player', 'goalkeeper'])
            
            # Score: expect 5-22 players visible for good view
            if player_count < 3:
                return player_count / 3 * 0.5
            elif player_count <= 22:
                return 0.5 + min((player_count - 3) / 15, 0.5)
            else:
                return 1.0
        
        # Heuristic: use contour detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Non-green areas (potential players)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        non_green = cv2.bitwise_not(green_mask)
        
        # Find contours
        contours, _ = cv2.findContours(non_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (player-sized objects)
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * 0.001  # 0.1% of frame
        max_area = frame_area * 0.05   # 5% of frame
        
        player_like_contours = [
            c for c in contours
            if min_area < cv2.contourArea(c) < max_area
        ]
        
        count = len(player_like_contours)
        
        if count < 3:
            return count / 3 * 0.5
        elif count <= 30:
            return 0.5 + min((count - 3) / 20, 0.5)
        else:
            return 0.8  # Many objects, might be noisy
    
    def compute_stability_score(self, frame: np.ndarray) -> float:
        """
        Compute camera stability using optical flow / feature matching.
        Returns score 0-1 (1 = very stable).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return 1.0  # Assume stable on first frame
        
        # Compute optical flow
        try:
            flow = cv2.calcOpticalFlowFarneback(
                self.previous_frame,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # Calculate flow magnitude
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_motion = np.mean(mag)
            
            # Update previous frame
            self.previous_frame = gray
            
            # Score: low motion = stable
            # Threshold: < 2 pixels avg motion = very stable
            if mean_motion < 1:
                return 1.0
            elif mean_motion < 3:
                return 1.0 - (mean_motion - 1) / 4
            elif mean_motion < 10:
                return 0.5 - (mean_motion - 3) / 14
            else:
                return 0.1  # Very unstable
                
        except Exception as e:
            logger.warning(f"Optical flow calculation failed: {e}")
            self.previous_frame = gray
            return 0.5  # Unknown stability
    
    def score_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        detections: Optional[List] = None
    ) -> FrameScore:
        """
        Compute overall eligibility score for a frame.
        """
        grass = self.compute_grass_score(frame)
        lines = self.compute_lines_score(frame)
        players = self.compute_players_score(frame, detections)
        stability = self.compute_stability_score(frame)
        
        total = (
            self.grass_weight * grass +
            self.lines_weight * lines +
            self.players_weight * players +
            self.stability_weight * stability
        )
        
        is_eligible = total >= self.threshold
        
        # Generate reason
        if is_eligible:
            reason = "Frame meets eligibility criteria"
        else:
            issues = []
            if grass < 0.5:
                issues.append("low grass visibility")
            if lines < 0.3:
                issues.append("field lines not detected")
            if players < 0.4:
                issues.append("insufficient player detection")
            if stability < 0.4:
                issues.append("camera unstable")
            reason = "Issues: " + ", ".join(issues) if issues else "Low overall score"
        
        return FrameScore(
            frame_index=frame_index,
            total_score=total,
            grass_score=grass,
            lines_score=lines,
            players_score=players,
            stability_score=stability,
            is_eligible=is_eligible,
            reason=reason
        )
    
    def filter_frames(
        self,
        frames: List[Tuple[int, np.ndarray]],
        detections_map: Optional[dict] = None
    ) -> Tuple[List[int], List[FrameScore]]:
        """
        Filter batch of frames and return eligible frame indices.
        
        Args:
            frames: List of (frame_index, frame) tuples
            detections_map: Optional dict mapping frame_index to detections
            
        Returns:
            (eligible_indices, all_scores)
        """
        eligible_indices = []
        all_scores = []
        
        for frame_idx, frame in frames:
            detections = detections_map.get(frame_idx) if detections_map else None
            score = self.score_frame(frame, frame_idx, detections)
            all_scores.append(score)
            
            if score.is_eligible:
                eligible_indices.append(frame_idx)
        
        logger.info(f"Frame filtering: {len(eligible_indices)}/{len(frames)} eligible")
        
        return eligible_indices, all_scores
    
    def reset(self) -> None:
        """Reset filter state"""
        self.previous_frame = None
        self.previous_keypoints = None
