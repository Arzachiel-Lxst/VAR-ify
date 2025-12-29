"""
Object Tracking Module
ByteTrack-based multi-object tracking for consistent player IDs
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from .detector import Detection, FrameDetections
from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Single tracked object"""
    track_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    confidence: float
    
    # Track history
    age: int = 0  # Frames since track started
    hits: int = 0  # Successful detections
    misses: int = 0  # Consecutive missed detections
    
    # Position history for trajectory analysis
    history: List[Tuple[float, float]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0, 0)
    
    # Team assignment
    team: Optional[str] = None
    
    def update_history(self, max_history: int = 30) -> None:
        """Update position history"""
        self.history.append(self.center)
        if len(self.history) > max_history:
            self.history.pop(0)
    
    def calculate_velocity(self) -> Tuple[float, float]:
        """Calculate velocity from recent history"""
        if len(self.history) < 2:
            return (0, 0)
        
        # Use last 5 positions
        recent = self.history[-5:]
        if len(recent) < 2:
            return (0, 0)
        
        vx = (recent[-1][0] - recent[0][0]) / len(recent)
        vy = (recent[-1][1] - recent[0][1]) / len(recent)
        
        self.velocity = (vx, vy)
        return self.velocity
    
    def predict_position(self, frames_ahead: int = 1) -> Tuple[float, float]:
        """Predict future position based on velocity"""
        vx, vy = self.calculate_velocity()
        
        pred_x = self.center[0] + vx * frames_ahead
        pred_y = self.center[1] + vy * frames_ahead
        
        return (pred_x, pred_y)


@dataclass
class TrackedFrame:
    """All tracks for a single frame"""
    frame_index: int
    tracks: List[Track] = field(default_factory=list)
    
    @property
    def players(self) -> List[Track]:
        return [t for t in self.tracks if t.class_name in ["player", "goalkeeper"]]
    
    @property
    def ball(self) -> Optional[Track]:
        balls = [t for t in self.tracks if t.class_name == "ball"]
        return balls[0] if balls else None
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None


class ByteTracker:
    """
    ByteTrack-based multi-object tracker.
    
    Features:
    - Consistent ID assignment across frames
    - Handles occlusions and missed detections
    - Maintains track history for trajectory analysis
    - Resets on scene cuts
    """
    
    def __init__(
        self,
        track_buffer: int = None,
        match_threshold: float = None
    ):
        self.track_buffer = track_buffer or settings.TRACK_BUFFER
        self.match_threshold = match_threshold or settings.MATCH_THRESHOLD
        
        # State
        self.tracks: Dict[int, Track] = {}
        self.next_id: int = 1
        self.frame_count: int = 0
        
        # History
        self.track_history: Dict[int, List[Track]] = defaultdict(list)
    
    def update(self, detections: FrameDetections) -> TrackedFrame:
        """
        Update tracks with new detections.
        
        Args:
            detections: Frame detections from detector
            
        Returns:
            TrackedFrame with updated tracks
        """
        self.frame_count += 1
        
        if not detections.detections:
            # No detections - mark all tracks as missed
            self._handle_no_detections()
            return TrackedFrame(
                frame_index=detections.frame_index,
                tracks=list(self.tracks.values())
            )
        
        # Separate high and low confidence detections
        high_conf_dets = [d for d in detections.detections if d.confidence >= 0.5]
        low_conf_dets = [d for d in detections.detections if d.confidence < 0.5]
        
        # Match existing tracks with high confidence detections
        matched_track_ids, matched_det_indices, unmatched_tracks = self._match_detections(
            list(self.tracks.values()),
            high_conf_dets
        )
        
        # Update matched tracks
        for track_id, det_idx in zip(matched_track_ids, matched_det_indices):
            det = high_conf_dets[det_idx]
            self._update_track(track_id, det)
        
        # Try to match unmatched tracks with low confidence detections
        if unmatched_tracks and low_conf_dets:
            second_matched_ids, second_matched_indices, still_unmatched = self._match_detections(
                unmatched_tracks,
                low_conf_dets
            )
            
            for track_id, det_idx in zip(second_matched_ids, second_matched_indices):
                det = low_conf_dets[det_idx]
                self._update_track(track_id, det)
            
            unmatched_tracks = still_unmatched
        
        # Handle unmatched tracks (missed detections)
        for track in unmatched_tracks:
            track.misses += 1
            if track.misses > self.track_buffer:
                # Remove track
                if track.track_id in self.tracks:
                    del self.tracks[track.track_id]
        
        # Create new tracks for unmatched detections
        matched_det_set = set(matched_det_indices)
        for i, det in enumerate(high_conf_dets):
            if i not in matched_det_set:
                self._create_track(det)
        
        return TrackedFrame(
            frame_index=detections.frame_index,
            tracks=list(self.tracks.values())
        )
    
    def _match_detections(
        self,
        tracks: List[Track],
        detections: List[Detection]
    ) -> Tuple[List[int], List[int], List[Track]]:
        """
        Match tracks with detections using IoU.
        
        Returns:
            (matched_track_ids, matched_detection_indices, unmatched_tracks)
        """
        if not tracks or not detections:
            return [], [], tracks
        
        # Build cost matrix using IoU
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self._compute_iou(track.bbox, det.bbox)
                cost_matrix[i, j] = 1 - iou  # Cost = 1 - IoU
        
        # Hungarian matching
        try:
            from scipy.optimize import linear_sum_assignment
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
        except Exception:
            # Fallback to greedy matching
            track_indices, det_indices = self._greedy_match(cost_matrix)
        
        matched_track_ids = []
        matched_det_indices = []
        matched_track_set = set()
        
        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix[t_idx, d_idx] < (1 - self.match_threshold):
                matched_track_ids.append(tracks[t_idx].track_id)
                matched_det_indices.append(d_idx)
                matched_track_set.add(t_idx)
        
        unmatched_tracks = [t for i, t in enumerate(tracks) if i not in matched_track_set]
        
        return matched_track_ids, matched_det_indices, unmatched_tracks
    
    def _greedy_match(
        self,
        cost_matrix: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Greedy matching fallback"""
        track_indices = []
        det_indices = []
        
        used_tracks = set()
        used_dets = set()
        
        # Sort by cost
        costs = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                costs.append((cost_matrix[i, j], i, j))
        
        costs.sort()
        
        for cost, t_idx, d_idx in costs:
            if t_idx not in used_tracks and d_idx not in used_dets:
                track_indices.append(t_idx)
                det_indices.append(d_idx)
                used_tracks.add(t_idx)
                used_dets.add(d_idx)
        
        return track_indices, det_indices
    
    def _compute_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """Compute IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _update_track(self, track_id: int, detection: Detection) -> None:
        """Update existing track with new detection"""
        if track_id not in self.tracks:
            return
        
        track = self.tracks[track_id]
        track.bbox = detection.bbox
        track.center = detection.center
        track.confidence = detection.confidence
        track.age += 1
        track.hits += 1
        track.misses = 0
        
        if detection.team:
            track.team = detection.team
        
        track.update_history()
        track.calculate_velocity()
    
    def _create_track(self, detection: Detection) -> Track:
        """Create new track from detection"""
        track = Track(
            track_id=self.next_id,
            class_name=detection.class_name,
            bbox=detection.bbox,
            center=detection.center,
            confidence=detection.confidence,
            age=1,
            hits=1,
            misses=0,
            team=detection.team
        )
        
        track.update_history()
        
        self.tracks[self.next_id] = track
        self.next_id += 1
        
        return track
    
    def _handle_no_detections(self) -> None:
        """Handle frame with no detections"""
        to_remove = []
        
        for track_id, track in self.tracks.items():
            track.misses += 1
            if track.misses > self.track_buffer:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def reset(self) -> None:
        """Reset tracker state (e.g., on scene cut)"""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        logger.info("Tracker reset")
    
    def get_track_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """Get full trajectory for a track"""
        if track_id in self.tracks:
            return self.tracks[track_id].history.copy()
        return []
    
    def get_player_positions(self) -> Dict[int, Tuple[float, float]]:
        """Get current positions of all player tracks"""
        positions = {}
        for track_id, track in self.tracks.items():
            if track.class_name in ["player", "goalkeeper"]:
                positions[track_id] = track.center
        return positions


class SoccerTracker:
    """
    High-level tracker for soccer-specific tracking logic.
    """
    
    def __init__(self):
        self.tracker = ByteTracker()
        self.team_assignments: Dict[int, str] = {}
    
    def update(self, detections: FrameDetections) -> TrackedFrame:
        """Update with new detections"""
        tracked = self.tracker.update(detections)
        
        # Propagate team assignments
        for track in tracked.players:
            if track.track_id in self.team_assignments:
                track.team = self.team_assignments[track.track_id]
            elif track.team:
                self.team_assignments[track.track_id] = track.team
        
        return tracked
    
    def on_scene_cut(self) -> None:
        """Handle scene cut - reset tracker"""
        self.tracker.reset()
        # Keep team assignments as they may still be valid
    
    def get_last_defender(
        self,
        tracked: TrackedFrame,
        attacking_team: str,
        goal_x: float
    ) -> Optional[Track]:
        """
        Find the last defender (excluding goalkeeper).
        
        Args:
            tracked: Current tracked frame
            attacking_team: Team that is attacking ("home" or "away")
            goal_x: X-coordinate of the attacking goal
            
        Returns:
            Track of the last defender
        """
        defending_team = "away" if attacking_team == "home" else "home"
        
        defenders = [
            t for t in tracked.players
            if t.team == defending_team and t.class_name != "goalkeeper"
        ]
        
        if not defenders:
            return None
        
        # Find defender closest to own goal
        # (furthest from attacking goal in x direction)
        if goal_x > 0:  # Attacking right
            # Last defender is the one with smallest x
            last_defender = min(defenders, key=lambda t: t.center[0])
        else:  # Attacking left
            # Last defender is the one with largest x
            last_defender = max(defenders, key=lambda t: t.center[0])
        
        return last_defender
    
    def get_ball_track(self) -> Optional[Track]:
        """Get current ball track if exists"""
        for track in self.tracker.tracks.values():
            if track.class_name == "ball":
                return track
        return None
