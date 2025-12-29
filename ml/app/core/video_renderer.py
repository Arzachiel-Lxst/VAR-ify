"""
Video Renderer Module
Renders analysis results back to video with visualizations
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from .config import settings

logger = logging.getLogger(__name__)


# Colors (BGR format)
COLORS = {
    "player_home": (255, 100, 100),    # Blue-ish
    "player_away": (100, 100, 255),    # Red-ish
    "goalkeeper": (0, 255, 255),        # Yellow
    "ball": (0, 255, 0),                # Green
    "offside_line": (0, 0, 255),        # Red
    "onside_line": (0, 255, 0),         # Green
    "handball": (0, 165, 255),          # Orange
    "text_bg": (0, 0, 0),               # Black
    "text": (255, 255, 255),            # White
    "eligible": (0, 255, 0),            # Green
    "not_eligible": (0, 0, 255),        # Red
}


@dataclass
class RenderConfig:
    """Configuration for video rendering"""
    draw_players: bool = True
    draw_ball: bool = True
    draw_tracks: bool = True
    draw_offside_line: bool = True
    draw_frame_info: bool = True
    draw_decisions: bool = True
    show_confidence: bool = True
    output_fps: int = 30
    codec: str = "mp4v"


class VideoRenderer:
    """
    Render analysis results onto video frames.
    
    Features:
    - Player bounding boxes with team colors
    - Ball tracking visualization
    - Offside line overlay
    - Handball zone markers
    - Decision text overlay
    - Frame eligibility indicator
    """
    
    def __init__(self, config: RenderConfig = None):
        self.config = config or RenderConfig()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
    
    def draw_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        label: str,
        color: Tuple[int, int, int],
        confidence: float = None
    ) -> np.ndarray:
        """Draw bounding box with label"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        if confidence is not None and self.config.show_confidence:
            label = f"{label} {confidence:.0%}"
        
        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 2, y1 - 5), self.font, self.font_scale, COLORS["text"], self.thickness)
        
        return frame
    
    def draw_player(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        track_id: int,
        team: str = None,
        is_goalkeeper: bool = False
    ) -> np.ndarray:
        """Draw player with team color"""
        if is_goalkeeper:
            color = COLORS["goalkeeper"]
            label = f"GK #{track_id}"
        elif team == "home":
            color = COLORS["player_home"]
            label = f"H{track_id}"
        elif team == "away":
            color = COLORS["player_away"]
            label = f"A{track_id}"
        else:
            color = (200, 200, 200)  # Gray for unknown
            label = f"#{track_id}"
        
        return self.draw_bbox(frame, bbox, label, color)
    
    def draw_ball(
        self,
        frame: np.ndarray,
        position: Tuple[float, float],
        radius: int = 15
    ) -> np.ndarray:
        """Draw ball marker"""
        x, y = map(int, position)
        
        # Draw circle
        cv2.circle(frame, (x, y), radius, COLORS["ball"], 3)
        cv2.circle(frame, (x, y), radius + 5, COLORS["ball"], 1)
        
        # Draw crosshair
        cv2.line(frame, (x - 20, y), (x + 20, y), COLORS["ball"], 1)
        cv2.line(frame, (x, y - 20), (x, y + 20), COLORS["ball"], 1)
        
        return frame
    
    def draw_trajectory(
        self,
        frame: np.ndarray,
        positions: List[Tuple[float, float]],
        color: Tuple[int, int, int] = (0, 255, 255)
    ) -> np.ndarray:
        """Draw trajectory line"""
        if len(positions) < 2:
            return frame
        
        points = np.array(positions, dtype=np.int32)
        
        # Draw fading trajectory
        for i in range(1, len(points)):
            alpha = i / len(points)
            pt_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, tuple(points[i-1]), tuple(points[i]), pt_color, 2)
        
        return frame
    
    def draw_offside_line(
        self,
        frame: np.ndarray,
        x_position: float,
        is_offside: bool,
        label: str = "OFFSIDE LINE"
    ) -> np.ndarray:
        """Draw vertical offside line"""
        h, w = frame.shape[:2]
        x = int(x_position)
        
        color = COLORS["offside_line"] if is_offside else COLORS["onside_line"]
        
        # Draw dashed line
        dash_length = 20
        for y in range(0, h, dash_length * 2):
            cv2.line(frame, (x, y), (x, min(y + dash_length, h)), color, 3)
        
        # Draw label
        cv2.putText(frame, label, (x + 10, 50), self.font, 0.8, color, 2)
        
        return frame
    
    def draw_handball_zone(
        self,
        frame: np.ndarray,
        hand_position: Tuple[float, float],
        is_violation: bool,
        arm_extended: bool = False
    ) -> np.ndarray:
        """Draw handball detection marker"""
        x, y = map(int, hand_position)
        
        color = COLORS["handball"] if is_violation else COLORS["onside_line"]
        
        # Draw circle around hand
        cv2.circle(frame, (x, y), 30, color, 3)
        
        if is_violation:
            # Draw warning symbol
            cv2.putText(frame, "!", (x - 5, y + 5), self.font, 1, color, 3)
        
        return frame
    
    def draw_frame_info(
        self,
        frame: np.ndarray,
        frame_index: int,
        fps: float,
        is_eligible: bool,
        score: float = None
    ) -> np.ndarray:
        """Draw frame information overlay"""
        h, w = frame.shape[:2]
        
        # Background panel
        cv2.rectangle(frame, (10, 10), (300, 100), COLORS["text_bg"], -1)
        cv2.rectangle(frame, (10, 10), (300, 100), (100, 100, 100), 1)
        
        # Frame info
        time_sec = frame_index / fps if fps > 0 else 0
        cv2.putText(frame, f"Frame: {frame_index}", (20, 35), self.font, 0.6, COLORS["text"], 1)
        cv2.putText(frame, f"Time: {time_sec:.2f}s", (20, 55), self.font, 0.6, COLORS["text"], 1)
        
        # Eligibility
        status = "ELIGIBLE" if is_eligible else "NOT ELIGIBLE"
        status_color = COLORS["eligible"] if is_eligible else COLORS["not_eligible"]
        cv2.putText(frame, status, (20, 80), self.font, 0.6, status_color, 2)
        
        if score is not None:
            cv2.putText(frame, f"Score: {score:.2f}", (150, 80), self.font, 0.5, COLORS["text"], 1)
        
        return frame
    
    def draw_decision(
        self,
        frame: np.ndarray,
        decision_type: str,
        decision_result: str,
        confidence: float,
        reason: str = ""
    ) -> np.ndarray:
        """Draw decision overlay"""
        h, w = frame.shape[:2]
        
        # Large decision panel at bottom
        panel_h = 120
        cv2.rectangle(frame, (0, h - panel_h), (w, h), COLORS["text_bg"], -1)
        
        # Decision type and result
        if decision_result in ["YES", "PROBABLE"]:
            result_color = COLORS["offside_line"]  # Red
        else:
            result_color = COLORS["onside_line"]  # Green
        
        # Title
        title = f"VAR CHECK: {decision_type.upper()}"
        cv2.putText(frame, title, (20, h - panel_h + 35), self.font, 1, COLORS["text"], 2)
        
        # Result
        cv2.putText(frame, decision_result, (20, h - panel_h + 70), self.font, 1.2, result_color, 3)
        
        # Confidence bar
        bar_x = 250
        bar_w = 200
        bar_h = 20
        bar_y = h - panel_h + 55
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * confidence), bar_y + bar_h), result_color, -1)
        cv2.putText(frame, f"{confidence:.0%}", (bar_x + bar_w + 10, bar_y + 15), self.font, 0.6, COLORS["text"], 1)
        
        # Reason
        if reason:
            cv2.putText(frame, reason[:60], (20, h - 15), self.font, 0.5, (180, 180, 180), 1)
        
        return frame
    
    def render_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        detections: Dict = None,
        tracks: Dict = None,
        decision: Dict = None,
        frame_score: Dict = None,
        fps: float = 30
    ) -> np.ndarray:
        """
        Render all visualizations on a single frame.
        """
        rendered = frame.copy()
        
        # Draw detections
        if detections and self.config.draw_players:
            for det in detections.get("players", []):
                rendered = self.draw_player(
                    rendered,
                    det["bbox"],
                    det.get("track_id", 0),
                    det.get("team"),
                    det.get("is_goalkeeper", False)
                )
            
            if self.config.draw_ball and detections.get("ball"):
                ball = detections["ball"]
                rendered = self.draw_ball(rendered, ball["center"])
        
        # Draw tracks with trajectory
        if tracks and self.config.draw_tracks:
            for track in tracks.get("players", []):
                if track.get("history"):
                    rendered = self.draw_trajectory(rendered, track["history"])
        
        # Draw frame info
        if self.config.draw_frame_info:
            is_eligible = frame_score.get("is_eligible", True) if frame_score else True
            score = frame_score.get("total_score") if frame_score else None
            rendered = self.draw_frame_info(rendered, frame_index, fps, is_eligible, score)
        
        # Draw decision
        if decision and self.config.draw_decisions:
            rendered = self.draw_decision(
                rendered,
                decision.get("type", ""),
                decision.get("decision", ""),
                decision.get("confidence", 0),
                decision.get("reason", "")
            )
        
        return rendered


def create_output_video(
    input_path: str,
    output_path: str,
    analysis_results: Dict,
    config: RenderConfig = None
) -> str:
    """
    Create annotated output video with analysis visualizations.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        analysis_results: Results from VAR pipeline
        config: Render configuration
        
    Returns:
        Path to output video
    """
    config = config or RenderConfig()
    renderer = VideoRenderer(config)
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*config.codec)
    out = cv2.VideoWriter(output_path, fourcc, config.output_fps or fps, (width, height))
    
    # Get events indexed by frame
    events_by_frame = {}
    for event in analysis_results.get("events", []):
        frame_idx = event.get("frame_index", 0)
        # Show decision for 90 frames (3 seconds at 30fps)
        for i in range(frame_idx - 30, frame_idx + 60):
            events_by_frame[i] = event
    
    logger.info(f"Rendering video: {total_frames} frames")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get decision for this frame if any
        decision = events_by_frame.get(frame_idx)
        
        # Basic frame info
        frame_score = {"is_eligible": True, "total_score": 0.8}
        
        # Render frame
        rendered = renderer.render_frame(
            frame,
            frame_idx,
            decision=decision,
            frame_score=frame_score,
            fps=fps
        )
        
        out.write(rendered)
        frame_idx += 1
        
        # Progress
        if frame_idx % 500 == 0:
            logger.info(f"Rendered {frame_idx}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    logger.info(f"Output video saved: {output_path}")
    
    return output_path
