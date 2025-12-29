"""
Ball Tracker Module
Kalman Filter-based ball tracking and pass moment detection
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import deque
import logging

from ..core.config import settings
from ..utils.geometry import KalmanFilter2D, detect_velocity_change

logger = logging.getLogger(__name__)


@dataclass
class BallState:
    """Ball state at a specific frame"""
    frame_index: int
    position: Tuple[float, float]  # (x, y) in pixels
    velocity: Tuple[float, float]  # (vx, vy) pixels/frame
    speed: float  # magnitude of velocity
    is_detected: bool  # True if detected, False if predicted
    confidence: float


@dataclass 
class PassEvent:
    """Detected pass moment"""
    frame_index: int
    position: Tuple[float, float]
    velocity_before: Tuple[float, float]
    velocity_after: Tuple[float, float]
    velocity_change: float
    confidence: float


class BallTracker:
    """
    Track ball using Kalman filter for smooth trajectory.
    Detect pass moments based on velocity changes.
    """
    
    def __init__(
        self,
        velocity_threshold: float = None,
        history_size: int = 30
    ):
        self.velocity_threshold = velocity_threshold or settings.BALL_VELOCITY_THRESHOLD
        self.history_size = history_size
        
        # Kalman filter
        self.kf = KalmanFilter2D(
            process_noise=0.5,
            measurement_noise=2.0
        )
        
        # State history
        self.states: deque = deque(maxlen=history_size)
        self.pass_events: List[PassEvent] = []
        
        # Tracking state
        self.frames_since_detection = 0
        self.max_prediction_frames = 10
    
    def update(
        self,
        frame_index: int,
        detection: Optional[Tuple[float, float]],
        confidence: float = 1.0
    ) -> BallState:
        """
        Update ball tracker with new detection or predict if not detected.
        
        Args:
            frame_index: Current frame index
            detection: Ball position (x, y) or None if not detected
            confidence: Detection confidence
            
        Returns:
            Current BallState
        """
        if detection is not None:
            # Update with detection
            position = self.kf.update(detection[0], detection[1])
            self.frames_since_detection = 0
            is_detected = True
        else:
            # Predict position
            if self.frames_since_detection < self.max_prediction_frames:
                position = self.kf.predict()
                self.frames_since_detection += 1
                is_detected = False
                confidence = max(0.3, confidence - 0.1 * self.frames_since_detection)
            else:
                # Lost track
                return self._create_lost_state(frame_index)
        
        velocity = self.kf.get_velocity()
        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
        
        state = BallState(
            frame_index=frame_index,
            position=position,
            velocity=velocity,
            speed=speed,
            is_detected=is_detected,
            confidence=confidence
        )
        
        # Check for pass event
        self._check_pass_event(state)
        
        # Store state
        self.states.append(state)
        
        return state
    
    def _check_pass_event(self, current_state: BallState) -> Optional[PassEvent]:
        """Check if current state indicates a pass moment"""
        if len(self.states) < 2:
            return None
        
        prev_state = self.states[-1]
        
        # Calculate velocity change
        dv = np.sqrt(
            (current_state.velocity[0] - prev_state.velocity[0])**2 +
            (current_state.velocity[1] - prev_state.velocity[1])**2
        )
        
        # Check threshold
        if dv > self.velocity_threshold:
            pass_event = PassEvent(
                frame_index=current_state.frame_index,
                position=current_state.position,
                velocity_before=prev_state.velocity,
                velocity_after=current_state.velocity,
                velocity_change=dv,
                confidence=min(dv / self.velocity_threshold, 1.0)
            )
            
            self.pass_events.append(pass_event)
            logger.debug(f"Pass detected at frame {current_state.frame_index}, dv={dv:.2f}")
            
            return pass_event
        
        return None
    
    def _create_lost_state(self, frame_index: int) -> BallState:
        """Create state when ball is lost"""
        last_pos = self.states[-1].position if self.states else (0, 0)
        
        return BallState(
            frame_index=frame_index,
            position=last_pos,
            velocity=(0, 0),
            speed=0,
            is_detected=False,
            confidence=0.0
        )
    
    def get_pass_moments(self) -> List[PassEvent]:
        """Get all detected pass moments"""
        return self.pass_events.copy()
    
    def get_last_pass(self) -> Optional[PassEvent]:
        """Get most recent pass event"""
        return self.pass_events[-1] if self.pass_events else None
    
    def get_trajectory(self, last_n: int = None) -> List[Tuple[float, float]]:
        """Get ball trajectory from history"""
        states = list(self.states)
        if last_n:
            states = states[-last_n:]
        return [s.position for s in states]
    
    def predict_future_position(self, frames_ahead: int) -> Tuple[float, float]:
        """Predict ball position N frames in the future"""
        if not self.states:
            return (0, 0)
        
        current = self.states[-1]
        pred_x = current.position[0] + current.velocity[0] * frames_ahead
        pred_y = current.position[1] + current.velocity[1] * frames_ahead
        
        return (pred_x, pred_y)
    
    def reset(self) -> None:
        """Reset tracker state"""
        self.kf = KalmanFilter2D(process_noise=0.5, measurement_noise=2.0)
        self.states.clear()
        self.pass_events.clear()
        self.frames_since_detection = 0


def detect_pass_moments(
    ball_positions: List[Tuple[int, Tuple[float, float]]],
    threshold: float = None
) -> List[PassEvent]:
    """
    Detect pass moments from sequence of ball positions.
    
    Args:
        ball_positions: List of (frame_index, (x, y)) tuples
        threshold: Velocity change threshold
        
    Returns:
        List of PassEvent objects
    """
    threshold = threshold or settings.BALL_VELOCITY_THRESHOLD
    
    tracker = BallTracker(velocity_threshold=threshold)
    
    for frame_idx, position in ball_positions:
        tracker.update(frame_idx, position)
    
    return tracker.get_pass_moments()
