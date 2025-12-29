"""
Pose Estimation Module
MediaPipe-based pose detection for handball analysis
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging

from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PoseLandmark:
    """Single pose landmark"""
    name: str
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1
    z: float  # Depth (relative)
    visibility: float  # Confidence 0-1


@dataclass
class PlayerPose:
    """Full pose for a player"""
    player_id: int
    bbox: Tuple[float, float, float, float]
    landmarks: Dict[str, PoseLandmark]
    
    # Derived metrics
    left_arm_angle: float = 0
    right_arm_angle: float = 0
    is_arm_extended: bool = False
    arm_in_unnatural_position: bool = False
    
    def get_landmark(self, name: str) -> Optional[PoseLandmark]:
        return self.landmarks.get(name)
    
    def get_hand_positions(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Get (left_hand, right_hand) positions in pixel coords"""
        left = self.landmarks.get("left_wrist")
        right = self.landmarks.get("right_wrist")
        
        x1, y1, x2, y2 = self.bbox
        w, h = x2 - x1, y2 - y1
        
        left_pos = None
        right_pos = None
        
        if left and left.visibility > 0.5:
            left_pos = (x1 + left.x * w, y1 + left.y * h)
        
        if right and right.visibility > 0.5:
            right_pos = (x1 + right.x * w, y1 + right.y * h)
        
        return left_pos, right_pos


@dataclass
class FramePoses:
    """All poses for a frame"""
    frame_index: int
    poses: List[PlayerPose] = field(default_factory=list)


class PoseEstimator:
    """
    MediaPipe-based pose estimation for handball analysis.
    
    Key landmarks for handball:
    - Shoulders (11, 12)
    - Elbows (13, 14)
    - Wrists (15, 16)
    - Hips (23, 24)
    """
    
    # MediaPipe landmark indices
    LANDMARK_NAMES = {
        0: "nose",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        17: "left_pinky",
        18: "right_pinky",
        19: "left_index",
        20: "right_index",
        21: "left_thumb",
        22: "right_thumb",
        23: "left_hip",
        24: "right_hip",
    }
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.pose = None
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize MediaPipe Pose"""
        try:
            import mediapipe as mp
            
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            logger.info("MediaPipe Pose initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise
    
    def estimate_single(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        player_id: int
    ) -> Optional[PlayerPose]:
        """
        Estimate pose for a single player within bounding box.
        
        Args:
            frame: Full frame image
            bbox: Player bounding box (x1, y1, x2, y2)
            player_id: ID to associate with this pose
            
        Returns:
            PlayerPose or None if detection failed
        """
        if self.pose is None:
            return None
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bbox is within frame
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Crop player region
        player_crop = frame[y1:y2, x1:x2]
        
        if player_crop.size == 0:
            return None
        
        # Convert BGR to RGB
        player_rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
        
        # Run pose estimation
        results = self.pose.process(player_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = {}
        
        for idx, name in self.LANDMARK_NAMES.items():
            if idx < len(results.pose_landmarks.landmark):
                lm = results.pose_landmarks.landmark[idx]
                landmarks[name] = PoseLandmark(
                    name=name,
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility
                )
        
        # Create pose object
        player_pose = PlayerPose(
            player_id=player_id,
            bbox=bbox,
            landmarks=landmarks
        )
        
        # Calculate arm angles and extended state
        self._calculate_arm_metrics(player_pose)
        
        return player_pose
    
    def estimate_frame(
        self,
        frame: np.ndarray,
        player_bboxes: List[Tuple[int, Tuple[float, float, float, float]]],
        frame_index: int
    ) -> FramePoses:
        """
        Estimate poses for all players in a frame.
        
        Args:
            frame: Full frame image
            player_bboxes: List of (player_id, bbox) tuples
            frame_index: Index of the frame
            
        Returns:
            FramePoses object
        """
        poses = []
        
        for player_id, bbox in player_bboxes:
            pose = self.estimate_single(frame, bbox, player_id)
            if pose:
                poses.append(pose)
        
        return FramePoses(frame_index=frame_index, poses=poses)
    
    def _calculate_arm_metrics(self, pose: PlayerPose) -> None:
        """Calculate arm angles and extension state"""
        # Left arm angle
        left_shoulder = pose.landmarks.get("left_shoulder")
        left_elbow = pose.landmarks.get("left_elbow")
        left_wrist = pose.landmarks.get("left_wrist")
        
        if all([left_shoulder, left_elbow, left_wrist]):
            pose.left_arm_angle = self._calculate_angle(
                (left_shoulder.x, left_shoulder.y),
                (left_elbow.x, left_elbow.y),
                (left_wrist.x, left_wrist.y)
            )
        
        # Right arm angle
        right_shoulder = pose.landmarks.get("right_shoulder")
        right_elbow = pose.landmarks.get("right_elbow")
        right_wrist = pose.landmarks.get("right_wrist")
        
        if all([right_shoulder, right_elbow, right_wrist]):
            pose.right_arm_angle = self._calculate_angle(
                (right_shoulder.x, right_shoulder.y),
                (right_elbow.x, right_elbow.y),
                (right_wrist.x, right_wrist.y)
            )
        
        # Check if arm is extended
        arm_extended_threshold = settings.ARM_EXTENDED_ANGLE
        
        pose.is_arm_extended = (
            pose.left_arm_angle > 180 - arm_extended_threshold or
            pose.right_arm_angle > 180 - arm_extended_threshold
        )
        
        # Check for unnatural arm position (relative to body)
        pose.arm_in_unnatural_position = self._check_unnatural_arm_position(pose)
    
    def _calculate_angle(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float]
    ) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3.
        Returns angle in degrees.
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def _check_unnatural_arm_position(self, pose: PlayerPose) -> bool:
        """
        Check if arm is in unnatural position for playing.
        Used for handball detection according to FIFA rules.
        """
        left_shoulder = pose.landmarks.get("left_shoulder")
        left_wrist = pose.landmarks.get("left_wrist")
        right_shoulder = pose.landmarks.get("right_shoulder")
        right_wrist = pose.landmarks.get("right_wrist")
        
        # Check if wrist is higher than shoulder (arm raised)
        left_raised = False
        right_raised = False
        
        if left_shoulder and left_wrist:
            left_raised = left_wrist.y < left_shoulder.y - 0.1
        
        if right_shoulder and right_wrist:
            right_raised = right_wrist.y < right_shoulder.y - 0.1
        
        # Check if arm is away from body (horizontal spread)
        left_spread = False
        right_spread = False
        
        left_hip = pose.landmarks.get("left_hip")
        right_hip = pose.landmarks.get("right_hip")
        
        if left_shoulder and left_wrist and left_hip:
            body_width = abs(left_hip.x - left_shoulder.x)
            left_spread = abs(left_wrist.x - left_shoulder.x) > body_width * 1.5
        
        if right_shoulder and right_wrist and right_hip:
            body_width = abs(right_hip.x - right_shoulder.x)
            right_spread = abs(right_wrist.x - right_shoulder.x) > body_width * 1.5
        
        return (left_raised or right_raised or left_spread or right_spread)
    
    def release(self) -> None:
        """Release resources"""
        if self.pose:
            self.pose.close()
            self.pose = None


class HandZoneAnalyzer:
    """
    Analyze if hand contact occurred in legal/illegal zone.
    """
    
    # Legal hand zone boundary relative to shoulder
    LEGAL_ZONE_OFFSET = 0.15  # 15% of bbox height below shoulder
    
    def __init__(self):
        pass
    
    def get_hand_zone(
        self,
        pose: PlayerPose,
        hand: str = "left"
    ) -> str:
        """
        Determine which zone the hand is in.
        
        Returns:
            "legal" - below shoulder line (natural position)
            "illegal" - above shoulder line (unnatural position)
            "unknown" - cannot determine
        """
        if hand == "left":
            wrist = pose.landmarks.get("left_wrist")
            shoulder = pose.landmarks.get("left_shoulder")
        else:
            wrist = pose.landmarks.get("right_wrist")
            shoulder = pose.landmarks.get("right_shoulder")
        
        if not wrist or not shoulder:
            return "unknown"
        
        if wrist.visibility < 0.5 or shoulder.visibility < 0.5:
            return "unknown"
        
        # Compare y positions (lower y = higher in image)
        shoulder_y = shoulder.y
        wrist_y = wrist.y
        
        # Allow small tolerance below shoulder
        legal_boundary = shoulder_y - self.LEGAL_ZONE_OFFSET
        
        if wrist_y >= legal_boundary:
            return "legal"
        else:
            return "illegal"
    
    def analyze_handball_zone(
        self,
        pose: PlayerPose,
        ball_contact_hand: str
    ) -> Dict:
        """
        Analyze handball zone for potential violation.
        
        Args:
            pose: Player pose
            ball_contact_hand: "left" or "right"
            
        Returns:
            Analysis results dict
        """
        zone = self.get_hand_zone(pose, ball_contact_hand)
        
        return {
            "hand": ball_contact_hand,
            "zone": zone,
            "arm_angle": pose.left_arm_angle if ball_contact_hand == "left" else pose.right_arm_angle,
            "is_extended": pose.is_arm_extended,
            "is_unnatural": pose.arm_in_unnatural_position
        }
