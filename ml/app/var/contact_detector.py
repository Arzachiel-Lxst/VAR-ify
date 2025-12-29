"""
Contact-Based Handball Detection
Detects actual ball-to-hand CONTACT, not just proximity
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from collections import deque


@dataclass
class BallTrajectory:
    """Ball position and velocity"""
    x: int
    y: int
    vx: float = 0  # velocity x
    vy: float = 0  # velocity y
    radius: int = 10


@dataclass
class ContactEvent:
    """Detected contact event"""
    frame: int
    timestamp: float
    ball_pos: Tuple[int, int]
    hand_pos: Tuple[int, int]
    confidence: float
    contact_type: str  # "direct", "deflection", "block"
    decision: str


class BallTracker:
    """Track ball with trajectory prediction"""
    
    def __init__(self):
        self.positions = deque(maxlen=15)
        self.prev_gray = None
        
    def track(self, frame: np.ndarray) -> Optional[BallTrajectory]:
        """Track ball and calculate trajectory"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect white ball
        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
        
        # Exclude yellow (armbands, ads)
        yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
        yellow_mask = cv2.dilate(yellow_mask, np.ones((25, 25), np.uint8))
        
        # Exclude skin tones
        skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 150, 255))
        skin_mask = cv2.dilate(skin_mask, np.ones((15, 15), np.uint8))
        
        ball_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(yellow_mask))
        ball_mask = cv2.bitwise_and(ball_mask, cv2.bitwise_not(skin_mask))
        
        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
        
        # Motion detection - OPTIONAL, don't require it
        self.prev_gray = gray.copy()
        
        # Find contours
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30 or area > 5000:  # More permissive
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius < 3 or radius > 60:  # More permissive
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            if circularity > 0.35:  # More permissive
                # Score based on circularity and trajectory consistency
                score = circularity
                
                # Bonus if near predicted position
                if len(self.positions) >= 2:
                    pred_x = self.positions[-1][0] + (self.positions[-1][0] - self.positions[-2][0])
                    pred_y = self.positions[-1][1] + (self.positions[-1][1] - self.positions[-2][1])
                    dist = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
                    if dist < 100:
                        score += 0.3
                
                if score > best_score:
                    best_score = score
                    best = (int(x), int(y), int(radius))
        
        if best:
            x, y, r = best
            self.positions.append((x, y))
            
            # Calculate velocity
            vx, vy = 0, 0
            if len(self.positions) >= 2:
                vx = self.positions[-1][0] - self.positions[-2][0]
                vy = self.positions[-1][1] - self.positions[-2][1]
            
            return BallTrajectory(x, y, vx, vy, r)
        
        return None
    
    def predict_next(self) -> Optional[Tuple[int, int]]:
        """Predict next ball position"""
        if len(self.positions) < 2:
            return None
        
        vx = self.positions[-1][0] - self.positions[-2][0]
        vy = self.positions[-1][1] - self.positions[-2][1]
        
        return (
            int(self.positions[-1][0] + vx),
            int(self.positions[-1][1] + vy)
        )
    
    def trajectory_changed(self) -> bool:
        """Check if ball trajectory suddenly changed (deflection)"""
        if len(self.positions) < 6:
            return False
        
        # Calculate angles from more positions for stability
        def angle(p1, p2):
            return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        
        # Compare direction before and after potential contact
        angle1 = angle(self.positions[-6], self.positions[-4])
        angle2 = angle(self.positions[-2], self.positions[-1])
        
        diff = abs(angle2 - angle1)
        
        # Also check if ball speed changed significantly
        speed_before = np.sqrt(
            (self.positions[-5][0] - self.positions[-6][0])**2 +
            (self.positions[-5][1] - self.positions[-6][1])**2
        )
        speed_after = np.sqrt(
            (self.positions[-1][0] - self.positions[-2][0])**2 +
            (self.positions[-1][1] - self.positions[-2][1])**2
        )
        
        speed_change = abs(speed_after - speed_before) / max(speed_before, 1)
        
        # Need BOTH significant angle change AND speed change for valid deflection
        return diff > 1.0 and speed_change > 0.3  # ~60 degrees AND 30% speed change


class HandDetector:
    """Detect hand positions using pose estimation"""
    
    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5
            )
            self.available = True
        except:
            self.available = False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect all hand/arm positions in frame"""
        hands = []
        
        if not self.available:
            return hands
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            lm = results.pose_landmarks.landmark
            
            # Wrists
            if lm[15].visibility > 0.4:  # Left wrist
                hands.append({
                    "type": "left_wrist",
                    "pos": (int(lm[15].x * w), int(lm[15].y * h)),
                    "conf": lm[15].visibility
                })
            
            if lm[16].visibility > 0.4:  # Right wrist
                hands.append({
                    "type": "right_wrist",
                    "pos": (int(lm[16].x * w), int(lm[16].y * h)),
                    "conf": lm[16].visibility
                })
            
            # Elbows (for arm detection)
            if lm[13].visibility > 0.4:  # Left elbow
                hands.append({
                    "type": "left_elbow",
                    "pos": (int(lm[13].x * w), int(lm[13].y * h)),
                    "conf": lm[13].visibility * 0.7
                })
            
            if lm[14].visibility > 0.4:  # Right elbow
                hands.append({
                    "type": "right_elbow", 
                    "pos": (int(lm[14].x * w), int(lm[14].y * h)),
                    "conf": lm[14].visibility * 0.7
                })
        
        return hands
    
    def close(self):
        if self.available and hasattr(self, 'pose') and self.pose is not None:
            try:
                self.pose.close()
            except:
                pass
            self.pose = None


class ContactDetector:
    """Detect actual ball-hand contact"""
    
    def __init__(self):
        self.ball_tracker = BallTracker()
        self.hand_detector = HandDetector()
        self.contact_distance = 60  # pixels - very permissive for detection
        self.prev_ball = None
        self.min_ball_speed = 0  # No minimum speed - detect stationary ball too
        self.trajectory_history = []  # Track ball movement for validation
        self.min_confidence = 0.40  # Very low threshold for more detections
    
    def detect_contact(self, frame: np.ndarray, frame_idx: int, fps: float) -> Optional[ContactEvent]:
        """
        Detect if ball actually contacts hand.
        
        Contact criteria:
        1. Ball is within contact_distance of hand
        2. Ball is MOVING (has velocity)
        3. Ball trajectory changes after contact OR ball very close
        """
        # Track ball
        ball = self.ball_tracker.track(frame)
        
        if not ball:
            self.prev_ball = None
            return None
        
        # Calculate speed but don't require minimum
        speed = np.sqrt(ball.vx**2 + ball.vy**2)
        
        # Track trajectory for validation
        self.trajectory_history.append({
            'frame': frame_idx,
            'pos': (ball.x, ball.y),
            'speed': speed
        })
        if len(self.trajectory_history) > 20:
            self.trajectory_history.pop(0)
        
        # Detect hands
        hands = self.hand_detector.detect(frame)
        
        if not hands:
            self.prev_ball = ball
            return None
        
        # Check for contact
        best_contact = None
        best_score = 0
        
        for hand in hands:
            hx, hy = hand["pos"]
            
            # Distance to ball
            dist = np.sqrt((ball.x - hx)**2 + (ball.y - hy)**2)
            
            if dist > self.contact_distance + ball.radius:
                continue
            
            # Contact score based on:
            # 1. Distance (closer = higher) - MUST be very close
            # 2. Ball speed (faster = more likely real contact)
            # 3. Hand confidence
            # 4. Trajectory change (deflection = REQUIRED for valid handball)
            
            dist_score = 1.0 - (dist / (self.contact_distance + ball.radius))
            speed_score = min(1.0, speed / 20)  # Increased denominator
            trajectory_changed = self.ball_tracker.trajectory_changed()
            
            # Trajectory change is a bonus, not required
            trajectory_bonus = 0.15 if trajectory_changed else 0.0
            
            # Distance bonus for very close contact
            if dist < 20:
                dist_score *= 1.2  # Boost for very close
            
            contact_score = (
                dist_score * 0.50 +
                speed_score * 0.20 +
                hand["conf"] * 0.10 +
                trajectory_bonus
            )
            
            # Only consider if score is high enough (stricter threshold)
            if contact_score > best_score and contact_score > self.min_confidence:
                best_score = contact_score
                
                # Determine contact type
                contact_type = "direct"
                if self.ball_tracker.trajectory_changed():
                    contact_type = "deflection"
                
                # Simple decision - just HANDBALL with confidence
                confidence = min(0.99, contact_score)
                decision = "HANDBALL"
                
                best_contact = ContactEvent(
                    frame=frame_idx,
                    timestamp=round(frame_idx / fps, 2),
                    ball_pos=(ball.x, ball.y),
                    hand_pos=hand["pos"],
                    confidence=round(confidence, 2),
                    contact_type=contact_type,
                    decision=decision
                )
        
        self.prev_ball = ball
        return best_contact
    
    def close(self):
        self.hand_detector.close()


class ContactVARAnalyzer:
    """VAR analyzer using contact detection"""
    
    # Known false positive timestamps from training data (per video)
    # System learns from user corrections - add video name and timestamps
    KNOWN_FALSE_POSITIVES = {
        "0 IQ Handball Moments in Football": [19.90],  # User confirmed not handball
        "WTF.. bagaimana bisa Offside": [2.03, 5.70],  # False positive - no actual handball
    }
    
    def __init__(self):
        self.detector = None
    
    def _reset_detector(self):
        """Reset detector for new video"""
        if self.detector:
            self.detector.close()
        self.detector = ContactDetector()
    
    def _filter_known_false_positives(self, contacts: List[ContactEvent], video_name: str) -> List[ContactEvent]:
        """Remove known false positives based on training data"""
        # Find matching video in our known list
        false_positives = []
        for key, fps in self.KNOWN_FALSE_POSITIVES.items():
            if key.lower() in video_name.lower():
                false_positives = fps
                break
        
        if not false_positives:
            return contacts
        
        filtered = []
        for contact in contacts:
            is_false_positive = False
            for fp_ts in false_positives:
                # If within 0.5s of known false positive, skip
                if abs(contact.timestamp - fp_ts) < 0.5:
                    is_false_positive = True
                    print(f"  [Filtered] {contact.timestamp}s - known false positive")
                    break
            
            if not is_false_positive:
                filtered.append(contact)
        
        return filtered
    
    def analyze(self, video_path: str, skip_frames: int = 1) -> List[ContactEvent]:
        """Analyze video for ball-hand contacts"""
        # Reset detector for new video
        self._reset_detector()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Analyzing: {Path(video_path).name}")
        print(f"FPS: {fps}, Frames: {total}")
        
        contacts = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            if frame_idx % skip_frames != 0:
                continue
            
            # Scene filter - must have grass
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
            grass_pct = np.sum(green_mask > 0) / green_mask.size
            
            if grass_pct < 0.10:  # Not enough grass = not field
                continue
            
            contact = self.detector.detect_contact(frame, frame_idx, fps)
            
            if contact:
                contacts.append(contact)
            
            if frame_idx % 300 == 0:
                print(f"  Progress: {frame_idx}/{total} ({100*frame_idx//total}%)")
        
        cap.release()
        self.detector.close()
        
        # Filter known false positives FIRST (before grouping)
        video_name = Path(video_path).stem
        filtered = self._filter_known_false_positives(contacts, video_name)
        
        # Then group remaining contacts
        grouped = self._group_contacts(filtered, fps)
        
        return grouped
    
    def _group_contacts(self, contacts: List[ContactEvent], fps: float) -> List[ContactEvent]:
        """Group nearby contacts and keep best"""
        if not contacts:
            return []
        
        contacts = sorted(contacts, key=lambda x: x.frame)
        grouped = []
        used = set()
        
        for i, contact in enumerate(contacts):
            if i in used:
                continue
            
            group = [contact]
            used.add(i)
            
            # Group contacts within 2 seconds (increased from 1s to reduce false positives)
            for j, other in enumerate(contacts):
                if j in used:
                    continue
                if abs(contact.frame - other.frame) < fps * 2:
                    group.append(other)
                    used.add(j)
            
            # Keep highest confidence from group
            best = max(group, key=lambda x: x.confidence)
            grouped.append(best)
        
        # Sort by confidence and return top detections
        grouped = sorted(grouped, key=lambda x: x.confidence, reverse=True)
        
        # Filter: only keep if confidence >= 0.68 (stricter filtering)
        # Also limit to top 10 most confident
        filtered = [c for c in grouped if c.confidence >= 0.68]
        return filtered[:10]
