"""
Advanced Ball Detector
Combines multiple methods for accurate ball detection:
1. YOLO detection
2. Color-based detection (white ball on green field)
3. Motion-based detection (ball moves fast)
4. Shape analysis (ball is circular)
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class BallDetection:
    """Ball detection result"""
    position: Tuple[int, int]  # (x, y) center
    confidence: float
    radius: int
    method: str  # 'yolo', 'color', 'motion', 'combined'
    bbox: Optional[Tuple[int, int, int, int]] = None


class AdvancedBallDetector:
    """
    Multi-method ball detector for soccer videos.
    
    Methods:
    1. YOLO - Deep learning object detection
    2. Color - HSV color filtering for white ball
    3. Motion - Frame difference for moving objects
    4. Shape - Circular object detection
    """
    
    def __init__(self, use_yolo: bool = True):
        self.use_yolo = use_yolo
        self.yolo_model = None
        
        # History for motion detection
        self.prev_frame = None
        self.prev_positions = deque(maxlen=10)
        
        # Ball size constraints (in pixels)
        self.min_ball_radius = 5
        self.max_ball_radius = 50
        
        # Color ranges for ball detection (HSV)
        # White ball
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 50, 255])
        
        # Orange/Yellow ball (some matches)
        self.orange_lower = np.array([10, 100, 100])
        self.orange_upper = np.array([25, 255, 255])
        
        # Green field color (to exclude from ball search)
        self.green_lower = np.array([35, 40, 40])
        self.green_upper = np.array([85, 255, 255])
        
        if use_yolo:
            self._load_yolo()
    
    def _load_yolo(self):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
            logger.info("YOLO model loaded for ball detection")
        except Exception as e:
            logger.warning(f"Could not load YOLO: {e}")
            self.yolo_model = None
    
    def detect_ball_yolo(self, frame: np.ndarray) -> Optional[BallDetection]:
        """Detect ball using YOLO"""
        if self.yolo_model is None:
            return None
        
        results = self.yolo_model(frame, verbose=False, conf=0.2)[0]
        
        best_ball = None
        best_conf = 0
        
        for box in results.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            
            # Class 32 = sports ball in COCO
            if cls == 32 and conf > best_conf:
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max(x2 - x1, y2 - y1) // 2
                
                best_ball = BallDetection(
                    position=(cx, cy),
                    confidence=conf,
                    radius=radius,
                    method='yolo',
                    bbox=(x1, y1, x2, y2)
                )
                best_conf = conf
        
        return best_ball
    
    def detect_ball_color(self, frame: np.ndarray) -> Optional[BallDetection]:
        """Detect ball using color filtering
        NOTE: Excludes yellow objects (captain armbands) from detection
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        # Create mask for white objects
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # EXCLUDE yellow objects (captain armbands, ads)
        yellow_lower = np.array([15, 100, 100])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow_mask = cv2.dilate(yellow_mask, np.ones((20, 20), np.uint8))
        white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(yellow_mask))
        
        # Create mask for green field
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Dilate green mask to cover field
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.dilate(green_mask, kernel, iterations=2)
        
        # Ball should be white AND near green (on field)
        # Create a "near green" mask
        near_green = cv2.dilate(green_mask, np.ones((50, 50), np.uint8), iterations=1)
        
        # Final mask: white objects near the green field
        ball_mask = cv2.bitwise_and(white_mask, near_green)
        
        # Clean up mask
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Size filter
            if area < 50 or area > 5000:
                continue
            
            # Get enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            
            if radius < self.min_ball_radius or radius > self.max_ball_radius:
                continue
            
            # Circularity check
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            if circularity < 0.5:  # Not circular enough
                continue
            
            # Score based on circularity and position (prefer center-lower of frame)
            position_score = 1 - abs(x / w - 0.5) * 0.5  # Prefer center horizontally
            score = circularity * position_score
            
            if score > best_score:
                best_score = score
                best_ball = BallDetection(
                    position=(int(x), int(y)),
                    confidence=min(circularity, 0.95),
                    radius=int(radius),
                    method='color'
                )
        
        return best_ball
    
    def detect_ball_motion(self, frame: np.ndarray) -> Optional[BallDetection]:
        """Detect ball using motion detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return None
        
        # Frame difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray.copy()
        
        # Threshold
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for small, fast-moving circular objects
        candidates = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area < 30 or area > 3000:
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            
            if radius < 3 or radius > 40:
                continue
            
            # Circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            if circularity > 0.4:
                candidates.append({
                    'position': (int(x), int(y)),
                    'radius': int(radius),
                    'circularity': circularity,
                    'area': area
                })
        
        if not candidates:
            return None
        
        # Select best candidate (most circular, reasonable size)
        best = max(candidates, key=lambda c: c['circularity'])
        
        return BallDetection(
            position=best['position'],
            confidence=best['circularity'] * 0.7,  # Motion is less reliable
            radius=best['radius'],
            method='motion'
        )
    
    def detect(self, frame: np.ndarray) -> Optional[BallDetection]:
        """
        Detect ball using combined methods.
        Returns the most confident detection.
        """
        detections = []
        
        # Method 1: YOLO
        yolo_det = self.detect_ball_yolo(frame)
        if yolo_det and yolo_det.confidence > 0.3:
            detections.append(yolo_det)
        
        # Method 2: Color
        color_det = self.detect_ball_color(frame)
        if color_det and color_det.confidence > 0.5:
            detections.append(color_det)
        
        # Method 3: Motion
        motion_det = self.detect_ball_motion(frame)
        if motion_det and motion_det.confidence > 0.4:
            detections.append(motion_det)
        
        if not detections:
            return None
        
        # If multiple methods agree (positions close), boost confidence
        if len(detections) >= 2:
            # Check if any two detections are close
            for i, d1 in enumerate(detections):
                for d2 in detections[i+1:]:
                    dist = np.sqrt((d1.position[0] - d2.position[0])**2 + 
                                   (d1.position[1] - d2.position[1])**2)
                    
                    if dist < 50:  # Within 50 pixels
                        # Methods agree - return combined detection
                        avg_x = (d1.position[0] + d2.position[0]) // 2
                        avg_y = (d1.position[1] + d2.position[1]) // 2
                        avg_radius = (d1.radius + d2.radius) // 2
                        combined_conf = min((d1.confidence + d2.confidence) / 1.5, 0.98)
                        
                        combined = BallDetection(
                            position=(avg_x, avg_y),
                            confidence=combined_conf,
                            radius=avg_radius,
                            method='combined'
                        )
                        
                        # Save position history
                        self.prev_positions.append((avg_x, avg_y))
                        
                        return combined
        
        # Return highest confidence detection
        best = max(detections, key=lambda d: d.confidence)
        self.prev_positions.append(best.position)
        
        return best
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Get recent ball positions"""
        return list(self.prev_positions)


class SceneClassifier:
    """
    Classify video frames to filter out non-field scenes.
    
    Classes:
    - field_play: Main field view, suitable for VAR analysis
    - crowd: Spectator view
    - replay: Slow motion replay
    - close_up: Close up of player/referee
    - other: Unknown/transition
    """
    
    def __init__(self):
        # Thresholds
        self.min_grass_percentage = 0.15  # At least 15% green
        self.max_grass_percentage = 0.85  # Not more than 85% (close-up of grass)
        self.min_field_lines = 2  # At least 2 field lines visible
    
    def calculate_grass_percentage(self, frame: np.ndarray) -> float:
        """Calculate percentage of green (grass) pixels"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green grass color range
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, green_lower, green_upper)
        
        return np.sum(mask > 0) / mask.size
    
    def detect_field_lines(self, frame: np.ndarray) -> int:
        """Count visible field lines"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return 0
        
        # Filter for white lines (field markings)
        valid_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is on white area
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
            
            line_pixels = gray[mask > 0]
            if len(line_pixels) > 0 and np.mean(line_pixels) > 180:
                valid_lines += 1

        return valid_lines

    def detect_many_faces(self, frame: np.ndarray) -> bool:
        """Detect if frame has many faces (crowd scene)"""
        # Simple check: crowd scenes have many skin-colored regions
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Skin color range
        skin_lower = np.array([0, 20, 70])
        skin_upper = np.array([20, 255, 255])

        mask = cv2.inRange(hsv, skin_lower, skin_upper)
        skin_percentage = np.sum(mask > 0) / mask.size

        # Crowd scenes typically have >5% skin tones
        return skin_percentage > 0.05

    def is_zoomed_in(self, frame: np.ndarray) -> bool:
        """Check if frame is a close-up shot"""
        grass_pct = self.calculate_grass_percentage(frame)
        lines = self.detect_field_lines(frame)
        
        # Close-up: lots of grass OR no grass, few lines
        if grass_pct > 0.7 and lines < 2:
            return True
        if grass_pct < 0.1:
            return True
        
        return False
    
    def classify(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Classify frame type.
        
        Returns:
            (class_name, confidence)
        """
        grass_pct = self.calculate_grass_percentage(frame)
        line_count = self.detect_field_lines(frame)
        has_crowd = self.detect_many_faces(frame)
        is_close = self.is_zoomed_in(frame)
        
        # Decision logic
        if is_close:
            return ("close_up", 0.8)
        
        if grass_pct < 0.1 and has_crowd:
            return ("crowd", 0.85)
        
        if grass_pct < 0.1:
            return ("other", 0.6)
        
        if self.min_grass_percentage <= grass_pct <= self.max_grass_percentage:
            if line_count >= self.min_field_lines:
                confidence = min(0.9, 0.5 + grass_pct + line_count * 0.05)
                return ("field_play", confidence)
            else:
                return ("field_play", 0.6)
        
        return ("other", 0.5)
    
    def is_var_eligible(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Check if frame is suitable for VAR analysis.
        
        Returns:
            (is_eligible, reason)
        """
        scene_type, confidence = self.classify(frame)
        
        if scene_type == "field_play" and confidence >= 0.6:
            return (True, f"Field view detected ({confidence:.0%})")
        
        if scene_type == "crowd":
            return (False, "Crowd/spectator view")
        
        if scene_type == "close_up":
            return (False, "Close-up shot")
        
        if scene_type == "replay":
            return (False, "Replay footage")
        
        return (False, f"Non-field scene ({scene_type})")
