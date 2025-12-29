"""
Geometry Utilities Module
Mathematical functions for VAR calculations
"""
import numpy as np
from typing import Tuple, List, Optional
import math


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def point_to_line_distance(
    point: Tuple[float, float],
    line_point1: Tuple[float, float],
    line_point2: Tuple[float, float]
) -> float:
    """
    Calculate perpendicular distance from point to line defined by two points.
    """
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    
    # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + (x2 - x1) * y1 - (y2 - y1) * x1)
    denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    if denominator == 0:
        return euclidean_distance(point, line_point1)
    
    return numerator / denominator


def signed_point_to_line_distance(
    point: Tuple[float, float],
    line_point1: Tuple[float, float],
    line_point2: Tuple[float, float]
) -> float:
    """
    Calculate signed distance from point to line.
    Positive = left side, Negative = right side
    """
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    
    return ((y2 - y1) * x0 - (x2 - x1) * y0 + (x2 - x1) * y1 - (y2 - y1) * x1) / \
           math.sqrt((y2 - y1)**2 + (x2 - x1)**2 + 1e-8)


def line_intersection(
    line1: Tuple[Tuple[float, float], Tuple[float, float]],
    line2: Tuple[Tuple[float, float], Tuple[float, float]]
) -> Optional[Tuple[float, float]]:
    """
    Find intersection point of two lines.
    Returns None if lines are parallel.
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None  # Parallel lines
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (x, y)


def get_bounding_box_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Get center point of bounding box (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_bounding_box_bottom(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Get bottom center point of bounding box (feet position)"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


def get_bounding_box_area(bbox: Tuple[float, float, float, float]) -> float:
    """Calculate area of bounding box"""
    x1, y1, x2, y2 = bbox
    return abs(x2 - x1) * abs(y2 - y1)


def bbox_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
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
    area1 = get_bounding_box_area(bbox1)
    area2 = get_bounding_box_area(bbox2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def homography_transform(
    points: np.ndarray,
    H: np.ndarray
) -> np.ndarray:
    """
    Apply homography transformation to points.
    
    Args:
        points: Nx2 array of points
        H: 3x3 homography matrix
    
    Returns:
        Nx2 array of transformed points
    """
    if len(points) == 0:
        return points
    
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    
    # Apply transformation
    transformed = H @ points_h.T
    
    # Convert back from homogeneous
    transformed = transformed.T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    
    return transformed


def compute_homography(
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Compute homography matrix from corresponding points.
    
    Returns:
        (H, reprojection_error)
    """
    import cv2
    
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    if H is None:
        return None, float('inf')
    
    # Calculate reprojection error
    projected = homography_transform(src_points, H)
    error = np.mean(np.linalg.norm(projected - dst_points, axis=1))
    
    return H, error


def find_vanishing_point(
    lines: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> Optional[Tuple[float, float]]:
    """
    Find vanishing point from parallel lines in image.
    Used for perspective correction.
    """
    if len(lines) < 2:
        return None
    
    intersections = []
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection = line_intersection(lines[i], lines[j])
            if intersection is not None:
                intersections.append(intersection)
    
    if not intersections:
        return None
    
    # Return median intersection (robust to outliers)
    intersections = np.array(intersections)
    return (float(np.median(intersections[:, 0])), float(np.median(intersections[:, 1])))


def create_offside_line(
    player_position: Tuple[float, float],
    goal_line_y: float,
    line_length: float = 1000
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Create offside line parallel to goal line at player's x position.
    
    Args:
        player_position: (x, y) position of the player (in field coordinates)
        goal_line_y: y-coordinate of goal line
        line_length: length of the line to draw
    
    Returns:
        Two points defining the offside line
    """
    x = player_position[0]
    
    # Create horizontal line at player's x position
    return ((x, goal_line_y - line_length / 2), (x, goal_line_y + line_length / 2))


def angle_between_vectors(
    v1: Tuple[float, float],
    v2: Tuple[float, float]
) -> float:
    """Calculate angle between two 2D vectors in degrees"""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 * mag2 == 0:
        return 0
    
    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def velocity_from_positions(
    positions: List[Tuple[float, float]],
    dt: float = 1.0
) -> List[Tuple[float, float]]:
    """
    Calculate velocity vectors from position history.
    
    Args:
        positions: List of (x, y) positions
        dt: Time delta between positions
    
    Returns:
        List of velocity vectors
    """
    if len(positions) < 2:
        return []
    
    velocities = []
    for i in range(1, len(positions)):
        vx = (positions[i][0] - positions[i-1][0]) / dt
        vy = (positions[i][1] - positions[i-1][1]) / dt
        velocities.append((vx, vy))
    
    return velocities


def acceleration_from_velocities(
    velocities: List[Tuple[float, float]],
    dt: float = 1.0
) -> List[Tuple[float, float]]:
    """Calculate acceleration from velocity history"""
    return velocity_from_positions(velocities, dt)


def smooth_trajectory(
    positions: List[Tuple[float, float]],
    window_size: int = 5
) -> List[Tuple[float, float]]:
    """
    Apply moving average smoothing to trajectory.
    """
    if len(positions) < window_size:
        return positions
    
    positions_array = np.array(positions)
    smoothed = np.zeros_like(positions_array)
    
    for i in range(len(positions)):
        start = max(0, i - window_size // 2)
        end = min(len(positions), i + window_size // 2 + 1)
        smoothed[i] = np.mean(positions_array[start:end], axis=0)
    
    return [(float(p[0]), float(p[1])) for p in smoothed]


def detect_velocity_change(
    velocities: List[Tuple[float, float]],
    threshold: float
) -> List[int]:
    """
    Detect frames where velocity change exceeds threshold.
    Used for pass moment detection.
    
    Returns:
        List of frame indices with significant velocity change
    """
    if len(velocities) < 2:
        return []
    
    change_frames = []
    
    for i in range(1, len(velocities)):
        v_prev = velocities[i - 1]
        v_curr = velocities[i]
        
        # Calculate velocity magnitude change
        mag_prev = math.sqrt(v_prev[0]**2 + v_prev[1]**2)
        mag_curr = math.sqrt(v_curr[0]**2 + v_curr[1]**2)
        
        # Also consider direction change
        delta_v = math.sqrt((v_curr[0] - v_prev[0])**2 + (v_curr[1] - v_prev[1])**2)
        
        if delta_v > threshold or abs(mag_curr - mag_prev) > threshold:
            change_frames.append(i)
    
    return change_frames


class KalmanFilter2D:
    """
    2D Kalman filter for position tracking with velocity estimation.
    Used for ball tracking and prediction.
    """
    
    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 1.0):
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # State covariance
        self.P = np.eye(4) * 1000
        
        self.initialized = False
    
    def initialize(self, x: float, y: float) -> None:
        """Initialize filter with first measurement"""
        self.state = np.array([x, y, 0, 0])
        self.initialized = True
    
    def predict(self) -> Tuple[float, float]:
        """Predict next state"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (self.state[0], self.state[1])
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Update state with measurement"""
        if not self.initialized:
            self.initialize(x, y)
            return (x, y)
        
        # Predict
        self.predict()
        
        # Measurement
        z = np.array([x, y])
        
        # Innovation
        y_innovation = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y_innovation
        
        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        return (self.state[0], self.state[1])
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate"""
        return (self.state[2], self.state[3])
    
    def get_state(self) -> Tuple[float, float, float, float]:
        """Get full state (x, y, vx, vy)"""
        return tuple(self.state)
