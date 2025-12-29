"""
Field Calibration Module
Homography-based field calibration for accurate position mapping
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from ..core.config import settings
from ..utils.geometry import (
    homography_transform, 
    compute_homography,
    find_vanishing_point,
    line_intersection
)

logger = logging.getLogger(__name__)


# Standard soccer field dimensions in meters (FIFA regulations)
FIELD_LENGTH = 105.0  # meters
FIELD_WIDTH = 68.0    # meters
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.3
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_WIDTH = 18.3
CENTER_CIRCLE_RADIUS = 9.15
PENALTY_SPOT_DISTANCE = 11.0


@dataclass
class FieldCalibration:
    """Calibration result for field-to-image mapping"""
    homography_matrix: np.ndarray  # Image to field
    inverse_matrix: np.ndarray     # Field to image
    reprojection_error: float
    is_valid: bool
    
    # Calibration metadata
    num_correspondences: int = 0
    confidence: float = 0.0


@dataclass
class FieldPoint:
    """Point on the field with both image and field coordinates"""
    name: str
    image_coords: Tuple[float, float]
    field_coords: Tuple[float, float]  # In meters from center


class FieldCalibrator:
    """
    Calibrate camera view to soccer field coordinates.
    
    Methods:
    1. Manual calibration with known correspondences
    2. Automatic line detection and matching
    3. Template matching with field model
    
    Output: Homography matrix for pixel -> field coordinate mapping
    """
    
    # Known field points in field coordinates (origin at center, x along length)
    FIELD_TEMPLATE = {
        "center": (0, 0),
        "left_corner_top": (-FIELD_LENGTH/2, FIELD_WIDTH/2),
        "left_corner_bottom": (-FIELD_LENGTH/2, -FIELD_WIDTH/2),
        "right_corner_top": (FIELD_LENGTH/2, FIELD_WIDTH/2),
        "right_corner_bottom": (FIELD_LENGTH/2, -FIELD_WIDTH/2),
        "left_penalty_top": (-FIELD_LENGTH/2 + PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH/2),
        "left_penalty_bottom": (-FIELD_LENGTH/2 + PENALTY_AREA_LENGTH, -PENALTY_AREA_WIDTH/2),
        "right_penalty_top": (FIELD_LENGTH/2 - PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH/2),
        "right_penalty_bottom": (FIELD_LENGTH/2 - PENALTY_AREA_LENGTH, -PENALTY_AREA_WIDTH/2),
        "left_goal_line_center": (-FIELD_LENGTH/2, 0),
        "right_goal_line_center": (FIELD_LENGTH/2, 0),
    }
    
    def __init__(self):
        self.current_calibration: Optional[FieldCalibration] = None
        self.calibration_points: List[FieldPoint] = []
    
    def calibrate_from_points(
        self,
        correspondences: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> FieldCalibration:
        """
        Calibrate using manual point correspondences.
        
        Args:
            correspondences: List of (image_point, field_point) tuples
            
        Returns:
            FieldCalibration object
        """
        if len(correspondences) < 4:
            logger.warning("Need at least 4 correspondences for homography")
            return FieldCalibration(
                homography_matrix=np.eye(3),
                inverse_matrix=np.eye(3),
                reprojection_error=float('inf'),
                is_valid=False
            )
        
        # Extract points
        image_points = np.array([c[0] for c in correspondences], dtype=np.float32)
        field_points = np.array([c[1] for c in correspondences], dtype=np.float32)
        
        # Compute homography
        H, error = compute_homography(image_points, field_points)
        
        if H is None:
            return FieldCalibration(
                homography_matrix=np.eye(3),
                inverse_matrix=np.eye(3),
                reprojection_error=float('inf'),
                is_valid=False
            )
        
        # Compute inverse
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.eye(3)
        
        # Determine validity based on error
        is_valid = error < 5.0  # 5 meter reprojection error threshold
        
        # Calculate confidence
        confidence = max(0, 1 - error / 10)
        
        calibration = FieldCalibration(
            homography_matrix=H,
            inverse_matrix=H_inv,
            reprojection_error=error,
            is_valid=is_valid,
            num_correspondences=len(correspondences),
            confidence=confidence
        )
        
        self.current_calibration = calibration
        
        logger.info(f"Calibration complete: error={error:.2f}m, valid={is_valid}")
        
        return calibration
    
    def calibrate_auto(self, frame: np.ndarray) -> FieldCalibration:
        """
        Automatic calibration using line detection.
        """
        # Detect field lines
        lines = self._detect_field_lines(frame)
        
        if len(lines) < 4:
            logger.warning("Insufficient lines detected for calibration")
            return FieldCalibration(
                homography_matrix=np.eye(3),
                inverse_matrix=np.eye(3),
                reprojection_error=float('inf'),
                is_valid=False
            )
        
        # Classify lines (sidelines, goal lines, penalty box)
        classified_lines = self._classify_lines(lines, frame.shape)
        
        # Find key intersections
        intersections = self._find_key_intersections(classified_lines)
        
        if len(intersections) < 4:
            return FieldCalibration(
                homography_matrix=np.eye(3),
                inverse_matrix=np.eye(3),
                reprojection_error=float('inf'),
                is_valid=False
            )
        
        # Match intersections to field template
        correspondences = self._match_to_template(intersections)
        
        return self.calibrate_from_points(correspondences)
    
    def _detect_field_lines(
        self,
        frame: np.ndarray
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Detect field lines using Hough transform"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance white lines
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Edge detection
        edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines_raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=100,
            maxLineGap=30
        )
        
        if lines_raw is None:
            return []
        
        lines = []
        for line in lines_raw:
            x1, y1, x2, y2 = line[0]
            lines.append(((x1, y1), (x2, y2)))
        
        # Merge similar lines
        merged_lines = self._merge_similar_lines(lines)
        
        return merged_lines
    
    def _merge_similar_lines(
        self,
        lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        angle_threshold: float = 5,
        distance_threshold: float = 20
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Merge lines that are similar (same angle and close distance)"""
        if not lines:
            return []
        
        merged = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            group = [line1]
            used.add(i)
            
            # Calculate line1 angle
            (x1, y1), (x2, y2) = line1
            angle1 = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            
            for j, line2 in enumerate(lines):
                if j in used:
                    continue
                
                (x3, y3), (x4, y4) = line2
                angle2 = np.degrees(np.arctan2(y4 - y3, x4 - x3)) % 180
                
                # Check angle similarity
                angle_diff = abs(angle1 - angle2)
                angle_diff = min(angle_diff, 180 - angle_diff)
                
                if angle_diff < angle_threshold:
                    # Check distance
                    mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
                    mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
                    dist = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
                    
                    if dist < distance_threshold:
                        group.append(line2)
                        used.add(j)
            
            # Merge group into single line
            if group:
                all_points = []
                for (p1, p2) in group:
                    all_points.extend([p1, p2])
                
                # Fit line to all points
                points = np.array(all_points)
                [vx, vy, x, y] = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
                
                # Extend line
                t = 1000
                pt1 = (int(x - t * vx), int(y - t * vy))
                pt2 = (int(x + t * vx), int(y + t * vy))
                
                merged.append((pt1, pt2))
        
        return merged
    
    def _classify_lines(
        self,
        lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        frame_shape: Tuple[int, int, int]
    ) -> Dict[str, List]:
        """Classify lines into horizontal (sidelines) and vertical (goal lines)"""
        h, w = frame_shape[:2]
        
        classified = {
            "horizontal": [],  # Sidelines, penalty box
            "vertical": [],    # Goal lines
        }
        
        for line in lines:
            (x1, y1), (x2, y2) = line
            
            # Calculate angle
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            if angle < 30 or angle > 150:
                classified["horizontal"].append(line)
            elif 60 < angle < 120:
                classified["vertical"].append(line)
        
        return classified
    
    def _find_key_intersections(
        self,
        classified_lines: Dict[str, List]
    ) -> List[Tuple[str, Tuple[float, float]]]:
        """Find key intersection points (corners, penalty box corners)"""
        intersections = []
        
        horizontal = classified_lines.get("horizontal", [])
        vertical = classified_lines.get("vertical", [])
        
        for h_line in horizontal:
            for v_line in vertical:
                point = line_intersection(h_line, v_line)
                if point:
                    intersections.append(("intersection", point))
        
        return intersections
    
    def _match_to_template(
        self,
        intersections: List[Tuple[str, Tuple[float, float]]]
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Match detected intersections to field template points"""
        # Simple matching based on position in image
        # In production, use more sophisticated matching
        
        correspondences = []
        
        if len(intersections) >= 4:
            # Sort by x coordinate
            points = [p[1] for p in intersections]
            points.sort(key=lambda p: p[0])
            
            # Assume corners are detected
            # This is a simplified version - real implementation needs more logic
            if len(points) >= 4:
                # Map to field template
                template_points = [
                    self.FIELD_TEMPLATE["left_corner_top"],
                    self.FIELD_TEMPLATE["right_corner_top"],
                    self.FIELD_TEMPLATE["right_corner_bottom"],
                    self.FIELD_TEMPLATE["left_corner_bottom"],
                ]
                
                for i, point in enumerate(points[:4]):
                    correspondences.append((point, template_points[i]))
        
        return correspondences
    
    def pixel_to_field(
        self,
        pixel_coords: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Convert pixel coordinates to field coordinates (meters)"""
        if self.current_calibration is None or not self.current_calibration.is_valid:
            return None
        
        points = np.array([[pixel_coords]], dtype=np.float32)
        transformed = homography_transform(
            points.reshape(-1, 2),
            self.current_calibration.homography_matrix
        )
        
        return tuple(transformed[0])
    
    def field_to_pixel(
        self,
        field_coords: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Convert field coordinates (meters) to pixel coordinates"""
        if self.current_calibration is None or not self.current_calibration.is_valid:
            return None
        
        points = np.array([[field_coords]], dtype=np.float32)
        transformed = homography_transform(
            points.reshape(-1, 2),
            self.current_calibration.inverse_matrix
        )
        
        return tuple(transformed[0])
    
    def transform_players_to_field(
        self,
        player_positions: Dict[int, Tuple[float, float]]
    ) -> Dict[int, Tuple[float, float]]:
        """Transform all player positions to field coordinates"""
        if self.current_calibration is None or not self.current_calibration.is_valid:
            return {}
        
        field_positions = {}
        
        for player_id, pixel_pos in player_positions.items():
            field_pos = self.pixel_to_field(pixel_pos)
            if field_pos:
                field_positions[player_id] = field_pos
        
        return field_positions
    
    def get_offside_line_x(
        self,
        defender_field_x: float,
        attacking_left_to_right: bool
    ) -> float:
        """
        Get the x-coordinate for offside line based on last defender position.
        
        Args:
            defender_field_x: X position of last defender in field coords
            attacking_left_to_right: True if attack is going left to right
            
        Returns:
            X coordinate of offside line
        """
        # The offside line is at the position of the second-to-last defender
        # (goalkeeper being the last)
        return defender_field_x
    
    def is_position_offside(
        self,
        player_x: float,
        offside_line_x: float,
        attacking_left_to_right: bool,
        tolerance_cm: float = None
    ) -> Tuple[bool, float]:
        """
        Check if player position is offside.
        
        Args:
            player_x: Player x position in field coords
            offside_line_x: Offside line x position
            attacking_left_to_right: Direction of attack
            tolerance_cm: Tolerance in cm (FIFA standard)
            
        Returns:
            (is_offside, distance_from_line_cm)
        """
        tolerance_cm = tolerance_cm or settings.OFFSIDE_TOLERANCE_CM
        tolerance_m = tolerance_cm / 100
        
        if attacking_left_to_right:
            # Offside if player is ahead of (greater x than) offside line
            distance = player_x - offside_line_x
            is_offside = distance > tolerance_m
        else:
            # Offside if player is ahead of (less x than) offside line
            distance = offside_line_x - player_x
            is_offside = distance > tolerance_m
        
        return is_offside, distance * 100  # Return distance in cm
