"""
VAR Pipeline Service
Main orchestrator that runs the complete VAR analysis pipeline
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..core.config import settings
from ..core.video_loader import VideoLoader
from ..core.scene_detector import SceneDetector, ContentClassifier
from ..core.frame_filter import FrameFilter, FrameScore
from ..core.camera_analyzer import CameraAnalyzer

from ..ai.detector import SoccerDetector, FrameDetections
from ..ai.tracker import SoccerTracker, TrackedFrame
from ..ai.pose import PoseEstimator, FramePoses
from ..ai.field_calibration import FieldCalibrator
from ..ai.ball_tracker import BallTracker, PassEvent
from ..ai.decision_engine import (
    DecisionEngine, VARDecision, 
    OffsideAnalysis, HandballAnalysis,
    create_clip_result
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for VAR pipeline"""
    analyze_offside: bool = True
    analyze_handball: bool = True
    frame_skip: int = 1  # Process every Nth frame
    min_frame_quality: float = 0.5
    save_debug_frames: bool = False


@dataclass
class PipelineResult:
    """Result from VAR pipeline analysis"""
    clip_id: str
    decisions: List[VARDecision]
    frames_processed: int
    frames_eligible: int
    pass_moments_detected: int
    processing_time: float
    
    def to_dict(self) -> dict:
        return create_clip_result(self.clip_id, self.decisions)


class VARPipeline:
    """
    Complete VAR analysis pipeline.
    
    Pipeline steps:
    1. Load video and extract frames
    2. Scene cut detection
    3. Frame eligibility filtering
    4. Object detection (YOLO)
    5. Object tracking (ByteTrack)
    6. Field calibration
    7. Ball tracking + pass detection
    8. Pose estimation (for handball)
    9. Offside analysis
    10. Handball analysis
    11. Decision generation
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.scene_detector = SceneDetector()
        self.content_classifier = ContentClassifier()
        self.frame_filter = FrameFilter()
        self.camera_analyzer = CameraAnalyzer()
        self.detector = SoccerDetector()
        self.tracker = SoccerTracker()
        self.pose_estimator = PoseEstimator()
        self.field_calibrator = FieldCalibrator()
        self.ball_tracker = BallTracker()
        self.decision_engine = DecisionEngine()
        
        logger.info("VAR Pipeline initialized")
    
    def analyze(self, video_path: str, clip_id: str = None) -> PipelineResult:
        """
        Run complete VAR analysis on video.
        
        Args:
            video_path: Path to video file
            clip_id: Optional clip identifier
            
        Returns:
            PipelineResult with decisions
        """
        import time
        start_time = time.time()
        
        video_path = Path(video_path)
        clip_id = clip_id or video_path.stem
        
        logger.info(f"Starting VAR analysis for: {clip_id}")
        
        # Step 1: Load video
        with VideoLoader(str(video_path)) as loader:
            clip_info = loader.get_clip_info()
            logger.info(f"Video loaded: {clip_info['frame_count']} frames, {clip_info['fps']} fps")
            
            # Initialize tracking data
            all_detections: List[FrameDetections] = []
            all_tracks: List[TrackedFrame] = []
            eligible_frames: List[int] = []
            frame_scores: List[FrameScore] = []
            pass_moments: List[PassEvent] = []
            
            # Step 2-7: Process frames
            frame_count = 0
            
            for frame_idx, frame in loader.iterate_frames(skip=self.config.frame_skip):
                frame_count += 1
                
                # Scene cut detection
                scene_cut = self.scene_detector.detect_cut(frame, frame_idx)
                if scene_cut:
                    logger.debug(f"Scene cut at frame {frame_idx}")
                    self.tracker.on_scene_cut()
                    self.ball_tracker.reset()
                    continue
                
                # Content classification
                content_class, content_conf = self.content_classifier.classify(frame)
                if content_class != "field_play":
                    continue
                
                # Camera analysis
                camera_state = self.camera_analyzer.analyze_frame(frame, frame_idx)
                
                # Object detection
                detections = self.detector.detect(frame, frame_idx)
                all_detections.append(detections)
                
                # Frame eligibility scoring
                score = self.frame_filter.score_frame(
                    frame, frame_idx,
                    [d.to_dict() for d in detections.detections]
                )
                frame_scores.append(score)
                
                if not score.is_eligible:
                    continue
                
                eligible_frames.append(frame_idx)
                
                # Object tracking
                tracked = self.tracker.update(detections)
                all_tracks.append(tracked)
                
                # Ball tracking
                ball = detections.ball
                if ball:
                    ball_state = self.ball_tracker.update(
                        frame_idx, ball.center, ball.confidence
                    )
                else:
                    ball_state = self.ball_tracker.update(frame_idx, None)
            
            # Get pass moments
            pass_moments = self.ball_tracker.get_pass_moments()
            logger.info(f"Detected {len(pass_moments)} pass moments")
            
            # Step 8-11: Analysis and decisions
            decisions = []
            
            # Offside analysis
            if self.config.analyze_offside and pass_moments:
                offside_decisions = self._analyze_offside(
                    loader, all_tracks, pass_moments, frame_scores
                )
                decisions.extend(offside_decisions)
            
            # Handball analysis
            if self.config.analyze_handball:
                handball_decisions = self._analyze_handball(
                    loader, all_tracks, all_detections, frame_scores
                )
                decisions.extend(handball_decisions)
            
            # Combine and finalize decisions
            decisions = self.decision_engine.combine_decisions(decisions)
        
        processing_time = time.time() - start_time
        
        result = PipelineResult(
            clip_id=clip_id,
            decisions=decisions,
            frames_processed=frame_count,
            frames_eligible=len(eligible_frames),
            pass_moments_detected=len(pass_moments),
            processing_time=processing_time
        )
        
        logger.info(f"Analysis complete: {len(decisions)} decisions in {processing_time:.2f}s")
        
        return result
    
    def _analyze_offside(
        self,
        loader: VideoLoader,
        tracks: List[TrackedFrame],
        pass_moments: List[PassEvent],
        frame_scores: List[FrameScore]
    ) -> List[VARDecision]:
        """Analyze potential offside situations"""
        decisions = []
        
        for pass_event in pass_moments:
            # Find tracked frame at pass moment
            tracked = self._find_tracked_frame(tracks, pass_event.frame_index)
            if tracked is None:
                continue
            
            # Find frame score
            score = self._find_frame_score(frame_scores, pass_event.frame_index)
            frame_quality = score.total_score if score else 0.5
            
            # Get player positions
            players = tracked.players
            if len(players) < 3:  # Need at least 2 players + ball
                continue
            
            # Try to identify attacking player (closest to ball)
            ball_pos = pass_event.position
            attacking_player = min(
                players,
                key=lambda p: np.sqrt(
                    (p.center[0] - ball_pos[0])**2 +
                    (p.center[1] - ball_pos[1])**2
                )
            )
            
            # Find defenders (players far from ball in opposite direction)
            other_players = [p for p in players if p.track_id != attacking_player.track_id]
            if not other_players:
                continue
            
            # Simplified: assume leftmost/rightmost non-attacking player is last defender
            last_defender = max(other_players, key=lambda p: p.center[0])
            
            # Create analysis
            analysis = OffsideAnalysis(
                frame_index=pass_event.frame_index,
                attacker_position=attacking_player.center,
                last_defender_position=last_defender.center,
                second_last_defender_position=None,
                ball_position=ball_pos,
                is_pass_moment=True,
                frame_quality_score=frame_quality,
                camera_stable=True,
                field_calibration_confidence=0.7
            )
            
            decision = self.decision_engine.decide_offside(analysis)
            decisions.append(decision)
        
        return decisions
    
    def _analyze_handball(
        self,
        loader: VideoLoader,
        tracks: List[TrackedFrame],
        detections: List[FrameDetections],
        frame_scores: List[FrameScore]
    ) -> List[VARDecision]:
        """Analyze potential handball situations"""
        decisions = []
        
        for i, tracked in enumerate(tracks):
            ball = tracked.ball
            if ball is None:
                continue
            
            frame_idx = tracked.frame_index
            
            # Get frame
            frame = loader.get_frame(frame_idx)
            if frame is None:
                continue
            
            # Find frame score
            score = self._find_frame_score(frame_scores, frame_idx)
            frame_quality = score.total_score if score else 0.5
            
            # Check each player for potential handball
            for player in tracked.players:
                # Calculate distance to ball
                distance = np.sqrt(
                    (player.center[0] - ball.center[0])**2 +
                    (player.center[1] - ball.center[1])**2
                )
                
                # Only analyze if ball is close
                if distance > 100:  # pixels
                    continue
                
                # Get pose estimation
                pose = self.pose_estimator.estimate_single(
                    frame, player.bbox, player.track_id
                )
                
                if pose is None:
                    continue
                
                # Get hand positions
                left_hand, right_hand = pose.get_hand_positions()
                
                # Check each hand
                for hand_pos, hand_name in [(left_hand, "left"), (right_hand, "right")]:
                    if hand_pos is None:
                        continue
                    
                    hand_ball_dist = np.sqrt(
                        (hand_pos[0] - ball.center[0])**2 +
                        (hand_pos[1] - ball.center[1])**2
                    )
                    
                    # Potential contact if very close
                    contact_detected = hand_ball_dist < 50  # pixels
                    
                    if not contact_detected:
                        continue
                    
                    # Determine hand zone
                    from ..ai.pose import HandZoneAnalyzer
                    zone_analyzer = HandZoneAnalyzer()
                    zone = zone_analyzer.get_hand_zone(pose, hand_name)
                    
                    analysis = HandballAnalysis(
                        frame_index=frame_idx,
                        player_id=player.track_id,
                        hand_position=hand_pos,
                        ball_position=ball.center,
                        distance=hand_ball_dist,
                        arm_angle=pose.left_arm_angle if hand_name == "left" else pose.right_arm_angle,
                        arm_extended=pose.is_arm_extended,
                        arm_unnatural=pose.arm_in_unnatural_position,
                        hand_zone=zone,
                        contact_detected=contact_detected,
                        frame_quality_score=frame_quality
                    )
                    
                    decision = self.decision_engine.decide_handball(analysis)
                    
                    # Only add significant decisions
                    if decision.confidence > 0.5:
                        decisions.append(decision)
        
        return decisions
    
    def _find_tracked_frame(
        self,
        tracks: List[TrackedFrame],
        frame_index: int
    ) -> Optional[TrackedFrame]:
        """Find tracked frame by index"""
        for tracked in tracks:
            if tracked.frame_index == frame_index:
                return tracked
        # Find closest
        if tracks:
            return min(tracks, key=lambda t: abs(t.frame_index - frame_index))
        return None
    
    def _find_frame_score(
        self,
        scores: List[FrameScore],
        frame_index: int
    ) -> Optional[FrameScore]:
        """Find frame score by index"""
        for score in scores:
            if score.frame_index == frame_index:
                return score
        return None
    
    def reset(self) -> None:
        """Reset all components"""
        self.scene_detector.reset()
        self.frame_filter.reset()
        self.camera_analyzer.reset()
        self.tracker.tracker.reset()
        self.ball_tracker.reset()


def analyze_video(video_path: str, clip_id: str = None) -> dict:
    """
    Convenience function to analyze a video.
    
    Args:
        video_path: Path to video file
        clip_id: Optional clip identifier
        
    Returns:
        JSON-serializable result dict
    """
    pipeline = VARPipeline()
    result = pipeline.analyze(video_path, clip_id)
    return result.to_dict()
