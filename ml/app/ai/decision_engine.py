"""
Decision Engine Module
Combines all AI analysis to produce final VAR decisions
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..core.config import settings

logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    OFFSIDE = "offside"
    HANDBALL = "handball"
    FOUL = "foul"
    GOAL = "goal"


class DecisionResult(str, Enum):
    YES = "YES"              # Definitive violation
    NO = "NO"                # Definitive no violation
    PROBABLE = "PROBABLE"    # Likely violation but not certain
    NOT_DECIDABLE = "NOT_DECIDABLE"  # Cannot determine


@dataclass
class VARDecision:
    """Single VAR decision"""
    type: DecisionType
    decision: DecisionResult
    confidence: float
    frame_index: int
    reason: str
    
    # Additional details
    details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "decision": self.decision.value,
            "confidence": round(self.confidence, 2),
            "frame_index": self.frame_index,
            "reason": self.reason,
            "details": self.details
        }


@dataclass
class OffsideAnalysis:
    """Offside analysis input"""
    frame_index: int
    attacker_position: Tuple[float, float]  # Field coords
    last_defender_position: Tuple[float, float]  # Field coords
    second_last_defender_position: Optional[Tuple[float, float]]
    ball_position: Tuple[float, float]
    is_pass_moment: bool
    frame_quality_score: float
    camera_stable: bool
    field_calibration_confidence: float


@dataclass
class HandballAnalysis:
    """Handball analysis input"""
    frame_index: int
    player_id: int
    hand_position: Tuple[float, float]
    ball_position: Tuple[float, float]
    distance: float  # Distance between hand and ball
    arm_angle: float
    arm_extended: bool
    arm_unnatural: bool
    hand_zone: str  # "legal" or "illegal"
    contact_detected: bool
    frame_quality_score: float


class DecisionEngine:
    """
    Main decision engine that combines all analyses.
    
    Produces decisions:
    - YES: High confidence violation detected
    - NO: High confidence no violation
    - PROBABLE: Likely violation but frame quality limits certainty
    - NOT_DECIDABLE: Cannot make decision due to poor data
    """
    
    def __init__(self):
        self.confidence_high = settings.CONFIDENCE_HIGH
        self.confidence_medium = settings.CONFIDENCE_MEDIUM
        self.confidence_low = settings.CONFIDENCE_LOW
        
        # Offside tolerance (FIFA standard ~5cm)
        self.offside_tolerance_m = settings.OFFSIDE_TOLERANCE_CM / 100
    
    def decide_offside(self, analysis: OffsideAnalysis) -> VARDecision:
        """
        Make offside decision based on analysis.
        
        Multi-hypothesis approach:
        1. Main hypothesis: based on detected positions
        2. Tolerance hypothesis: considering measurement error
        3. Alternative hypothesis: considering tracking uncertainty
        """
        # Check if we have valid data
        if not analysis.is_pass_moment:
            return VARDecision(
                type=DecisionType.OFFSIDE,
                decision=DecisionResult.NOT_DECIDABLE,
                confidence=0.0,
                frame_index=analysis.frame_index,
                reason="No pass moment detected",
                details={"error": "pass_not_detected"}
            )
        
        if analysis.frame_quality_score < 0.5:
            return VARDecision(
                type=DecisionType.OFFSIDE,
                decision=DecisionResult.NOT_DECIDABLE,
                confidence=0.0,
                frame_index=analysis.frame_index,
                reason="Frame quality too low for analysis",
                details={"frame_quality": analysis.frame_quality_score}
            )
        
        if analysis.field_calibration_confidence < 0.5:
            return VARDecision(
                type=DecisionType.OFFSIDE,
                decision=DecisionResult.NOT_DECIDABLE,
                confidence=0.0,
                frame_index=analysis.frame_index,
                reason="Field calibration unreliable",
                details={"calibration_confidence": analysis.field_calibration_confidence}
            )
        
        # Calculate offside distance
        # Using x-coordinate (along field length)
        attacker_x = analysis.attacker_position[0]
        defender_x = analysis.last_defender_position[0]
        
        # Positive distance = attacker is ahead (potential offside)
        offside_distance = attacker_x - defender_x
        
        # Multi-hypothesis analysis
        hypotheses = []
        
        # H1: Main hypothesis (exact positions)
        h1_offside = offside_distance > 0
        h1_confidence = min(abs(offside_distance) / 2.0, 1.0)  # Scale by 2m
        hypotheses.append(("main", h1_offside, h1_confidence))
        
        # H2: With tolerance (FIFA standard)
        h2_offside = offside_distance > self.offside_tolerance_m
        h2_confidence = min(abs(offside_distance - self.offside_tolerance_m) / 1.0, 1.0)
        hypotheses.append(("tolerance", h2_offside, h2_confidence))
        
        # H3: With calibration uncertainty
        uncertainty = 0.1 * (1 - analysis.field_calibration_confidence)
        h3_offside = offside_distance > uncertainty
        h3_confidence = analysis.field_calibration_confidence
        hypotheses.append(("uncertainty", h3_offside, h3_confidence))
        
        # Combine hypotheses
        offside_votes = sum(1 for _, is_off, _ in hypotheses if is_off)
        avg_confidence = np.mean([c for _, _, c in hypotheses])
        
        # Adjust confidence based on frame quality and camera stability
        quality_factor = analysis.frame_quality_score
        stability_factor = 1.0 if analysis.camera_stable else 0.7
        final_confidence = avg_confidence * quality_factor * stability_factor
        
        # Determine decision
        if offside_votes == 3 and final_confidence >= self.confidence_high:
            decision = DecisionResult.YES
            reason = "Clear offside position detected"
        elif offside_votes >= 2 and final_confidence >= self.confidence_medium:
            decision = DecisionResult.PROBABLE
            reason = "Probable offside, some measurement uncertainty"
        elif offside_votes == 0 and final_confidence >= self.confidence_high:
            decision = DecisionResult.NO
            reason = "Player clearly onside"
        elif offside_votes <= 1 and final_confidence >= self.confidence_medium:
            decision = DecisionResult.NO
            reason = "Player likely onside"
        else:
            decision = DecisionResult.NOT_DECIDABLE
            reason = "Insufficient confidence for decision"
        
        return VARDecision(
            type=DecisionType.OFFSIDE,
            decision=decision,
            confidence=final_confidence,
            frame_index=analysis.frame_index,
            reason=reason,
            details={
                "offside_distance_m": round(offside_distance, 3),
                "hypotheses_offside": offside_votes,
                "attacker_x": attacker_x,
                "defender_x": defender_x,
                "camera_stable": analysis.camera_stable,
                "frame_quality": analysis.frame_quality_score
            }
        )
    
    def decide_handball(self, analysis: HandballAnalysis) -> VARDecision:
        """
        Make handball decision based on FIFA rules.
        
        FIFA Handball Rules (simplified):
        - Ball touches hand/arm
        - Arm in unnatural position
        - Makes body unnaturally bigger
        - Hand above shoulder line
        """
        # Check frame quality
        if analysis.frame_quality_score < 0.5:
            return VARDecision(
                type=DecisionType.HANDBALL,
                decision=DecisionResult.NOT_DECIDABLE,
                confidence=0.0,
                frame_index=analysis.frame_index,
                reason="Frame quality too low",
                details={"frame_quality": analysis.frame_quality_score}
            )
        
        # Check if contact detected
        if not analysis.contact_detected:
            return VARDecision(
                type=DecisionType.HANDBALL,
                decision=DecisionResult.NO,
                confidence=0.8,
                frame_index=analysis.frame_index,
                reason="No ball-hand contact detected",
                details={"distance": analysis.distance}
            )
        
        # Score factors for handball violation
        scores = []
        reasons = []
        
        # Factor 1: Contact detection confidence
        if analysis.distance < 30:  # pixels
            contact_score = 1.0 - (analysis.distance / 30)
            scores.append(contact_score)
            reasons.append(f"Contact distance: {analysis.distance:.1f}px")
        else:
            return VARDecision(
                type=DecisionType.HANDBALL,
                decision=DecisionResult.NO,
                confidence=0.9,
                frame_index=analysis.frame_index,
                reason="Ball not close enough to hand",
                details={"distance": analysis.distance}
            )
        
        # Factor 2: Arm position (extended = more likely violation)
        if analysis.arm_extended:
            scores.append(0.8)
            reasons.append("Arm extended")
        else:
            scores.append(0.2)
        
        # Factor 3: Arm unnatural position
        if analysis.arm_unnatural:
            scores.append(0.9)
            reasons.append("Arm in unnatural position")
        else:
            scores.append(0.3)
        
        # Factor 4: Hand zone (above shoulder = violation)
        if analysis.hand_zone == "illegal":
            scores.append(1.0)
            reasons.append("Hand above shoulder line")
        else:
            scores.append(0.1)
        
        # Factor 5: Arm angle
        if analysis.arm_angle > 90:
            angle_score = min((analysis.arm_angle - 90) / 90, 1.0)
            scores.append(angle_score)
            if angle_score > 0.5:
                reasons.append(f"Arm angle: {analysis.arm_angle:.0f}°")
        else:
            scores.append(0.2)
        
        # Calculate final score
        weights = [0.25, 0.20, 0.25, 0.15, 0.15]
        handball_score = sum(s * w for s, w in zip(scores, weights))
        
        # Adjust by frame quality
        final_confidence = handball_score * analysis.frame_quality_score
        
        # Determine decision
        if final_confidence >= settings.HANDBALL_ZONE_THRESHOLD:
            decision = DecisionResult.YES
            reason = "Handball violation: " + ", ".join(reasons[:3])
        elif final_confidence >= 0.5:
            decision = DecisionResult.PROBABLE
            reason = "Possible handball: " + ", ".join(reasons[:2])
        else:
            decision = DecisionResult.NO
            reason = "No handball violation detected"
        
        return VARDecision(
            type=DecisionType.HANDBALL,
            decision=decision,
            confidence=final_confidence,
            frame_index=analysis.frame_index,
            reason=reason,
            details={
                "contact_distance": analysis.distance,
                "arm_extended": analysis.arm_extended,
                "arm_unnatural": analysis.arm_unnatural,
                "hand_zone": analysis.hand_zone,
                "arm_angle": analysis.arm_angle,
                "score_breakdown": dict(zip(
                    ["contact", "extended", "unnatural", "zone", "angle"],
                    scores
                ))
            }
        )
    
    def combine_decisions(
        self,
        decisions: List[VARDecision]
    ) -> List[VARDecision]:
        """
        Combine multiple decisions, removing duplicates and conflicts.
        """
        # Group by type
        by_type: Dict[DecisionType, List[VARDecision]] = {}
        for d in decisions:
            if d.type not in by_type:
                by_type[d.type] = []
            by_type[d.type].append(d)
        
        combined = []
        
        for decision_type, type_decisions in by_type.items():
            if len(type_decisions) == 1:
                combined.append(type_decisions[0])
            else:
                # Take highest confidence decision
                best = max(type_decisions, key=lambda d: d.confidence)
                combined.append(best)
        
        return combined


def create_clip_result(
    clip_id: str,
    decisions: List[VARDecision]
) -> dict:
    """
    Create final JSON result for a clip.
    """
    return {
        "clip_id": clip_id,
        "events": [d.to_dict() for d in decisions]
    }
