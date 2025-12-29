"""
VAR Module
Video Assistant Referee system for handball and offside detection
"""
from .contact_detector import ContactVARAnalyzer, ContactEvent
from .offside_detector import OffsideVARAnalyzer, OffsideEvent

__all__ = [
    "ContactVARAnalyzer",
    "ContactEvent",
    "OffsideVARAnalyzer", 
    "OffsideEvent",
]
