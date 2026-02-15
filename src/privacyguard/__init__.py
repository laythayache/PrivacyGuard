"""PrivacyGuard - Privacy-first edge AI de-identification pipeline."""

from .anonymizer import Anonymizer, Method
from .core import PrivacyGuard
from .detector import Detection, ONNXDetector
from .ensemble import EnsembleConfig, EnsembleDetector
from .metadata import MetadataStripper
from .profiler import Profiler, ProfileReport
from .stream import VideoStream

__version__ = "0.2.0"
__all__ = [
    "PrivacyGuard",
    "ONNXDetector",
    "EnsembleDetector",
    "EnsembleConfig",
    "Detection",
    "Anonymizer",
    "Method",
    "VideoStream",
    "MetadataStripper",
    "Profiler",
    "ProfileReport",
]
