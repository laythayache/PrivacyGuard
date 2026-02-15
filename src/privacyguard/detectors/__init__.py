"""Regional and specialized detectors."""

from .arabic_plate import ArabicPlateDetector
from .document import DocumentDetector
from .text import ArabicTextDetector

__all__ = ["ArabicPlateDetector", "DocumentDetector", "ArabicTextDetector"]
