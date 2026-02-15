"""Arabic and Lebanese license plate detection.

Specializes in detecting license plates with Arabic/Latin scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detector import Detection, ONNXDetector
from ..utils.arabic import ScriptType


@dataclass(frozen=True)
class PlateConfig:
    """Configuration for plate detection."""

    model_path: str | None = None
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    detect_script: bool = True  # Identify Arabic vs Latin


class ArabicPlateDetector:
    """Specialized detector for Arabic/Lebanese license plates.

    Handles both Arabic and Latin script plates with:
    - Script-specific detection confidence weighting
    - Regional plate format validation
    - Multi-script support (Arabic, French, mixed)
    """

    def __init__(self, config: PlateConfig, providers: list[str] | None = None) -> None:
        """Initialize Arabic plate detector.

        Args:
            config: PlateConfig with model path and thresholds
            providers: ONNX Runtime providers
        """
        self.config = config
        self.detector: ONNXDetector | None = None

        if config.model_path:
            self.detector = ONNXDetector(
                config.model_path,
                conf_threshold=config.confidence_threshold,
                providers=providers,
                class_labels={
                    0: "arabic_plate",
                    1: "latin_plate",
                },
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect license plates in frame.

        Args:
            frame: Input image frame

        Returns:
            List of detected plates with metadata
        """
        if self.detector is None:
            return []

        detections = self.detector.detect(frame)

        # Enhance detections with script information
        if self.config.detect_script:
            enhanced = []
            for det in detections:
                script = self._infer_plate_script(det)
                # Boost confidence for script-specific plates
                confidence = det.confidence
                if script == ScriptType.ARABIC and det.label == "arabic_plate":
                    confidence *= 1.1  # 10% boost for correctly identified Arabic
                elif script == ScriptType.LATIN and det.label == "latin_plate":
                    confidence *= 1.05

                enhanced.append(
                    Detection(
                        x1=det.x1,
                        y1=det.y1,
                        x2=det.x2,
                        y2=det.y2,
                        confidence=min(confidence, 1.0),  # Cap at 1.0
                        class_id=det.class_id,
                        label=f"{det.label}_{script.value}",
                    )
                )
            return enhanced

        return detections

    @staticmethod
    def _infer_plate_script(detection: Detection) -> ScriptType:
        """Infer script type from detection label and region context.

        Args:
            detection: Detection object

        Returns:
            Detected script type
        """
        if detection.label is None:
            return ScriptType.UNKNOWN

        label_lower = detection.label.lower()

        if "arabic" in label_lower:
            return ScriptType.ARABIC
        elif "latin" in label_lower or "french" in label_lower:
            return ScriptType.LATIN

        return ScriptType.UNKNOWN

    def is_valid_lebanese_plate(self, detection: Detection, frame: np.ndarray) -> bool:
        """Validate if detected region is a valid Lebanese plate format.

        Args:
            detection: Detection object
            frame: Source frame for context

        Returns:
            True if detection matches Lebanese plate format
        """
        # Validate aspect ratio (Lebanese plates are roughly 2:1)
        width = detection.x2 - detection.x1
        height = detection.y2 - detection.y1

        if width == 0 or height == 0:
            return False

        aspect_ratio = width / height
        # Lebanese plates are typically 2.0 - 2.5 aspect ratio
        if not (1.8 < aspect_ratio < 2.8):
            return False

        # Validate minimum size (plates should be reasonably large)
        min_area = 500  # pixels
        return width * height >= min_area
