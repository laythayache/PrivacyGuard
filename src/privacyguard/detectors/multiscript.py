"""Multi-script document processing for bilingual content.

Handles simultaneous processing of Arabic and Latin text in:
- Bilingual documents
- Mixed-language video frames
- Regional documents with dual text
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.arabic import ScriptType
from .text import ArabicTextDetector, TextDetectorConfig


@dataclass(frozen=True)
class MultiScriptConfig:
    """Configuration for multi-script processing."""

    process_arabic: bool = True
    process_latin: bool = True
    process_mixed: bool = True
    blur_kernel: int = 31
    confidence_threshold: float = 0.3
    preserve_structure: bool = True  # Maintain document layout


class MultiScriptProcessor:
    """Processes documents with multiple scripts (Arabic + Latin).

    Features:
    - Simultaneous Arabic and Latin text detection
    - Script-aware layout preservation
    - Bilingual document anonymization
    - Region-specific processing strategies
    """

    def __init__(self, config: MultiScriptConfig) -> None:
        """Initialize multi-script processor.

        Args:
            config: MultiScriptConfig
        """
        self.config = config
        self.text_detector = ArabicTextDetector(
            TextDetectorConfig(
                confidence_threshold=config.confidence_threshold,
                blur_kernel=config.blur_kernel,
            )
        )

    def analyze_document(self, frame: np.ndarray) -> dict[str, Any]:
        """Analyze document to detect scripts and structure.

        Args:
            frame: Input image

        Returns:
            Analysis dict with:
            - primary_script: Main script (ARABIC, LATIN, MIXED)
            - script_regions: Dict of regions by script type
            - bilingual: Whether document contains multiple scripts
            - confidence: Analysis confidence
        """
        regions = self.text_detector.detect_text_regions(frame)

        script_counts = {script: 0 for script in ScriptType}
        script_regions: dict[ScriptType, list[dict[str, Any]]] = {
            script: [] for script in ScriptType
        }

        for region in regions:
            script = region["script"]
            script_counts[script] += 1
            script_regions[script].append(region)

        # Determine primary script
        total_regions = len(regions)
        if total_regions == 0:
            return {
                "primary_script": ScriptType.UNKNOWN,
                "script_regions": script_regions,
                "bilingual": False,
                "confidence": 0.0,
                "text_density": 0.0,
            }

        arabic_ratio = script_counts[ScriptType.ARABIC] / total_regions
        latin_ratio = script_counts[ScriptType.LATIN] / total_regions
        mixed_ratio = script_counts[ScriptType.MIXED] / total_regions

        if arabic_ratio > 0.5:
            primary = ScriptType.ARABIC
        elif latin_ratio > 0.5:
            primary = ScriptType.LATIN
        else:
            primary = ScriptType.MIXED

        bilingual = (arabic_ratio > 0.1) and (latin_ratio > 0.1)

        # Calculate text density
        total_pixels = frame.shape[0] * frame.shape[1]
        text_pixels = sum(
            (region["bbox"][2] - region["bbox"][0]) * (region["bbox"][3] - region["bbox"][1])
            for region in regions
        )
        text_density = text_pixels / total_pixels

        return {
            "primary_script": primary,
            "script_regions": script_regions,
            "bilingual": bilingual,
            "confidence": 1.0 - mixed_ratio,
            "text_density": text_density,
            "region_count": total_regions,
        }

    def anonymize_document(
        self, frame: np.ndarray, strategy: str = "balanced"
    ) -> np.ndarray:
        """Anonymize multi-script document.

        Args:
            frame: Input image
            strategy: Anonymization strategy
                - "aggressive": Blur all text
                - "balanced": Blur non-structural text
                - "selective": Blur only sensitive info

        Returns:
            Anonymized frame
        """
        result: np.ndarray = np.array(frame, copy=True)
        analysis = self.analyze_document(frame)

        regions = analysis["script_regions"]

        if strategy == "aggressive":
            # Blur all detected text regions
            for script_regions_list in regions.values():
                for region in script_regions_list:
                    result = self._blur_region(result, region)

        elif strategy == "balanced":
            # Blur based on script and confidence
            for _script, script_regions_list in regions.items():
                for region in script_regions_list:
                    # Only blur high-confidence detections
                    if region["confidence"] > self.config.confidence_threshold:
                        result = self._blur_region(result, region)

        elif strategy == "selective":
            # Only blur sensitive information
            for _script, script_regions_list in regions.items():
                for region in script_regions_list:
                    # Skip structural elements
                    if self._is_sensitive_content(region):
                        result = self._blur_region(result, region)

        return result

    def process_mixed_document(
        self, frame: np.ndarray, arabic_strategy: str = "blur", latin_strategy: str = "preserve"
    ) -> np.ndarray:
        """Process document with different strategies for each script.

        Args:
            frame: Input image
            arabic_strategy: Strategy for Arabic text (blur, preserve, selective)
            latin_strategy: Strategy for Latin text

        Returns:
            Processed frame
        """
        result: np.ndarray = np.array(frame, copy=True)
        regions = self.text_detector.detect_text_regions(frame)

        for region in regions:
            script = region["script"]

            arabic_blur = script == ScriptType.ARABIC and arabic_strategy == "blur"
            latin_blur = script == ScriptType.LATIN and latin_strategy == "blur"

            if arabic_blur or latin_blur:
                result = self._blur_region(result, region)
            elif script == ScriptType.MIXED and (
                arabic_strategy == "blur" or latin_strategy == "blur"
            ):
                # For mixed regions, apply more conservative blur
                result = self._blur_region(result, region, kernel=21)

        return result

    @staticmethod
    def _blur_region(
        frame: np.ndarray, region: dict[str, Any], kernel: int | None = None
    ) -> np.ndarray:
        """Blur a specific region in the frame.

        Args:
            frame: Input image
            region: Region dict with bbox
            kernel: Blur kernel size

        Returns:
            Frame with blurred region
        """
        result: np.ndarray = np.array(frame, copy=True)
        x1, y1, x2, y2 = region["bbox"]

        # Add padding
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(result.shape[1], x2 + pad)
        y2 = min(result.shape[0], y2 + pad)

        roi = result[y1:y2, x1:x2]
        blur_k = kernel or 31
        blurred = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
        result[y1:y2, x1:x2] = blurred

        return result

    @staticmethod
    def _is_sensitive_content(region: dict[str, Any]) -> bool:
        """Determine if region contains sensitive information.

        Args:
            region: Region dict with text and metadata

        Returns:
            True if region is likely sensitive
        """
        text = region.get("text", "").lower()

        sensitive_keywords = [
            "رقم",  # Number
            "هوية",  # ID
            "جواز",  # Passport
            "رخصة",  # License
            "id",
            "passport",
            "number",
            "code",
        ]

        return any(kw in text for kw in sensitive_keywords)


# Import cv2 after class definition to avoid circular imports
import cv2  # noqa: E402
