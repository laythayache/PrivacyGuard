"""Arabic text detection and anonymization.

Detects Arabic/bilingual text regions and supports selective blurring.
Uses PaddleOCR for robust multi-script text detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..utils.arabic import ArabicProcessor, ScriptType


@dataclass(frozen=True)
class TextDetectorConfig:
    """Configuration for Arabic text detection."""

    use_paddle_ocr: bool = True  # Uses PaddleOCR if available
    confidence_threshold: float = 0.3
    min_text_size: int = 10  # Minimum text height in pixels
    blur_kernel: int = 31  # Blur kernel size for anonymization
    detect_arabic_only: bool = False  # Only detect Arabic text


class ArabicTextDetector:
    """Detects and anonymizes Arabic and bilingual text regions.

    Features:
    - Multi-script detection (Arabic, Latin, mixed)
    - Optional PaddleOCR integration for accuracy
    - Selective blurring by script type
    - Document-aware processing
    """

    def __init__(self, config: TextDetectorConfig) -> None:
        """Initialize Arabic text detector.

        Args:
            config: TextDetectorConfig
        """
        self.config = config
        self.ocr: Any | None = None

        if config.use_paddle_ocr:
            try:
                from paddleocr import PaddleOCR  # type: ignore[import-not-found]

                self.ocr = PaddleOCR(use_angle_cls=True, lang="ar")
            except ImportError:
                # Fallback if PaddleOCR not installed
                pass

    def detect_text_regions(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect text regions in frame.

        Args:
            frame: Input image

        Returns:
            List of text region dicts with:
            - bbox: [x1, y1, x2, y2]
            - text: Detected text
            - confidence: Detection confidence
            - script: ScriptType (ARABIC, LATIN, MIXED)
        """
        if self.ocr is None:
            return self._fallback_text_detection(frame)

        results = self.ocr.ocr(frame, cls=True)
        regions = []

        for line in results:
            if not line:
                continue

            for word_info in line:
                bbox, (text, conf) = word_info[0], word_info[1]
                if conf < self.config.confidence_threshold:
                    continue

                # Convert bbox to [x1, y1, x2, y2]
                points = np.array(bbox, dtype=np.float32)
                x1, y1 = points[:, 0].min(), points[:, 1].min()
                x2, y2 = points[:, 0].max(), points[:, 1].max()

                height = y2 - y1
                if height < self.config.min_text_size:
                    continue

                script = ArabicProcessor.detect_script(text)

                # Skip Latin if only Arabic detection enabled
                if self.config.detect_arabic_only and script == ScriptType.LATIN:
                    continue

                regions.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "text": text,
                        "confidence": float(conf),
                        "script": script,
                    }
                )

        return regions

    def anonymize_text(
        self, frame: np.ndarray, blur_only_arabic: bool = True
    ) -> np.ndarray:
        """Anonymize text regions in frame.

        Args:
            frame: Input image
            blur_only_arabic: If True, only blur Arabic text

        Returns:
            Frame with text regions blurred
        """
        result = frame.copy()
        regions = self.detect_text_regions(frame)

        for region in regions:
            if blur_only_arabic and region["script"] == ScriptType.LATIN:
                continue

            x1, y1, x2, y2 = region["bbox"]

            # Add padding
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(result.shape[1], x2 + pad)
            y2 = min(result.shape[0], y2 + pad)

            # Apply Gaussian blur
            roi = result[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(roi, (self.config.blur_kernel, self.config.blur_kernel), 0)
            result[y1:y2, x1:x2] = blurred

        return result

    def _fallback_text_detection(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Fallback text detection using contours (no OCR).

        Args:
            frame: Input image

        Returns:
            List of potential text regions (lower accuracy)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size
            if h < self.config.min_text_size or w < self.config.min_text_size:
                continue

            regions.append(
                {
                    "bbox": [x, y, x + w, y + h],
                    "text": "Unknown",
                    "confidence": 0.5,
                    "script": ScriptType.UNKNOWN,
                }
            )

        return regions
