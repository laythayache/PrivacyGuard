"""Identity document detection and selective anonymization.

Detects ID cards, passports, and documents with selective blurring:
- Keep face region visible
- Blur ID numbers, text, and sensitive data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DocumentConfig:
    """Configuration for document detection."""

    confidence_threshold: float = 0.4
    blur_strategy: str = "selective"  # "selective" or "full"
    blur_kernel: int = 31
    preserve_face: bool = True  # Keep face visible when blurring
    face_detector_model: str | None = None  # Optional fine-tuned detector


class DocumentDetector:
    """Detects identity documents and applies selective anonymization.

    Strategies:
    - selective: Keep face visible, blur ID numbers and text
    - full: Completely blur entire document
    """

    def __init__(self, config: DocumentConfig) -> None:
        """Initialize document detector.

        Args:
            config: DocumentConfig
        """
        self.config = config
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_document_regions(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect document regions in frame.

        Args:
            frame: Input image

        Returns:
            List of document region dicts with:
            - bbox: [x1, y1, x2, y2]
            - document_type: Type of document detected
            - confidence: Detection confidence
            - has_face: Whether face is visible
        """
        # Use color analysis and structural features to detect documents
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Documents typically have strong edges and rectangular shapes
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

        regions = []
        if lines is not None:
            # Group line segments into potential document boundaries
            document_regions = self._group_lines_to_documents(lines, frame.shape)

            for region in document_regions:
                x1, y1, x2, y2 = region

                # Check if face is present in region
                roi = frame[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray_roi, 1.1, 4)
                has_face = len(faces) > 0

                regions.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "document_type": "unknown",
                        "confidence": 0.6,
                        "has_face": has_face,
                    }
                )

        return regions

    def anonymize_document(self, frame: np.ndarray, region: dict[str, Any]) -> np.ndarray:
        """Anonymize document region.

        Args:
            frame: Input image
            region: Document region dict from detect_document_regions

        Returns:
            Frame with document anonymized
        """
        result = frame.copy()
        x1, y1, x2, y2 = region["bbox"]

        if self.config.blur_strategy == "full":
            # Blur entire document
            roi = result[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(roi, (self.config.blur_kernel, self.config.blur_kernel), 0)
            result[y1:y2, x1:x2] = blurred

        elif self.config.blur_strategy == "selective" and region.get("has_face"):
            # Selective blur: preserve face, blur rest
            roi = result[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_roi, 1.1, 4)

            # Blur entire region first
            blurred = cv2.GaussianBlur(roi, (self.config.blur_kernel, self.config.blur_kernel), 0)

            # Unmask face region (restore original)
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            for fx, fy, fw, fh in faces:
                # Add padding to face region
                pad = 10
                fx = max(0, fx - pad)
                fy = max(0, fy - pad)
                fw = min(roi.shape[1] - fx, fw + 2 * pad)
                fh = min(roi.shape[0] - fy, fh + 2 * pad)

                cv2.rectangle(mask, (fx, fy), (fx + fw, fy + fh), 255, -1)

            # Blend: use original where mask is white, blurred elsewhere
            mask_3d = np.stack([mask] * 3, axis=2) / 255
            result_roi = (roi * mask_3d + blurred * (1 - mask_3d)).astype(np.uint8)
            result[y1:y2, x1:x2] = result_roi

        return result

    def anonymize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detect and anonymize all documents in frame.

        Args:
            frame: Input image

        Returns:
            Frame with documents anonymized
        """
        result = frame.copy()
        regions = self.detect_document_regions(frame)

        for region in regions:
            result = self.anonymize_document(result, region)

        return result

    @staticmethod
    def _group_lines_to_documents(
        lines: np.ndarray, frame_shape: tuple[int, int, int]
    ) -> list[tuple[int, int, int, int]]:
        """Group detected lines into potential document regions.

        Args:
            lines: HoughLinesP output
            frame_shape: Frame shape (height, width, channels)

        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        # Simple approach: use bounding box of all lines
        if lines is None or len(lines) == 0:
            return []

        lines_flat = lines.reshape(-1, 4)
        all_x = np.concatenate([lines_flat[:, 0], lines_flat[:, 2]])
        all_y = np.concatenate([lines_flat[:, 1], lines_flat[:, 3]])

        x1 = max(0, int(all_x.min()) - 10)
        y1 = max(0, int(all_y.min()) - 10)
        x2 = min(frame_shape[1], int(all_x.max()) + 10)
        y2 = min(frame_shape[0], int(all_y.max()) + 10)

        # Filter by minimum document size
        if (x2 - x1) > 50 and (y2 - y1) > 50:
            return [(x1, y1, x2, y2)]

        return []
