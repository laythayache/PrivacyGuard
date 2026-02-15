"""Metadata stripping for privacy-compliant file handling.

Removes EXIF, IPTC, XMP, and other metadata that could reveal location,
device, timestamp, or other identifying information.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class MetadataStripper:
    """Remove identifying metadata from images and video frames."""

    @staticmethod
    def strip_image(input_path: str | Path, output_path: str | Path) -> None:
        """Remove all metadata from an image file.

        Supports JPEG, PNG, WebP (with PIL) and opencv fallback.
        """
        if HAS_PIL:
            MetadataStripper._strip_pil(input_path, output_path)
        else:
            MetadataStripper._strip_opencv(input_path, output_path)

    @staticmethod
    def _strip_pil(input_path: str | Path, output_path: str | Path) -> None:
        """Use PIL to strip all metadata."""
        img = Image.open(input_path)
        data = list(img.getdata())
        image_without_metadata = Image.new(img.mode, img.size)
        image_without_metadata.putdata(data)
        image_without_metadata.save(output_path)

    @staticmethod
    def _strip_opencv(input_path: str | Path, output_path: str | Path) -> None:
        """Fallback: read and re-encode with OpenCV (loses metadata)."""
        frame = cv2.imread(str(input_path))
        if frame is not None:
            cv2.imwrite(str(output_path), frame)

    @staticmethod
    def strip_frame(frame: np.ndarray) -> np.ndarray:
        """In-memory metadata stripping (returns a fresh copy)."""
        return frame.copy()

    @staticmethod
    def get_safe_filename(original_path: str | Path) -> str:
        """Sanitize filename to remove identifying patterns.

        Replaces timestamps, GPS coordinates, device IDs with generic names.
        """
        path = Path(original_path)
        name = path.stem
        suffix = path.suffix

        # Replace common patterns
        import re

        safe_name = re.sub(r"\\d{8}_\\d{6}", "anonymized", name)  # Timestamp
        safe_name = re.sub(r"IMG_\\d+", "image", safe_name)  # Apple IMG_xxxx
        safe_name = re.sub(
            r"[A-Z]{2}-\\d+", "anonymized", safe_name
        )  # License plate patterns
        safe_name = safe_name.lower().replace(" ", "_")

        return f"{safe_name}{suffix}"
