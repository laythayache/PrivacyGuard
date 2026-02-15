"""Anonymization strategies for detected sensitive regions.

Provides multiple techniques: Gaussian blur, pixelation, and solid fill.
Each can be applied independently per-detection class.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .utils.validation import (
    validate_color,
    validate_kernel_size,
    validate_non_negative,
    validate_positive,
)

if TYPE_CHECKING:
    from .detector import Detection


class Method(Enum):
    """Available anonymization methods."""

    GAUSSIAN = "gaussian"
    PIXELATE = "pixelate"
    SOLID = "solid"


# Default kernel sizes / parameters per method
_DEFAULTS = {
    Method.GAUSSIAN: {"ksize": (51, 51)},
    Method.PIXELATE: {"block_size": 10},
    Method.SOLID: {"color": (0, 0, 0)},
}


class Anonymizer:
    """Applies anonymization to detected regions within a frame.

    Args:
        method: Default anonymization method.
        class_methods: Per-class overrides, e.g. {0: Method.PIXELATE}.
        gaussian_ksize: Kernel size for Gaussian blur (must be odd).
        pixelate_block: Block size for pixelation (lower = coarser).
        solid_color: BGR color tuple for solid fill.
        padding: Extra pixels to expand each detection box.
    """

    def __init__(
        self,
        method: Method = Method.GAUSSIAN,
        class_methods: dict[int, Method] | None = None,
        gaussian_ksize: tuple[int, int] = (51, 51),
        pixelate_block: int = 10,
        solid_color: tuple[int, int, int] = (0, 0, 0),
        padding: int = 0,
    ) -> None:
        # Validate parameters first
        validate_kernel_size(gaussian_ksize, "gaussian_ksize")
        validate_positive(pixelate_block, "pixelate_block")
        validate_color(solid_color, "solid_color")
        validate_non_negative(padding, "padding")

        self.method = method
        self.class_methods = class_methods or {}
        self.gaussian_ksize = gaussian_ksize
        self.pixelate_block = pixelate_block
        self.solid_color = solid_color
        self.padding = padding

    def apply(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """Anonymize all detected regions in-place and return the frame."""
        h, w = frame.shape[:2]

        for det in detections:
            method = self.class_methods.get(det.class_id, self.method)

            # Apply padding and clip to frame bounds
            x1 = max(0, det.x1 - self.padding)
            y1 = max(0, det.y1 - self.padding)
            x2 = min(w, det.x2 + self.padding)
            y2 = min(h, det.y2 + self.padding)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]

            if method == Method.GAUSSIAN:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, self.gaussian_ksize, 0)
            elif method == Method.PIXELATE:
                frame[y1:y2, x1:x2] = self._pixelate(roi)
            elif method == Method.SOLID:
                frame[y1:y2, x1:x2] = self.solid_color

        return frame

    def _pixelate(self, roi: np.ndarray) -> np.ndarray:
        """Down-scale then up-scale to create a pixelation effect."""
        h, w = roi.shape[:2]
        small_w = max(1, w // self.pixelate_block)
        small_h = max(1, h // self.pixelate_block)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
