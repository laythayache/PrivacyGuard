"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from privacyguard.detector import Detection


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A 480x640 BGR test frame with a white rectangle (simulated face)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a white rectangle to simulate a bright region
    frame[100:200, 200:300] = 255
    return frame


@pytest.fixture
def sample_detections() -> list[Detection]:
    """A pair of mock detections covering known regions."""
    return [
        Detection(
            x1=200, y1=100, x2=300, y2=200, confidence=0.95, class_id=0, label="face",
        ),
        Detection(
            x1=400, y1=300, x2=550, y2=350, confidence=0.80, class_id=1, label="license_plate",
        ),
    ]
