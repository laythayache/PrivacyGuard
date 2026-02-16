"""Tests for the PrivacyGuard orchestrator.

Uses monkeypatching to avoid loading a real ONNX model.
"""

from __future__ import annotations

import numpy as np
import pytest

from privacyguard.core import PrivacyGuard
from privacyguard.detector import Detection


@pytest.fixture
def guard(monkeypatch, tmp_path):
    """Create a PrivacyGuard with a mocked detector."""
    # Create a dummy model file so the FileNotFoundError check passes
    model_file = tmp_path / "dummy.onnx"
    model_file.write_bytes(b"")

    # Mock the ONNXDetector.__init__ to skip real ONNX loading
    from privacyguard import detector

    def mock_init(self, **kwargs):
        self.model_path = model_file
        self.input_size = (640, 640)
        self.conf_threshold = 0.4
        self.iou_threshold = 0.45
        self.class_labels = {0: "face", 1: "license_plate"}
        self.session = None

    monkeypatch.setattr(detector.ONNXDetector, "__init__", mock_init)

    # Mock detect to return predictable results
    fake_dets = [
        Detection(x1=100, y1=100, x2=200, y2=200, confidence=0.9, class_id=0, label="face"),
    ]
    monkeypatch.setattr(detector.ONNXDetector, "detect", lambda self, frame: fake_dets)

    return PrivacyGuard(model_path=str(model_file))


class TestProcessFrame:
    def test_returns_same_shape(self, guard: PrivacyGuard):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = guard.process_frame(frame)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype

    def test_does_not_mutate_input(self, guard: PrivacyGuard):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        guard.process_frame(frame)
        assert np.array_equal(frame, original)

    def test_modifies_detected_region(self, guard: PrivacyGuard):
        # Use a gradient frame so Gaussian blur produces visible change
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[100:200, 100:200] = np.arange(100).reshape(100, 1, 1)
        result2 = guard.process_frame(frame2)
        roi_orig = frame2[100:200, 100:200].copy()
        roi_result = result2[100:200, 100:200]
        assert not np.array_equal(roi_orig, roi_result)

    def test_records_nonzero_monitor_latency(self, guard: PrivacyGuard):
        class DummyMonitor:
            def __init__(self):
                self.calls: list[tuple[float, int]] = []

            def record_frame(self, processing_time_ms: float, detection_count: int) -> None:
                self.calls.append((processing_time_ms, detection_count))

        monitor = DummyMonitor()
        guard.monitor = monitor
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        _ = guard.process_frame(frame)

        assert len(monitor.calls) == 1
        latency_ms, count = monitor.calls[0]
        assert latency_ms > 0
        assert count == 1


class TestTargetClasses:
    def test_filters_by_target_class(self, guard: PrivacyGuard):
        # Set target to only class 1 (license_plate)
        guard.target_classes = {1}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = guard.detect(frame)
        # The mock detector returns class_id=0, so it should be filtered out
        assert len(dets) == 0

    def test_allows_matching_class(self, guard: PrivacyGuard):
        guard.target_classes = {0}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = guard.detect(frame)
        assert len(dets) == 1
        assert dets[0].class_id == 0


class TestProcessVideo:
    def test_audit_logger_uses_frame_metrics(self, guard: PrivacyGuard, monkeypatch):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        class DummyCapture:
            def __init__(self, *_):
                self._read_count = 0

            def isOpened(self):  # noqa: N802
                return True

            def get(self, prop):
                import cv2

                if prop == cv2.CAP_PROP_FPS:
                    return 30.0
                if prop == cv2.CAP_PROP_FRAME_WIDTH:
                    return 160
                if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                    return 120
                return 0

            def read(self):
                if self._read_count == 0:
                    self._read_count += 1
                    return True, frame.copy()
                return False, None

            def release(self):
                pass

        class DummyWriter:
            def __init__(self, *_):
                self.frames = 0

            def write(self, _):
                self.frames += 1

            def release(self):
                pass

        class DummyAudit:
            def __init__(self):
                self.calls = []

            def log_anonymization(self, **kwargs):
                self.calls.append(kwargs)

        monkeypatch.setattr("privacyguard.core.cv2.VideoCapture", DummyCapture)
        monkeypatch.setattr("privacyguard.core.cv2.VideoWriter", DummyWriter)
        monkeypatch.setattr(
            "privacyguard.core.cv2.VideoWriter_fourcc",
            lambda *args, **kwargs: 0,
        )

        audit = DummyAudit()
        guard.audit_logger = audit
        guard.process_video("input.mp4", "output.mp4", show=False)

        assert len(audit.calls) == 1
        assert audit.calls[0]["detections_count"] == 1
        assert audit.calls[0]["processing_time_ms"] > 0
