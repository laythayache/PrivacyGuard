"""Tests for the ONNXDetector module.

These tests use numpy arrays to mock ONNX model outputs, avoiding
a real model dependency in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from privacyguard.detector import Detection, ONNXDetector


class TestDetection:
    def test_fields(self):
        det = Detection(x1=10, y1=20, x2=30, y2=40, confidence=0.9, class_id=0, label="face")
        assert det.x1 == 10
        assert det.label == "face"
        assert det.confidence == 0.9

    def test_immutable(self):
        det = Detection(x1=0, y1=0, x2=1, y2=1, confidence=0.5, class_id=0)
        with pytest.raises(AttributeError):
            det.x1 = 999  # type: ignore[misc]


class TestONNXDetectorInit:
    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            ONNXDetector(model_path=tmp_path / "nonexistent.onnx")


class TestPostprocessYOLOv8:
    """Test the YOLOv8 postprocess path with synthetic arrays."""

    def _make_detector_stub(self) -> ONNXDetector:
        """Create an ONNXDetector without loading a real model."""
        det = object.__new__(ONNXDetector)
        det.conf_threshold = 0.4
        det.iou_threshold = 0.45
        det.class_labels = {0: "face", 1: "license_plate"}
        det.input_size = (640, 640)
        return det

    def test_yolov8_single_detection(self):
        det = self._make_detector_stub()

        # Simulate YOLOv8 output: [1, 6, N] where 6 = 4 (xywh) + 2 classes
        # One strong detection at center of 640x640 input
        n = 1
        out = np.zeros((1, 6, n), dtype=np.float32)
        out[0, 0, 0] = 320  # cx
        out[0, 1, 0] = 240  # cy
        out[0, 2, 0] = 100  # w
        out[0, 3, 0] = 100  # h
        out[0, 4, 0] = 0.95  # class 0 score (face)
        out[0, 5, 0] = 0.10  # class 1 score (plate)

        scale = (1.0, 1.0)
        orig_shape = (640, 640, 3)
        results = det._postprocess_yolov8(out, scale, orig_shape)

        assert len(results) == 1
        assert results[0].class_id == 0
        assert results[0].label == "face"
        assert results[0].confidence == pytest.approx(0.95, abs=0.01)

    def test_yolov8_filters_low_confidence(self):
        det = self._make_detector_stub()

        n = 2
        out = np.zeros((1, 6, n), dtype=np.float32)
        # Detection 0: high confidence
        out[0, :4, 0] = [320, 240, 100, 100]
        out[0, 4, 0] = 0.9
        # Detection 1: below threshold
        out[0, :4, 1] = [100, 100, 50, 50]
        out[0, 4, 1] = 0.1

        results = det._postprocess_yolov8(out, (1.0, 1.0), (640, 640, 3))
        assert len(results) == 1

    def test_yolov8_empty_output(self):
        det = self._make_detector_stub()
        out = np.zeros((1, 6, 0), dtype=np.float32)
        results = det._postprocess_yolov8(out, (1.0, 1.0), (640, 640, 3))
        assert results == []

    def test_yolov8_scaling(self):
        det = self._make_detector_stub()

        out = np.zeros((1, 6, 1), dtype=np.float32)
        out[0, 0, 0] = 320  # cx
        out[0, 1, 0] = 320  # cy
        out[0, 2, 0] = 100  # w
        out[0, 3, 0] = 100  # h
        out[0, 4, 0] = 0.9

        # Original frame is 2x the input size
        scale = (2.0, 2.0)
        orig_shape = (1280, 1280, 3)
        results = det._postprocess_yolov8(out, scale, orig_shape)

        assert len(results) == 1
        r = results[0]
        # Center at (320,320) with w/h=100 -> corners (270,270)-(370,370)
        # Scaled 2x -> (540,540)-(740,740)
        assert r.x1 == 540
        assert r.y1 == 540
        assert r.x2 == 740
        assert r.y2 == 740


class TestPostprocessLegacy:
    """Test the legacy [N, 6] postprocess path."""

    def _make_detector_stub(self) -> ONNXDetector:
        det = object.__new__(ONNXDetector)
        det.conf_threshold = 0.4
        det.iou_threshold = 0.45
        det.class_labels = {0: "face"}
        det.input_size = (640, 640)
        return det

    def test_legacy_single_detection(self):
        det = self._make_detector_stub()

        # [N, 6] = (x1, y1, x2, y2, conf, class)
        out = np.array([[100, 100, 200, 200, 0.85, 0]], dtype=np.float32)
        results = det._postprocess_legacy(out, (1.0, 1.0), (640, 640, 3))

        assert len(results) == 1
        assert results[0].confidence == pytest.approx(0.85, abs=0.01)

    def test_legacy_filters_low_confidence(self):
        det = self._make_detector_stub()
        out = np.array([[100, 100, 200, 200, 0.1, 0]], dtype=np.float32)
        results = det._postprocess_legacy(out, (1.0, 1.0), (640, 640, 3))
        assert results == []


class TestPreprocess:
    def _make_detector_stub(self) -> ONNXDetector:
        det = object.__new__(ONNXDetector)
        det.input_size = (640, 640)
        return det

    def test_output_shape(self):
        det = self._make_detector_stub()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor, scale = det.preprocess(frame)

        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert 0.0 <= tensor.max() <= 1.0

    def test_scale_ratios(self):
        det = self._make_detector_stub()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, (sx, sy) = det.preprocess(frame)

        assert sx == pytest.approx(640 / 640)
        assert sy == pytest.approx(480 / 640)
