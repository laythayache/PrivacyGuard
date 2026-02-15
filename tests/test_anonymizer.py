"""Tests for the Anonymizer module."""

from __future__ import annotations

import numpy as np

from privacyguard.anonymizer import Anonymizer, Method
from privacyguard.detector import Detection


class TestAnonymizer:
    def test_gaussian_modifies_roi(self, sample_detections: list[Detection]):
        # Use a gradient frame so blur actually changes pixel values
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, 200:300] = np.arange(100).reshape(100, 1, 1)
        anon = Anonymizer(method=Method.GAUSSIAN)
        original = frame.copy()
        result = anon.apply(frame, sample_detections)

        roi_orig = original[100:200, 200:300]
        roi_result = result[100:200, 200:300]
        assert not np.array_equal(roi_orig, roi_result)

    def test_pixelate_modifies_roi(self, sample_detections: list[Detection]):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, 200:300] = np.arange(100).reshape(100, 1, 1)
        anon = Anonymizer(method=Method.PIXELATE, pixelate_block=5)
        original = frame.copy()
        result = anon.apply(frame, sample_detections)

        roi_orig = original[100:200, 200:300]
        roi_result = result[100:200, 200:300]
        assert not np.array_equal(roi_orig, roi_result)

    def test_solid_fills_roi(self, sample_frame: np.ndarray, sample_detections: list[Detection]):
        color = (128, 128, 128)
        anon = Anonymizer(method=Method.SOLID, solid_color=color)
        result = anon.apply(sample_frame, [sample_detections[0]])

        roi = result[100:200, 200:300]
        assert np.all(roi == color)

    def test_per_class_method(self, sample_frame: np.ndarray, sample_detections: list[Detection]):
        anon = Anonymizer(
            method=Method.GAUSSIAN,
            class_methods={1: Method.SOLID},
            solid_color=(0, 0, 0),
        )
        result = anon.apply(sample_frame, sample_detections)

        # Class 1 region should be solid black
        roi_plate = result[300:350, 400:550]
        assert np.all(roi_plate == 0)

    def test_empty_detections_unchanged(self, sample_frame: np.ndarray):
        anon = Anonymizer()
        original = sample_frame.copy()
        result = anon.apply(sample_frame, [])
        assert np.array_equal(original, result)

    def test_padding_expands_region(self, sample_frame: np.ndarray):
        det = Detection(x1=200, y1=100, x2=300, y2=200, confidence=0.9, class_id=0)
        anon = Anonymizer(method=Method.SOLID, solid_color=(255, 0, 0), padding=10)
        result = anon.apply(sample_frame, [det])

        # Pixel just outside the original box but inside padding should be modified
        assert np.array_equal(result[95, 195], [255, 0, 0])

    def test_zero_size_detection_skipped(self, sample_frame: np.ndarray):
        det = Detection(x1=100, y1=100, x2=100, y2=100, confidence=0.9, class_id=0)
        anon = Anonymizer()
        original = sample_frame.copy()
        result = anon.apply(sample_frame, [det])
        assert np.array_equal(original, result)

    def test_output_shape_preserved(
        self, sample_frame: np.ndarray, sample_detections: list[Detection],
    ):
        anon = Anonymizer()
        result = anon.apply(sample_frame, sample_detections)
        assert result.shape == sample_frame.shape
