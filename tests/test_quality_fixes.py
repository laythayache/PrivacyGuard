"""Regression tests for quality and reliability fixes."""

from __future__ import annotations

import math

import numpy as np

from privacyguard.detector import Detection
from privacyguard.detectors.text import ArabicTextDetector, TextDetectorConfig
from privacyguard.ensemble import EnsembleConfig, EnsembleDetector
from privacyguard.metadata import MetadataStripper
from privacyguard.profiler import Profiler
from privacyguard.utils.arabic import ScriptType


def _make_ensemble_stub(mode: str = "union") -> EnsembleDetector:
    detector = object.__new__(EnsembleDetector)
    detector.config = EnsembleConfig(ensemble_mode=mode)
    detector.detectors = {}
    detector.regional_detectors = {}
    detector.model_weights = {"face": 1.0, "plate": 1.0, "person": 1.0, "a": 1.0, "b": 1.0}
    return detector


class TestMetadataFixes:
    def test_safe_filename_rewrites_sensitive_patterns(self):
        assert MetadataStripper.get_safe_filename("IMG_1234.jpg") == "image.jpg"
        assert MetadataStripper.get_safe_filename("AB-1234.png") == "anonymized.png"
        assert (
            MetadataStripper.get_safe_filename("capture_20240101_120101.jpeg")
            == "capture_anonymized.jpeg"
        )


class TestProfilerFixes:
    def test_stop_without_frames_returns_stable_metrics(self):
        profiler = Profiler()
        profiler.start()
        report = profiler.stop()

        assert report.total_frames == 0
        assert report.mean_confidence == 0.0
        assert report.memory_peak_mb >= 0.0
        assert not math.isnan(report.mean_confidence)


class TestEnsembleFixes:
    def test_intersection_mode_returns_overlapping_detections(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        ens = _make_ensemble_stub(mode="intersection")

        class DummyDetector:
            def __init__(self, dets):
                self._dets = dets

            def detect(self, _frame):
                return self._dets

        ens.detectors = {
            "a": DummyDetector([Detection(10, 10, 30, 30, 0.9, 0, "face")]),
            "b": DummyDetector([Detection(12, 12, 32, 32, 0.85, 0, "face")]),
        }

        results = ens.detect(frame)
        assert len(results) == 1
        assert results[0].confidence > 0

    def test_regional_text_detector_contributes_detections(self, monkeypatch):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        ens = _make_ensemble_stub(mode="union")

        text_detector = ArabicTextDetector(TextDetectorConfig(use_paddle_ocr=False))
        monkeypatch.setattr(
            text_detector,
            "detect_text_regions",
            lambda _frame: [
                {
                    "bbox": [5, 5, 20, 20],
                    "confidence": 0.9,
                    "script": ScriptType.ARABIC,
                    "text": "اختبار",
                }
            ],
        )
        ens.regional_detectors = {"text": text_detector}

        results = ens.detect(frame)
        assert len(results) == 1
        assert results[0].class_id == 100
        assert "arabic_text" in (results[0].label or "")
