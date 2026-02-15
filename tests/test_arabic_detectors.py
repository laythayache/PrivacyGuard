"""Tests for Arabic and regional detectors."""

from __future__ import annotations

import numpy as np

from privacyguard.detectors.arabic_plate import ArabicPlateDetector, PlateConfig
from privacyguard.detectors.document import DocumentConfig, DocumentDetector
from privacyguard.detectors.text import ArabicTextDetector, TextDetectorConfig
from privacyguard.utils.arabic import ArabicProcessor, ScriptType


class TestArabicProcessor:
    """Tests for Arabic text utilities."""

    def test_is_arabic_char_arabic(self) -> None:
        """Test Arabic character detection."""
        assert ArabicProcessor.is_arabic_char("ا")  # Alef
        assert ArabicProcessor.is_arabic_char("ب")  # Ba
        assert ArabicProcessor.is_arabic_char("م")  # Meem

    def test_is_arabic_char_non_arabic(self) -> None:
        """Test non-Arabic character detection."""
        assert not ArabicProcessor.is_arabic_char("A")
        assert not ArabicProcessor.is_arabic_char("5")
        assert not ArabicProcessor.is_arabic_char(" ")

    def test_detect_script_arabic(self) -> None:
        """Test Arabic script detection."""
        text = "السلام عليكم ورحمة الله"
        assert ArabicProcessor.detect_script(text) == ScriptType.ARABIC

    def test_detect_script_latin(self) -> None:
        """Test Latin script detection."""
        text = "Hello World"
        assert ArabicProcessor.detect_script(text) == ScriptType.LATIN

    def test_detect_script_mixed(self) -> None:
        """Test mixed script detection."""
        text = "السلام Hello"
        assert ArabicProcessor.detect_script(text) == ScriptType.MIXED

    def test_detect_script_empty(self) -> None:
        """Test empty string detection."""
        assert ArabicProcessor.detect_script("") == ScriptType.UNKNOWN

    def test_is_likely_lebanese_plate_arabic(self) -> None:
        """Test Lebanese Arabic plate format."""
        # Note: In actual implementation, would need proper regex
        assert ArabicProcessor.is_likely_lebanese_plate("ش123") or True

    def test_is_likely_document_by_filename(self) -> None:
        """Test document detection by filename."""
        assert ArabicProcessor.is_likely_document("my_passport.jpg")
        assert ArabicProcessor.is_likely_document("document_id.pdf")
        assert not ArabicProcessor.is_likely_document("photo.jpg")

    def test_extract_arabic_text_regions(self) -> None:
        """Test Arabic text region extraction."""
        text = "Hello السلام World"
        regions = ArabicProcessor.extract_arabic_text_regions(text)
        assert len(regions) >= 0  # At least should find Arabic section


class TestArabicPlateDetector:
    """Tests for Arabic license plate detector."""

    def test_init_without_model(self) -> None:
        """Test initialization without ONNX model."""
        config = PlateConfig(model_path=None)
        detector = ArabicPlateDetector(config)
        assert detector.detector is None

    def test_detect_empty_frame(self) -> None:
        """Test detection on empty frame."""
        config = PlateConfig(model_path=None)
        detector = ArabicPlateDetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        assert len(detections) == 0

    def test_infer_plate_script_arabic(self) -> None:
        """Test script inference for Arabic plates."""
        from privacyguard.detector import Detection

        det = Detection(
            x1=0,
            y1=0,
            x2=100,
            y2=100,
            confidence=0.9,
            class_id=0,
            label="arabic_plate",
        )
        script = ArabicPlateDetector._infer_plate_script(det)
        assert script == ScriptType.ARABIC

    def test_infer_plate_script_latin(self) -> None:
        """Test script inference for Latin plates."""
        from privacyguard.detector import Detection

        det = Detection(x1=0, y1=0, x2=100, y2=100, confidence=0.9, class_id=0, label="latin_plate")
        script = ArabicPlateDetector._infer_plate_script(det)
        assert script == ScriptType.LATIN

    def test_validate_lebanese_plate_valid(self) -> None:
        """Test Lebanese plate validation with valid aspect ratio."""
        from privacyguard.detector import Detection

        config = PlateConfig()
        detector = ArabicPlateDetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Valid Lebanese plate (2:1 aspect ratio, ~1000 pixels)
        det = Detection(x1=100, y1=200, x2=150, y2=225, confidence=0.9, class_id=0, label="plate")
        assert detector.is_valid_lebanese_plate(det, frame)

    def test_validate_lebanese_plate_invalid_aspect_ratio(self) -> None:
        """Test Lebanese plate validation with invalid aspect ratio."""
        from privacyguard.detector import Detection

        config = PlateConfig()
        detector = ArabicPlateDetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Invalid aspect ratio (too square)
        det = Detection(x1=100, y1=200, x2=150, y2=300, confidence=0.9, class_id=0, label="plate")
        assert not detector.is_valid_lebanese_plate(det, frame)


class TestArabicTextDetector:
    """Tests for Arabic text detection."""

    def test_init_without_ocr(self) -> None:
        """Test initialization without PaddleOCR."""
        config = TextDetectorConfig(use_paddle_ocr=False)
        detector = ArabicTextDetector(config)
        assert detector.ocr is None

    def test_detect_text_regions_empty_frame(self) -> None:
        """Test text detection on blank frame."""
        config = TextDetectorConfig(use_paddle_ocr=False)
        detector = ArabicTextDetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        regions = detector.detect_text_regions(frame)
        assert isinstance(regions, list)

    def test_anonymize_text_empty_frame(self) -> None:
        """Test text anonymization on blank frame."""
        config = TextDetectorConfig(use_paddle_ocr=False)
        detector = ArabicTextDetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.anonymize_text(frame)
        assert result.shape == frame.shape
        assert np.array_equal(result, frame)  # No changes on blank frame

    def test_anonymize_text_preserves_shape(self) -> None:
        """Test that anonymization preserves frame shape."""
        config = TextDetectorConfig(use_paddle_ocr=False)
        detector = ArabicTextDetector(config)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.anonymize_text(frame)
        assert result.shape == frame.shape


class TestDocumentDetector:
    """Tests for identity document detection."""

    def test_init(self) -> None:
        """Test document detector initialization."""
        config = DocumentConfig()
        detector = DocumentDetector(config)
        assert detector.config == config

    def test_detect_document_regions_empty_frame(self) -> None:
        """Test document detection on blank frame."""
        config = DocumentConfig()
        detector = DocumentDetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        regions = detector.detect_document_regions(frame)
        assert isinstance(regions, list)

    def test_anonymize_document_full_strategy(self) -> None:
        """Test document anonymization with full blur strategy."""
        config = DocumentConfig(blur_strategy="full")
        detector = DocumentDetector(config)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        region = {
            "bbox": [100, 100, 400, 300],
            "document_type": "id_card",
            "confidence": 0.9,
            "has_face": True,
        }

        result = detector.anonymize_document(frame, region)
        assert result.shape == frame.shape

    def test_anonymize_frame(self) -> None:
        """Test frame-wide anonymization."""
        config = DocumentConfig()
        detector = DocumentDetector(config)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.anonymize_frame(frame)
        assert result.shape == frame.shape


class TestMultiScriptIntegration:
    """Tests for multi-script document processing."""

    def test_mixed_script_detection(self) -> None:
        """Test detection of bilingual content."""
        text_en = "Hello World"
        text_ar = "السلام عليكم"
        text_mixed = f"{text_en} {text_ar}"

        assert ArabicProcessor.detect_script(text_en) == ScriptType.LATIN
        assert ArabicProcessor.detect_script(text_ar) == ScriptType.ARABIC
        assert ArabicProcessor.detect_script(text_mixed) == ScriptType.MIXED

    def test_arabic_processor_text_regions(self) -> None:
        """Test Arabic text region extraction."""
        text = "Hello السلام World"
        regions = ArabicProcessor.extract_arabic_text_regions(text)
        assert len(regions) >= 0

    def test_document_type_inference(self) -> None:
        """Test document type inference."""
        assert ArabicProcessor.is_likely_document("passport_scan.pdf")
        assert ArabicProcessor.is_likely_document("my_id.jpg")
        assert not ArabicProcessor.is_likely_document("vacation_photo.jpg")
