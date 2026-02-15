"""Tests for enterprise features (audit logging, batch processing, monitoring, masking)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from privacyguard.enterprise import (
    AuditLogger,
    BatchProcessor,
    ComplianceWatermark,
    CustomRegionMasker,
    RealTimeMonitor,
)


class TestAuditLogger:
    """Test audit logging functionality."""

    def test_init_creates_parent_directory(self, tmp_path):
        """Test that AuditLogger creates parent directories."""
        nested_path = tmp_path / "subdir" / "audit.json"
        _logger = AuditLogger(nested_path)
        assert nested_path.parent.exists()

    def test_log_anonymization_appends_entry(self, tmp_path):
        """Test logging an anonymization operation."""
        log_file = tmp_path / "audit.json"
        logger = AuditLogger(log_file)

        logger.log_anonymization(
            source_file="input.jpg",
            output_file="output.jpg",
            detections_count=5,
            processing_time_ms=50.0,
            anonymization_method="gaussian",
            model_name="yolov8n-face.onnx",
        )

        assert len(logger.logs) == 1
        assert logger.logs[0].source_file == "input.jpg"
        assert logger.logs[0].detections_count == 5

    def test_save_logs_creates_json_file(self, tmp_path):
        """Test that logs are saved to JSON file."""
        log_file = tmp_path / "audit.json"
        logger = AuditLogger(log_file)

        logger.log_anonymization(
            source_file="input.jpg",
            output_file="output.jpg",
            detections_count=5,
            processing_time_ms=50.0,
            anonymization_method="gaussian",
            model_name="model.onnx",
        )

        assert log_file.exists()
        with open(log_file) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["source"] == "input.jpg"

    def test_load_logs_from_existing_file(self, tmp_path):
        """Test loading existing logs (previously broken feature)."""
        log_file = tmp_path / "audit.json"

        # Create initial logger and add a log
        logger1 = AuditLogger(log_file)
        logger1.log_anonymization(
            source_file="input1.jpg",
            output_file="output1.jpg",
            detections_count=10,
            processing_time_ms=33.0,
            anonymization_method="pixelate",
            model_name="model.onnx",
        )

        # Create new logger with same file - should load existing logs
        logger2 = AuditLogger(log_file)
        assert len(logger2.logs) == 1
        assert logger2.logs[0].source_file == "input1.jpg"
        assert logger2.logs[0].detections_count == 10

    def test_load_logs_handles_corrupt_file(self, tmp_path):
        """Test graceful handling of corrupt JSON."""
        log_file = tmp_path / "corrupt.json"
        log_file.write_text("{ invalid json }")

        # Should not raise, just start fresh
        logger = AuditLogger(log_file)
        assert len(logger.logs) == 0

    def test_get_compliance_report_empty(self, tmp_path):
        """Test compliance report with no logs."""
        log_file = tmp_path / "audit.json"
        logger = AuditLogger(log_file)

        report = logger.get_compliance_report()
        assert report == {}

    def test_get_compliance_report_with_data(self, tmp_path):
        """Test compliance report generation."""
        log_file = tmp_path / "audit.json"
        logger = AuditLogger(log_file)

        # Add multiple logs
        for i in range(3):
            logger.log_anonymization(
                source_file=f"input{i}.jpg",
                output_file=f"output{i}.jpg",
                detections_count=5 * (i + 1),
                processing_time_ms=30.0 + i,
                anonymization_method="gaussian" if i % 2 == 0 else "pixelate",
                model_name="model.onnx",
            )

        report = logger.get_compliance_report()
        assert report["total_operations"] == 3
        assert report["total_detections"] == 5 + 10 + 15
        assert "period_start" in report
        assert "period_end" in report
        assert len(report["methods_used"]) == 2  # gaussian and pixelate
        assert report["failed_operations"] == 0


class TestBatchProcessor:
    """Test batch processing functionality."""

    @patch("privacyguard.core.PrivacyGuard")
    def test_init_creates_output_directory(self, mock_guard_class, tmp_path, monkeypatch):
        """Test that BatchProcessor creates output directory."""
        output_dir = tmp_path / "output"
        monkeypatch.setattr("pathlib.Path.glob", lambda *args, **kwargs: [])

        mock_guard = MagicMock()
        mock_guard.detector.model_path = "test_model.onnx"
        mock_guard_class.return_value = mock_guard

        _processor = BatchProcessor("dummy.onnx", output_dir=output_dir)
        assert output_dir.exists()

    @patch("privacyguard.core.PrivacyGuard")
    def test_process_directory_no_files(self, mock_guard_class, tmp_path, monkeypatch):
        """Test processing empty directory."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Mock glob to return empty list
        monkeypatch.setattr("pathlib.Path.glob", lambda *args, **kwargs: [])

        mock_guard = MagicMock()
        mock_guard.detector.model_path = "test_model.onnx"
        mock_guard_class.return_value = mock_guard

        processor = BatchProcessor("dummy.onnx", output_dir=output_dir)
        results = processor.process_directory(str(input_dir))

        assert results["total_files"] == 0
        assert results["successful"] == 0
        assert results["failed"] == 0

    @patch("privacyguard.core.PrivacyGuard")
    def test_process_directory_with_images(self, mock_guard_class, tmp_path, monkeypatch):
        """Test processing directory with images."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create dummy image files
        img_files = [input_dir / f"img{i}.jpg" for i in range(3)]
        for img_file in img_files:
            img_file.write_bytes(b"dummy")

        # Mock PrivacyGuard
        mock_guard = MagicMock()
        mock_guard.detector.model_path = "test_model.onnx"
        mock_guard.process_image.return_value = [MagicMock(), MagicMock()]  # 2 detections
        mock_guard_class.return_value = mock_guard

        # Mock glob to return our image files
        def mock_glob(self, pattern):
            if "*.jpg" in pattern:
                return img_files
            return []

        monkeypatch.setattr("pathlib.Path.glob", mock_glob)

        processor = BatchProcessor("dummy.onnx", output_dir=output_dir)
        results = processor.process_directory(str(input_dir), pattern="*.jpg")

        assert results["total_files"] == 3
        assert results["successful"] == 3
        assert results["failed"] == 0
        assert results["total_detections"] == 6  # 2 detections x 3 files

    @patch("privacyguard.core.PrivacyGuard")
    def test_process_directory_uses_actual_model_name(
        self, mock_guard_class, tmp_path, monkeypatch
    ):
        """Test that model_name is extracted from detector (not hardcoded)."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        img_file = input_dir / "test.jpg"
        img_file.write_bytes(b"dummy")

        # Mock PrivacyGuard with actual model path
        mock_guard = MagicMock()
        mock_guard.detector.model_path = Path("my_model.onnx")
        mock_guard.process_image.return_value = []
        mock_guard_class.return_value = mock_guard

        # Mock audit logger to capture model name
        mock_audit = MagicMock()

        def mock_glob(self, pattern):
            return [img_file] if "*.jpg" in pattern else []

        monkeypatch.setattr("pathlib.Path.glob", mock_glob)

        processor = BatchProcessor("dummy.onnx", output_dir=output_dir, audit_logger=mock_audit)
        processor.process_directory(str(input_dir), pattern="*.jpg")

        # Verify that audit logger was called with actual model name, not "unknown"
        mock_audit.log_anonymization.assert_called_once()
        call_kwargs = mock_audit.log_anonymization.call_args[1]
        assert call_kwargs["model_name"] == "my_model.onnx"

    @patch("privacyguard.core.PrivacyGuard")
    def test_process_directory_with_audit_logger(
        self, mock_guard_class, tmp_path, monkeypatch
    ):
        """Test integration with AuditLogger."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        log_file = tmp_path / "audit.json"
        input_dir.mkdir()
        output_dir.mkdir()

        img_file = input_dir / "test.jpg"
        img_file.write_bytes(b"dummy")

        mock_guard = MagicMock()
        mock_guard.detector.model_path = "model.onnx"
        mock_guard.process_image.return_value = [MagicMock()]
        mock_guard_class.return_value = mock_guard

        def mock_glob(self, pattern):
            return [img_file] if "*.jpg" in pattern else []

        monkeypatch.setattr("pathlib.Path.glob", mock_glob)

        audit_logger = AuditLogger(log_file)
        processor = BatchProcessor("dummy.onnx", output_dir=output_dir, audit_logger=audit_logger)
        processor.process_directory(str(input_dir), pattern="*.jpg")

        # Verify audit logs were created
        assert len(audit_logger.logs) == 1
        assert log_file.exists()


class TestRealTimeMonitor:
    """Test real-time performance monitoring."""

    def test_init_empty_metrics(self):
        """Test monitor initializes with empty metrics."""
        monitor = RealTimeMonitor("test_camera")
        assert len(monitor.frame_times) == 0
        assert len(monitor.detection_counts) == 0

    def test_record_frame_appends_metrics(self):
        """Test recording frame metrics."""
        monitor = RealTimeMonitor()
        monitor.record_frame(processing_time_ms=33.0, detection_count=5)

        assert len(monitor.frame_times) == 1
        assert monitor.frame_times[0] == 33.0
        assert monitor.detection_counts[0] == 5

    def test_record_frame_limits_history(self):
        """Test that history is limited to 300 frames."""
        monitor = RealTimeMonitor()

        # Record 350 frames
        for _ in range(350):
            monitor.record_frame(processing_time_ms=30.0, detection_count=1)

        # Should only keep last 300
        assert len(monitor.frame_times) == 300
        assert len(monitor.detection_counts) == 300

    def test_get_stats_empty(self):
        """Test stats with no recorded frames."""
        monitor = RealTimeMonitor()
        stats = monitor.get_stats()
        assert stats == {}

    def test_get_stats_with_data(self):
        """Test FPS, P95, P99 calculations."""
        monitor = RealTimeMonitor()

        # Record 30 frames with 33ms each (≈30 FPS)
        for _ in range(30):
            monitor.record_frame(processing_time_ms=33.0, detection_count=5)

        stats = monitor.get_stats()
        assert "fps" in stats
        assert "avg_latency_ms" in stats
        assert "p95_latency_ms" in stats
        assert "p99_latency_ms" in stats
        assert "avg_detections" in stats
        # At 33ms per frame: 1000/33 ≈ 30 FPS
        assert 29 < stats["fps"] < 31

    def test_should_alert_below_threshold(self):
        """Test FPS alert triggering."""
        monitor = RealTimeMonitor()

        # Slow frames (100ms = 10 FPS)
        for _ in range(10):
            monitor.record_frame(processing_time_ms=100.0, detection_count=1)

        # Should alert when FPS drops below 15
        assert monitor.should_alert(fps_threshold=15.0)
        # Should not alert when threshold is low
        assert not monitor.should_alert(fps_threshold=5.0)


class TestCustomRegionMasker:
    """Test custom region masking."""

    def test_init_empty_regions(self):
        """Test masker initializes with no regions."""
        masker = CustomRegionMasker()
        assert len(masker.regions) == 0

    def test_add_region(self):
        """Test adding a region."""
        masker = CustomRegionMasker()
        masker.add_region("logo", x1=10, y1=10, x2=100, y2=50, method="solid")

        assert len(masker.regions) == 1
        assert masker.regions[0]["name"] == "logo"
        assert masker.regions[0]["method"] == "solid"

    def test_apply_masks_gaussian(self, tmp_path):
        """Test Gaussian blur masking."""
        # Create a frame with varying values so blur is visible
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:50, 10:50] = [255, 0, 0]  # Red region

        masker = CustomRegionMasker()
        masker.add_region("test", x1=10, y1=10, x2=50, y2=50, method="gaussian")
        result = masker.apply_masks(frame)

        # Verify result has same shape and is a numpy array
        assert result.shape == frame.shape
        assert isinstance(result, np.ndarray)

    def test_apply_masks_pixelate(self, tmp_path):
        """Test pixelate masking."""
        # Create a frame with varying values so pixelation is visible
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:50, 10:50] = [0, 255, 0]  # Green region

        masker = CustomRegionMasker()
        masker.add_region("test", x1=10, y1=10, x2=50, y2=50, method="pixelate")
        result = masker.apply_masks(frame)

        # Verify result has same shape and is a numpy array
        assert result.shape == frame.shape
        assert isinstance(result, np.ndarray)

    def test_apply_masks_solid(self):
        """Test solid black masking."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 200

        masker = CustomRegionMasker()
        masker.add_region("test", x1=10, y1=10, x2=50, y2=50, method="solid")
        result = masker.apply_masks(frame)

        # Verify the region is solid black (0, 0, 0)
        assert np.all(result[10:50, 10:50] == 0)

    def test_save_and_load_config(self, tmp_path):
        """Test saving and loading configuration."""
        config_file = tmp_path / "regions.json"

        masker1 = CustomRegionMasker()
        masker1.add_region("logo", x1=0, y1=0, x2=100, y2=100, method="solid")
        masker1.add_region("sign", x1=500, y1=200, x2=700, y2=400, method="gaussian")
        masker1.save_config(str(config_file))

        masker2 = CustomRegionMasker()
        masker2.load_config(str(config_file))

        assert len(masker2.regions) == 2
        assert masker2.regions[0]["name"] == "logo"
        assert masker2.regions[1]["name"] == "sign"


class TestComplianceWatermark:
    """Test compliance watermarking."""

    def test_add_compliance_badge_default(self):
        """Test adding default compliance badge."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

        result = ComplianceWatermark.add_compliance_badge(frame)

        # Verify output is a frame with same shape
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
        # Verify the badge area was modified (top-left corner should be greener)
        assert not np.array_equal(result[:80, :], frame[:80, :])

    def test_add_compliance_badge_custom_text(self):
        """Test custom badge text."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

        result = ComplianceWatermark.add_compliance_badge(frame, text="GDPR COMPLIANT")

        # Should have same shape
        assert result.shape == frame.shape
        # Should be different from original
        assert not np.array_equal(result, frame)

    def test_add_compliance_badge_custom_opacity(self):
        """Test custom opacity."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

        result_opaque = ComplianceWatermark.add_compliance_badge(frame, opacity=0.9)
        result_transparent = ComplianceWatermark.add_compliance_badge(frame, opacity=0.1)

        # Both should be different from original
        assert not np.array_equal(result_opaque, frame)
        assert not np.array_equal(result_transparent, frame)
        # Opaque version should differ more from original than transparent
        diff_opaque = np.sum(np.abs(result_opaque.astype(float) - frame.astype(float)))
        diff_transparent = np.sum(np.abs(result_transparent.astype(float) - frame.astype(float)))
        assert diff_opaque > diff_transparent

    def test_badge_includes_timestamp(self):
        """Test that timestamp is included in badge."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        result = ComplianceWatermark.add_compliance_badge(frame)

        # Verify that the badge area (especially lower part with timestamp) was modified
        # The timestamp is written in the lower part of the badge area
        assert not np.array_equal(result[50:80, :], frame[50:80, :])
