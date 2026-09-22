"""Operational helpers for logging, batch processing, monitoring, and overlays."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class AuditLog:
    """Single audit log entry."""

    timestamp: str
    source_file: str
    output_file: str
    detections_count: int
    processing_time_ms: float
    anonymization_method: str
    model_name: str
    region_count: int = 0
    user: str = "system"
    status: str = "success"
    error: str | None = None


class AuditLogger:
    """Record masking operations in a local JSON log.

    The log is an operational record, not a tamper-evident audit system or
    evidence of legal compliance. Callers are responsible for access control,
    retention, redaction, integrity protection, and secure storage. File paths
    and user values may themselves contain sensitive information.
    """

    def __init__(self, log_path: str | Path) -> None:
        """Initialize audit logger.

        Args:
            log_path: Path to JSON audit log file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logs: list[AuditLog] = []

        # Load existing logs
        if self.log_path.exists():
            self._load_logs()

    def log_anonymization(
        self,
        source_file: str,
        output_file: str,
        detections_count: int,
        processing_time_ms: float,
        anonymization_method: str,
        model_name: str,
        user: str = "system",
    ) -> None:
        """Log an anonymization operation.

        Args:
            source_file: Input file path
            output_file: Output file path
            detections_count: Number of objects detected
            processing_time_ms: Processing time in milliseconds
            anonymization_method: Method used (blur, pixelate, etc.)
            model_name: Detection model name
            user: User who initiated processing
        """
        log = AuditLog(
            timestamp=datetime.utcnow().isoformat(),
            source_file=source_file,
            output_file=output_file,
            detections_count=detections_count,
            processing_time_ms=processing_time_ms,
            anonymization_method=anonymization_method,
            model_name=model_name,
            user=user,
        )
        self.logs.append(log)
        self._save_logs()

    def get_operation_summary(self) -> dict[str, Any]:
        """Summarize the operations currently loaded by this logger.

        Returns:
            Operational counts and timing statistics. These values do not
            establish that processing was complete, correct, or compliant.
        """
        if not self.logs:
            return {}

        timestamps = [datetime.fromisoformat(log.timestamp) for log in self.logs]
        return {
            "total_operations": len(self.logs),
            "period_start": min(timestamps).isoformat(),
            "period_end": max(timestamps).isoformat(),
            "total_detections": sum(log.detections_count for log in self.logs),
            "avg_processing_time_ms": np.mean(
                [log.processing_time_ms for log in self.logs]
            ),
            "methods_used": list(set(log.anonymization_method for log in self.logs)),
            "models_used": list(set(log.model_name for log in self.logs)),
            "failed_operations": sum(1 for log in self.logs if log.status != "success"),
        }

    def get_compliance_report(self) -> dict[str, Any]:
        """Return :meth:`get_operation_summary` for backward compatibility.

        The historical method name does not imply certification or a legal
        compliance determination. New code should use ``get_operation_summary``.
        """
        return self.get_operation_summary()

    def _save_logs(self) -> None:
        """Save logs to file."""
        with open(self.log_path, "w") as f:
            json.dump(
                [
                    {
                        "timestamp": log.timestamp,
                        "source": log.source_file,
                        "output": log.output_file,
                        "detections": log.detections_count,
                        "time_ms": log.processing_time_ms,
                        "method": log.anonymization_method,
                        "model": log.model_name,
                        "user": log.user,
                        "status": log.status,
                    }
                    for log in self.logs
                ],
                f,
                indent=2,
            )

    def _load_logs(self) -> None:
        """Load existing logs from file."""
        try:
            with open(self.log_path) as f:
                data = json.load(f)
                for entry in data:
                    log = AuditLog(
                        timestamp=entry["timestamp"],
                        source_file=entry["source"],
                        output_file=entry["output"],
                        detections_count=entry["detections"],
                        processing_time_ms=entry["time_ms"],
                        anonymization_method=entry["method"],
                        model_name=entry["model"],
                        user=entry.get("user", "system"),
                        status=entry.get("status", "success"),
                    )
                    self.logs.append(log)
        except (json.JSONDecodeError, OSError, KeyError):
            pass  # Start fresh if file corrupt/missing


class BatchProcessor:
    """Process matching files in a directory with progress tracking.

    Processing is sequential. Errors are counted in the returned summary and
    printed so callers can decide how to retry or escalate them.
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialize batch processor.

        Args:
            model_path: Path to detection model
            output_dir: Where to save anonymized files
            audit_logger: Optional audit logger
        """
        from .core import PrivacyGuard

        self.guard = PrivacyGuard(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audit_logger = audit_logger

    def process_directory(
        self, input_dir: str, pattern: str = "*.jpg", method: str = "gaussian"
    ) -> dict[str, Any]:
        """Process all images matching pattern in directory.

        Args:
            input_dir: Input directory path
            pattern: File pattern (*.jpg, *.png, etc.)
            method: Anonymization method

        Returns:
            Processing summary
        """
        input_path = Path(input_dir)
        files = list(input_path.glob(pattern))

        results = {
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "total_time_sec": 0.0,
            "total_detections": 0,
        }

        start_time = time.time()

        for i, file_path in enumerate(files):
            try:
                output_path = self.output_dir / file_path.name
                start = time.time()

                detections = self.guard.process_image(str(file_path), str(output_path))
                elapsed_ms = (time.time() - start) * 1000

                if self.audit_logger:
                    self.audit_logger.log_anonymization(
                        source_file=str(file_path),
                        output_file=str(output_path),
                        detections_count=len(detections),
                        processing_time_ms=elapsed_ms,
                        anonymization_method=method,
                        model_name=str(self.guard.detector.model_path),
                    )

                results["successful"] += 1
                results["total_detections"] += len(detections)

                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(files)} files")

            except Exception as e:
                results["failed"] += 1
                print(f"Error processing {file_path}: {e}")

        results["total_time_sec"] = time.time() - start_time
        return results


class RealTimeMonitor:
    """Track recent latency and detection-count metrics in memory."""

    def __init__(self, name: str = "monitor") -> None:
        """Initialize real-time monitor.

        Args:
            name: Monitor instance name
        """
        self.name = name
        self.frame_times: list[float] = []
        self.detection_counts: list[int] = []
        self.start_time = time.time()

    def record_frame(self, processing_time_ms: float, detection_count: int) -> None:
        """Record metrics for a frame.

        Args:
            processing_time_ms: Time to process frame
            detection_count: Number of detections
        """
        self.frame_times.append(processing_time_ms)
        self.detection_counts.append(detection_count)

        # Keep a bounded rolling window of recent frames.
        if len(self.frame_times) > 300:
            self.frame_times.pop(0)
            self.detection_counts.pop(0)

    def get_stats(self) -> dict[str, float]:
        """Get current performance statistics.

        Returns:
            Dict with FPS, avg latency, etc.
        """
        if not self.frame_times:
            return {}

        mean_latency = float(np.mean(self.frame_times))
        fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0
        return {
            "fps": float(fps),
            "avg_latency_ms": mean_latency,
            "p95_latency_ms": float(np.percentile(self.frame_times, 95)),
            "p99_latency_ms": float(np.percentile(self.frame_times, 99)),
            "avg_detections": float(np.mean(self.detection_counts)),
            "max_detections": int(np.max(self.detection_counts)),
        }

    def should_alert(self, fps_threshold: float = 15.0) -> bool:
        """Check if performance is degraded and alert needed.

        Args:
            fps_threshold: Alert if FPS drops below this

        Returns:
            True if alert needed
        """
        stats = self.get_stats()
        return stats.get("fps", 0) < fps_threshold


class CustomRegionMasker:
    """Define rectangular regions that should always be masked.

    Features:
    - Define rectangular zones
    - Different strategies per zone
    - Persistent configuration

    Use case: Always blur company logo, private areas, etc.
    """

    def __init__(self) -> None:
        """Initialize custom region masker."""
        self.regions: list[dict[str, Any]] = []

    def add_region(
        self, name: str, x1: int, y1: int, x2: int, y2: int, method: str = "gaussian"
    ) -> None:
        """Add a region to mask.

        Args:
            name: Region name
            x1, y1, x2, y2: Bounding box
            method: Anonymization method
        """
        self.regions.append(
            {"name": name, "bbox": [x1, y1, x2, y2], "method": method}
        )

    def apply_masks(self, frame: np.ndarray) -> np.ndarray:
        """Apply all custom masks to frame.

        Args:
            frame: Input frame

        Returns:
            Frame with custom regions masked
        """
        result: np.ndarray = np.array(frame, copy=True)

        for region in self.regions:
            x1, y1, x2, y2 = region["bbox"]
            method = region["method"]

            if x2 <= x1 or y2 <= y1:
                continue

            roi = result[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            masked: np.ndarray
            if method == "gaussian":
                masked = np.asarray(cv2.GaussianBlur(roi, (31, 31), 0), dtype=np.uint8)
            elif method == "pixelate":
                # Pixelate: resize down then up
                h, w = roi.shape[:2]
                down_w = max(1, w // 16)
                down_h = max(1, h // 16)
                temp = cv2.resize(roi, (down_w, down_h))
                masked = np.asarray(
                    cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST),
                    dtype=np.uint8,
                )
            else:
                masked = np.zeros_like(roi)

            result[y1:y2, x1:x2] = masked

        return result

    def save_config(self, path: str) -> None:
        """Save region configuration to file.

        Args:
            path: File path to save
        """
        with open(path, "w") as f:
            json.dump(self.regions, f, indent=2)

    def load_config(self, path: str) -> None:
        """Load region configuration from file.

        Args:
            path: File path to load
        """
        with open(path) as f:
            self.regions = json.load(f)


class ProcessingStatusWatermark:
    """Add a visible status label and timestamp to a frame.

    The overlay is informational only. It does not prove that masking was
    successful, that consent existed, or that processing complied with law.
    It is not cryptographically signed or tamper-evident.
    """

    @staticmethod
    def add_status_label(
        frame: np.ndarray, text: str = "MASKING APPLIED", opacity: float = 0.3
    ) -> np.ndarray:
        """Add a visible processing-status label to a copy of ``frame``.

        Args:
            frame: Input frame
            text: Informational label text
            opacity: Badge opacity (0-1)

        Returns:
            Frame with the status overlay

        Raises:
            ValueError: If opacity is outside the inclusive range 0 to 1
        """
        if not 0.0 <= opacity <= 1.0:
            raise ValueError("opacity must be between 0 and 1")

        result: np.ndarray = np.array(frame, copy=True)
        h, w = frame.shape[:2]

        # Add semi-transparent overlay
        overlay = result.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 80), (0, 255, 0), -1)

        blended = cv2.addWeighted(overlay, opacity, result, 1 - opacity, 0)
        result = np.asarray(blended, dtype=np.uint8)

        # Add text
        cv2.putText(
            result,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            result,
            datetime.utcnow().isoformat(),
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return result

    @staticmethod
    def add_compliance_badge(
        frame: np.ndarray, text: str = "MASKING APPLIED", opacity: float = 0.3
    ) -> np.ndarray:
        """Call :meth:`add_status_label` for backward compatibility.

        The historical method name has no legal or evidentiary meaning. New
        code should use ``add_status_label``.
        """
        return ProcessingStatusWatermark.add_status_label(frame, text, opacity)


class ComplianceWatermark(ProcessingStatusWatermark):
    """Backward-compatible name for :class:`ProcessingStatusWatermark`.

    This class does not assert, prove, or certify legal compliance.
    """
