"""Main orchestrator that ties detection, anonymization, and streaming together.

This is the primary public API surface. Most users only need this module.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

from .anonymizer import Anonymizer
from .detector import Detection, ONNXDetector
from .stream import VideoStream
from .utils.validation import validate_non_negative, validate_threshold

if TYPE_CHECKING:
    pass


class MonitorProtocol(Protocol):
    """Protocol for monitoring/profiling objects."""

    def record_frame(self, processing_time_ms: float, detection_count: int) -> None:
        """Record frame processing metrics."""
        ...


class AuditLoggerProtocol(Protocol):
    """Protocol for audit logging objects."""

    def log_anonymization(
        self,
        source_file: str,
        output_file: str,
        detections_count: int,
        processing_time_ms: float,
        anonymization_method: str,
        model_name: str,
    ) -> None:
        """Log an anonymization operation."""
        ...


class BatchProcessorProtocol(Protocol):
    """Protocol for batch processing objects."""

    def process_directory(self, source: str) -> None:
        """Process a directory of images/videos."""
        ...


class PrivacyGuard:
    """End-to-end privacy de-identification pipeline.

    Detects sensitive regions (faces, license plates, etc.) using an
    ONNX model and anonymizes them in real time.

    Args:
        model_path: Path to the ONNX detection model.
        method: Default anonymization method ("gaussian", "pixelate", "solid").
        conf_threshold: Minimum detection confidence (0-1).
        iou_threshold: NMS IoU threshold.
        input_size: Model input resolution (width, height).
        class_labels: Mapping of class_id -> human-readable label.
        target_classes: If set, only anonymize these class IDs.
        providers: ONNX Runtime execution providers.
        padding: Extra pixels around each detection box.

    Example::

        guard = PrivacyGuard("yolov8n-face.onnx")
        guard.run(source=0)  # webcam
    """

    def __init__(
        self,
        model_path: str | Path,
        method: str = "gaussian",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
        class_labels: dict[int, str] | None = None,
        target_classes: Sequence[int] | None = None,
        providers: Sequence[str] | None = None,
        padding: int = 0,
        class_methods: dict[int, str] | None = None,
        postprocess_hook: Callable[[np.ndarray, list[Detection]], np.ndarray] | None = None,
        audit_logger: AuditLoggerProtocol | None = None,
        batch_processor: BatchProcessorProtocol | None = None,
        monitor: MonitorProtocol | None = None,
    ) -> None:
        # Validate parameters (detector and anonymizer will also validate)
        validate_threshold(conf_threshold, "conf_threshold")
        validate_threshold(iou_threshold, "iou_threshold")
        validate_non_negative(padding, "padding")

        self.detector = ONNXDetector(
            model_path=model_path,
            input_size=input_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            providers=providers,
            class_labels=class_labels,
        )
        # Per-class anonymization
        from .anonymizer import Method
        cm = {k: Method(v) for k, v in (class_methods or {}).items()}
        self.anonymizer = Anonymizer(method=Method(method), class_methods=cm, padding=padding)
        self.target_classes = set(target_classes) if target_classes else None
        self.postprocess_hook = postprocess_hook
        self.audit_logger = audit_logger
        self.batch_processor = batch_processor
        self.monitor = monitor
        self._frame_count = 0
        self._fps = 0.0

    @property
    def fps(self) -> float:
        """Measured processing FPS (detection + anonymization)."""
        return self._fps

    # ------------------------------------------------------------------
    # Single-frame API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detect and anonymize a single BGR frame (returns a copy)."""
        result, detections, elapsed_ms = self._process_frame_with_metrics(frame)
        # Optional post-processing hook
        if self.postprocess_hook is not None:
            result = self.postprocess_hook(result, detections)
        # Optional real-time monitor
        if self.monitor:
            self.monitor.record_frame(
                processing_time_ms=elapsed_ms,
                detection_count=len(detections),
            )
        return result

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection only (no anonymization)."""
        dets = self.detector.detect(frame)
        if self.target_classes is not None:
            dets = [d for d in dets if d.class_id in self.target_classes]
        return dets

    def anonymize(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Apply anonymization given pre-computed detections."""
        return self.anonymizer.apply(frame, detections)

    # ------------------------------------------------------------------
    # Image file helpers
    # ------------------------------------------------------------------

    def process_image(self, input_path: str | Path, output_path: str | Path) -> list[Detection]:
        """Read an image, anonymize it, and write the result."""
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {input_path}")
        result, detections, elapsed_ms = self._process_frame_with_metrics(frame)
        if self.postprocess_hook is not None:
            result = self.postprocess_hook(result, detections)
        cv2.imwrite(str(output_path), result)
        # Optional audit logging
        if self.audit_logger:
            self.audit_logger.log_anonymization(
                source_file=str(input_path),
                output_file=str(output_path),
                detections_count=len(detections),
                processing_time_ms=elapsed_ms,
                anonymization_method=self.anonymizer.method.value,
                model_name=str(self.detector.model_path),
            )
        return detections

    # ------------------------------------------------------------------
    # Video file processing
    # ------------------------------------------------------------------

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        codec: str = "mp4v",
        show: bool = False,
    ) -> None:
        """Process an entire video file and write anonymized output."""
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result, detections, elapsed_ms = self._process_frame_with_metrics(frame)
                if self.postprocess_hook is not None:
                    result = self.postprocess_hook(result, detections)
                if self.monitor:
                    self.monitor.record_frame(
                        processing_time_ms=elapsed_ms,
                        detection_count=len(detections),
                    )
                writer.write(result)
                if self.audit_logger:
                    self.audit_logger.log_anonymization(
                        source_file=str(input_path),
                        output_file=str(output_path),
                        detections_count=len(detections),
                        processing_time_ms=elapsed_ms,
                        anonymization_method=self.anonymizer.method.value,
                        model_name=str(self.detector.model_path),
                    )
                if show:
                    cv2.imshow("PrivacyGuard", result)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            writer.release()
            if show:
                cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Live stream (real-time)
    # ------------------------------------------------------------------

    def run(
        self,
        source: int | str = 0,
        display: bool = True,
        output_path: str | Path | None = None,
        codec: str = "mp4v",
    ) -> None:
        """Run the de-identification pipeline on a live video source.

        Args:
            source: Camera index, video file path, or RTSP URL.
            display: Whether to show a live preview window.
            output_path: If set, write the anonymized stream to this file.
            codec: FourCC codec for the output file.
        """
        writer: cv2.VideoWriter | None = None

        with VideoStream(source) as stream:
            if output_path is not None:
                w, h = stream.frame_size
                fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
                writer = cv2.VideoWriter(str(output_path), fourcc, stream.fps, (w, h))

            try:
                self._loop(stream, display, writer)
            finally:
                if writer is not None:
                    writer.release()
                if display:
                    cv2.destroyAllWindows()
        # Optional batch processing
        if self.batch_processor:
            self.batch_processor.process_directory(str(source))

    def _loop(
        self,
        stream: VideoStream,
        display: bool,
        writer: cv2.VideoWriter | None,
    ) -> None:
        """Core processing loop."""
        prev_time = time.perf_counter()

        while stream.is_opened:
            frame = stream.read()
            if frame is None:
                continue

            result = self.process_frame(frame)
            self._frame_count += 1

            # Measure FPS
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            self._fps = 1.0 / dt if dt > 0 else 0.0

            if writer is not None:
                writer.write(result)

            if display:
                cv2.putText(
                    result,
                    f"FPS: {self._fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("PrivacyGuard", result)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    def _process_frame_with_metrics(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, list[Detection], float]:
        """Process a frame and return output, detections, and latency in ms."""
        start = time.perf_counter()
        out = frame.copy()
        detections = self.detect(out)
        result = self.anonymize(out, detections)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return result, detections, elapsed_ms
