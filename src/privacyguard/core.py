"""Main orchestrator that ties detection, anonymization, and streaming together.

This is the primary public API surface. Most users only need this module.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .anonymizer import Anonymizer, Method
from .detector import Detection, ONNXDetector
from .stream import VideoStream


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
    ) -> None:
        self.detector = ONNXDetector(
            model_path=model_path,
            input_size=input_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            providers=providers,
            class_labels=class_labels,
        )
        self.anonymizer = Anonymizer(method=Method(method), padding=padding)
        self.target_classes = set(target_classes) if target_classes else None
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
        out = frame.copy()
        detections = self.detect(out)
        return self.anonymize(out, detections)

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
        detections = self.detect(frame)
        result = self.anonymize(frame, detections)
        cv2.imwrite(str(output_path), result)
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
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = self.process_frame(frame)
                writer.write(result)
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
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(output_path), fourcc, stream.fps, (w, h))

            try:
                self._loop(stream, display, writer)
            finally:
                if writer is not None:
                    writer.release()
                if display:
                    cv2.destroyAllWindows()

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
