"""Threaded video stream reader for low-latency capture.

Decouples frame capture from processing so the main thread never
blocks on I/O, keeping inference latency as the only bottleneck.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np


class VideoStream:
    """Threaded wrapper around cv2.VideoCapture.

    Args:
        source: Camera index (int) or video file / RTSP URL (str).
        queue_size: Not used; kept for API compat. The stream always
            holds the most recent frame (no queue delay).
    """

    def __init__(self, source: int | str = 0) -> None:
        self.source = source
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stopped = True

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        """Return (width, height) of the capture."""
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def start(self) -> VideoStream:
        """Open the capture device and begin the reader thread."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise OSError(f"Cannot open video source: {self.source}")

        self._stopped = False
        thread = threading.Thread(target=self._reader, daemon=True)
        thread.start()
        return self

    def read(self) -> np.ndarray | None:
        """Return the most recent frame (or None if unavailable)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        """Signal the reader thread to stop and release resources."""
        self._stopped = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _reader(self) -> None:
        """Continuously grab frames in a background thread."""
        while not self._stopped:
            if self._cap is None:
                break
            ret, frame = self._cap.read()
            if not ret:
                self._stopped = True
                break
            with self._lock:
                self._frame = frame

    def __enter__(self) -> VideoStream:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()
