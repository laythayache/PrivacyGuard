"""ONNX-based object detector optimized for edge deployment.

Supports YOLOv8-nano and compatible architectures for detecting
faces, license plates, and other sensitive regions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Detection:
    """A single detected region."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    label: str = ""


class ONNXDetector:
    """Runs inference on an ONNX model and returns detections.

    Handles YOLOv8-style output (shape [1, 4+num_classes, N]) as well as
    the traditional [N, 6] format (x1, y1, x2, y2, conf, cls).
    """

    # Default class names for a face/plate model
    DEFAULT_LABELS: dict[int, str] = {0: "face", 1: "license_plate"}

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (640, 640),
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        providers: Sequence[str] | None = None,
        class_labels: dict[int, str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size  # (width, height)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_labels = class_labels or self.DEFAULT_LABELS

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        providers = providers or self._default_providers()
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _default_providers() -> list[str]:
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
        return [p for p in preferred if p in available] or ["CPUExecutionProvider"]

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Resize and normalize frame for model input.

        Returns the preprocessed tensor and (scale_x, scale_y) ratios
        needed to map detections back to the original frame.
        """
        h, w = frame.shape[:2]
        inp_w, inp_h = self.input_size
        scale_x, scale_y = w / inp_w, h / inp_h

        img = cv2.resize(frame, (inp_w, inp_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # HWC -> CHW, add batch dim
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # type: ignore[assignment]
        return img, (scale_x, scale_y)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run full detection pipeline on a single BGR frame."""
        tensor, scale = self.preprocess(frame)
        raw = self.session.run(None, {self.input_name: tensor})
        return self._postprocess(raw, scale, frame.shape)

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        scale: tuple[float, float],
        orig_shape: tuple[int, ...],
    ) -> list[Detection]:
        """Decode raw ONNX output into Detection objects."""
        out = outputs[0]

        # YOLOv8 output: [1, 4+num_classes, num_detections]
        if out.ndim == 3 and out.shape[1] < out.shape[2]:
            return self._postprocess_yolov8(out, scale, orig_shape)

        # Legacy / SSD-like output: [N, 6] -> (x1, y1, x2, y2, conf, cls)
        return self._postprocess_legacy(out, scale, orig_shape)

    def _postprocess_yolov8(
        self,
        out: np.ndarray,
        scale: tuple[float, float],
        orig_shape: tuple[int, ...],
    ) -> list[Detection]:
        """Decode YOLOv8 transposed output [1, 4+C, N] -> list[Detection]."""
        # Transpose to [N, 4+C]
        pred = out[0].T  # (N, 4+C)
        h_orig, w_orig = orig_shape[:2]
        sx, sy = scale

        # Split xywh and class scores
        xywh = pred[:, :4]
        scores = pred[:, 4:]

        # Best class per detection
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # Confidence filter
        mask = confidences > self.conf_threshold
        xywh, confidences, class_ids = xywh[mask], confidences[mask], class_ids[mask]

        if len(xywh) == 0:
            return []

        # xywh (center) -> xyxy (corner), scale to original
        x_c, y_c, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = ((x_c - w / 2) * sx).astype(int)
        y1 = ((y_c - h / 2) * sy).astype(int)
        x2 = ((x_c + w / 2) * sx).astype(int)
        y2 = ((y_c + h / 2) * sy).astype(int)

        # Clip to frame bounds
        np.clip(x1, 0, w_orig, out=x1)
        np.clip(y1, 0, h_orig, out=y1)
        np.clip(x2, 0, w_orig, out=x2)
        np.clip(y2, 0, h_orig, out=y2)

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        return self._apply_nms(boxes, confidences, class_ids)

    def _postprocess_legacy(
        self,
        out: np.ndarray,
        scale: tuple[float, float],
        orig_shape: tuple[int, ...],
    ) -> list[Detection]:
        """Decode [N, 6] format (x1, y1, x2, y2, conf, cls)."""
        if out.ndim == 3:
            out = out[0]

        h_orig, w_orig = orig_shape[:2]
        sx, sy = scale

        mask = out[:, 4] > self.conf_threshold
        out = out[mask]
        if len(out) == 0:
            return []

        boxes = out[:, :4].copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * sx).clip(0, w_orig)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * sy).clip(0, h_orig)
        boxes = boxes.astype(int)

        confidences = out[:, 4]
        class_ids = out[:, 5].astype(int)
        return self._apply_nms(boxes, confidences, class_ids)

    def _apply_nms(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
    ) -> list[Detection]:
        """Apply per-class Non-Maximum Suppression using OpenCV."""
        if len(boxes) == 0:
            return []

        # OpenCV NMS expects (x, y, w, h)
        xywh = boxes.copy()
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()  # type: ignore[assignment]
        detections: list[Detection] = []
        for i in indices:
            cid = int(class_ids[i])
            detections.append(
                Detection(
                    x1=int(boxes[i, 0]),
                    y1=int(boxes[i, 1]),
                    x2=int(boxes[i, 2]),
                    y2=int(boxes[i, 3]),
                    confidence=float(confidences[i]),
                    class_id=cid,
                    label=self.class_labels.get(cid, f"class_{cid}"),
                )
            )
        return detections
