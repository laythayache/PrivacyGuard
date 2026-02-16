"""Multi-model ensemble for robust detection across different scenarios.

Runs multiple detectors in parallel (face, plate, person) and merges results
using confidence-weighted voting and NMS across models.

Supports regional detectors for Arabic text and documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .detector import Detection, ONNXDetector
from .detectors.arabic_plate import ArabicPlateDetector, PlateConfig
from .detectors.document import DocumentConfig, DocumentDetector
from .detectors.text import (
    ArabicTextDetector,
    TextDetectorConfig,
)


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for ensemble detection."""

    # Standard detectors
    face_model: str | None = None
    plate_model: str | None = None
    person_model: str | None = None

    # Regional detectors
    enable_arabic_plate: bool = False
    enable_text_detection: bool = False
    enable_document_detection: bool = False

    # Thresholds
    confidence_threshold: float = 0.4
    nms_threshold: float = 0.45
    ensemble_mode: str = "union"  # "union" | "intersection" | "weighted_vote"


class EnsembleDetector:
    """Runs multiple detectors and merges results for robustness.

    Advantages:
    - Face detection + plate detection in parallel (faster than sequential)
    - Confidence weighting (high-confidence detections take precedence)
    - Model-specific thresholds (e.g., stricter for plates than faces)
    - Graceful degradation (works even if one model is missing)
    """

    def __init__(
        self,
        config: EnsembleConfig,
        providers: list[str] | None = None,
    ) -> None:
        self.config = config
        self.detectors: dict[str, ONNXDetector | None] = {}
        self.model_weights: dict[str, float] = {
            "face": 1.0,
            "plate": 1.2,  # Higher weight = stricter threshold
            "person": 0.8,
        }

        # Load models that are configured
        if config.face_model:
            self.detectors["face"] = ONNXDetector(
                config.face_model,
                conf_threshold=config.confidence_threshold * 0.95,
                providers=providers,
                class_labels={0: "face"},
            )
        if config.plate_model:
            self.detectors["plate"] = ONNXDetector(
                config.plate_model,
                conf_threshold=config.confidence_threshold * 1.1,  # Stricter for plates
                providers=providers,
                class_labels={0: "license_plate"},
            )
        if config.person_model:
            self.detectors["person"] = ONNXDetector(
                config.person_model,
                conf_threshold=config.confidence_threshold * 0.9,
                providers=providers,
                class_labels={0: "person"},
            )

        # Regional detectors (non-ONNX)
        self.regional_detectors: dict[str, object] = {}

        if config.enable_arabic_plate:
            plate_config = PlateConfig(confidence_threshold=config.confidence_threshold)
            self.regional_detectors["arabic_plate"] = ArabicPlateDetector(plate_config, providers)

        if config.enable_text_detection:
            text_config = TextDetectorConfig(confidence_threshold=config.confidence_threshold)
            self.regional_detectors["text"] = ArabicTextDetector(text_config)

        if config.enable_document_detection:
            doc_config = DocumentConfig(confidence_threshold=config.confidence_threshold)
            self.regional_detectors["document"] = DocumentDetector(doc_config)

        if not self.detectors and not self.regional_detectors:
            raise ValueError("At least one detector must be configured")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run all detectors in parallel and merge results."""
        all_detections: list[Detection] = []

        # Run each detector and collect results
        for model_name, detector in self.detectors.items():
            if detector is None:
                continue
            dets = detector.detect(frame)
            # Tag detections with source model
            for det in dets:
                all_detections.append(
                    Detection(
                        x1=det.x1,
                        y1=det.y1,
                        x2=det.x2,
                        y2=det.y2,
                        confidence=det.confidence * self.model_weights[model_name],
                        class_id=det.class_id,
                        label=f"{det.label}_{model_name}",
                    )
                )

        # Run optional regional detectors and normalize their output
        all_detections.extend(self._detect_regional(frame))

        # Merge using the selected strategy
        if self.config.ensemble_mode == "union":
            return self._merge_union(all_detections)
        elif self.config.ensemble_mode == "weighted_vote":
            return self._merge_weighted_vote(all_detections)
        elif self.config.ensemble_mode == "intersection":
            return self._merge_intersection(all_detections)
        raise ValueError(f"Unsupported ensemble_mode: {self.config.ensemble_mode}")

    def _detect_regional(self, frame: np.ndarray) -> list[Detection]:
        """Collect detections from regional detectors and normalize to Detection."""
        detections: list[Detection] = []
        for name, detector in self.regional_detectors.items():
            if name == "arabic_plate" and isinstance(detector, ArabicPlateDetector):
                for det in detector.detect(frame):
                    detections.append(
                        Detection(
                            x1=det.x1,
                            y1=det.y1,
                            x2=det.x2,
                            y2=det.y2,
                            confidence=det.confidence,
                            class_id=det.class_id,
                            label=f"{det.label}_{name}" if det.label else name,
                        )
                    )
            elif name == "text" and isinstance(detector, ArabicTextDetector):
                for region in detector.detect_text_regions(frame):
                    detection = self._region_to_detection(
                        region=region,
                        default_label="text",
                        class_id=100,
                        source=name,
                    )
                    if detection is not None:
                        detections.append(detection)
            elif name == "document" and isinstance(detector, DocumentDetector):
                for region in detector.detect_document_regions(frame):
                    detection = self._region_to_detection(
                        region=region,
                        default_label="document",
                        class_id=200,
                        source=name,
                    )
                    if detection is not None:
                        detections.append(detection)
        return detections

    @staticmethod
    def _region_to_detection(
        region: dict[str, Any],
        default_label: str,
        class_id: int,
        source: str,
    ) -> Detection | None:
        """Convert a regional detector dictionary payload to Detection."""
        bbox = region.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        confidence = float(region.get("confidence", 0.5))
        raw_label = region.get("script", region.get("document_type", default_label))
        label = raw_label.value if hasattr(raw_label, "value") else str(raw_label)
        return Detection(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=max(0.0, min(confidence, 1.0)),
            class_id=class_id,
            label=f"{label}_{source}",
        )

    def _merge_union(self, dets: list[Detection]) -> list[Detection]:
        """Union: keep all detections, sort by confidence."""
        return sorted(dets, key=lambda d: d.confidence, reverse=True)

    def _merge_weighted_vote(self, dets: list[Detection]) -> list[Detection]:
        """Weighted voting: merge overlapping boxes from different models."""
        if not dets:
            return []

        # Sort by confidence
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)

        # Greedy NMS with confidence weighting
        merged: list[Detection] = []
        used = set()

        for i, det in enumerate(dets):
            if i in used:
                continue

            # Find all overlapping detections
            group = [det]
            for j, other in enumerate(dets[i + 1 :], start=i + 1):
                if j in used:
                    continue
                if self._iou(det, other) > 0.3:  # Loose threshold for merging
                    group.append(other)
                    used.add(j)

            # Create merged detection (weighted average)
            if group:
                avg_conf = np.mean([d.confidence for d in group])
                # Filter out None labels before joining
                labels = [d.label for d in group if d.label is not None]
                merged.append(
                    Detection(
                        x1=int(np.mean([d.x1 for d in group])),
                        y1=int(np.mean([d.y1 for d in group])),
                        x2=int(np.mean([d.x2 for d in group])),
                        y2=int(np.mean([d.y2 for d in group])),
                        confidence=float(avg_conf),
                        class_id=group[0].class_id,
                        label="+".join(set(labels)) if labels else None,
                    )
                )
            used.add(i)

        return merged

    def _merge_intersection(self, dets: list[Detection]) -> list[Detection]:
        """Intersection: keep detections supported by overlap with another detection."""
        if not dets:
            return []
        if len(dets) == 1:
            return dets

        merged: list[Detection] = []
        used: set[int] = set()
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)

        for i, det in enumerate(dets):
            if i in used:
                continue

            group = [det]
            for j, other in enumerate(dets[i + 1 :], start=i + 1):
                if j in used:
                    continue
                if self._iou(det, other) >= 0.3:
                    group.append(other)
                    used.add(j)

            if len(group) >= 2:
                labels = sorted({d.label for d in group if d.label})
                merged.append(
                    Detection(
                        x1=int(np.mean([d.x1 for d in group])),
                        y1=int(np.mean([d.y1 for d in group])),
                        x2=int(np.mean([d.x2 for d in group])),
                        y2=int(np.mean([d.y2 for d in group])),
                        confidence=float(np.mean([d.confidence for d in group])),
                        class_id=group[0].class_id,
                        label="+".join(labels) if labels else None,
                    )
                )
                used.add(i)
        return merged

    @staticmethod
    def _iou(det1: Detection, det2: Detection) -> float:
        """Compute Intersection over Union."""
        x1_min, y1_min, x1_max, y1_max = det1.x1, det1.y1, det1.x2, det1.y2
        x2_min, y2_min, x2_max, y2_max = det2.x1, det2.y1, det2.x2, det2.y2

        inter_x_min, inter_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
        inter_x_max, inter_y_max = min(x1_max, x2_max), min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0
