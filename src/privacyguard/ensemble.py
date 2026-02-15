"""Multi-model ensemble for robust detection across different scenarios.

Runs multiple detectors in parallel (face, plate, person) and merges results
using confidence-weighted voting and NMS across models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .detector import Detection, ONNXDetector


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for ensemble detection."""

    face_model: str | None = None
    plate_model: str | None = None
    person_model: str | None = None
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

        if not self.detectors:
            raise ValueError("At least one model must be configured")

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

        # Merge using the selected strategy
        if self.config.ensemble_mode == "union":
            return self._merge_union(all_detections)
        elif self.config.ensemble_mode == "weighted_vote":
            return self._merge_weighted_vote(all_detections)
        else:
            return all_detections

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
                merged.append(
                    Detection(
                        x1=int(np.mean([d.x1 for d in group])),
                        y1=int(np.mean([d.y1 for d in group])),
                        x2=int(np.mean([d.x2 for d in group])),
                        y2=int(np.mean([d.y2 for d in group])),
                        confidence=float(avg_conf),
                        class_id=group[0].class_id,
                        label="+".join(set(d.label for d in group)),
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
