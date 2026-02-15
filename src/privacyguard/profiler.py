"""Performance profiling and benchmarking utilities.

Measures inference time, throughput, memory usage, and generates
detailed performance reports for optimization.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""

    timestamp: float
    inference_time_ms: float
    detection_count: int
    confidence_mean: float = 0.0


@dataclass
class ProfileReport:
    """Aggregated profiling report."""

    total_frames: int
    total_time_sec: float
    fps: float
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_confidence: float
    memory_peak_mb: float
    metrics: list[FrameMetrics] = field(default_factory=list)

    def __str__(self) -> str:
        return f"""
╔═══════════════════════════════════════════╗
║          PERFORMANCE REPORT               ║
╠═══════════════════════════════════════════╣
║ Frames Processed:    {self.total_frames:>25} ║
║ Total Time:          {self.total_time_sec:>24.2f}s ║
║ Throughput:          {self.fps:>26.1f} FPS ║
║                                           ║
║ Latency (mean):      {self.mean_latency_ms:>24.2f}ms ║
║ Latency (p95):       {self.p95_latency_ms:>24.2f}ms ║
║ Latency (p99):       {self.p99_latency_ms:>24.2f}ms ║
║                                           ║
║ Mean Confidence:     {self.mean_confidence:>24.1%} ║
║ Peak Memory:         {self.memory_peak_mb:>24.2f} MB ║
╚═══════════════════════════════════════════╝
"""


class Profiler:
    """Context-based profiler for performance tracking."""

    def __init__(self) -> None:
        self.metrics: list[FrameMetrics] = []
        self._start_time: float | None = None
        self._start_memory: float | None = None
        self._peak_memory: float = 0.0

    def start(self) -> None:
        """Begin profiling session."""
        gc.collect()
        self._start_memory = self._get_memory_mb()
        self._start_time = time.perf_counter()

    def record_frame(
        self,
        latency_ms: float,
        detection_count: int,
        confidences: list[float] | None = None,
    ) -> None:
        """Record metrics for a single frame."""
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        metric = FrameMetrics(
            timestamp=time.perf_counter() - (self._start_time or 0.0),
            inference_time_ms=latency_ms,
            detection_count=detection_count,
            confidence_mean=mean_conf,
        )
        self.metrics.append(metric)

    def stop(self) -> ProfileReport:
        """End profiling and generate report."""
        elapsed = time.perf_counter() - (self._start_time or 0.0)
        peak_mem = self._peak_memory

        latencies = [m.inference_time_ms for m in self.metrics]
        latencies_sorted = sorted(latencies)

        fps = len(self.metrics) / elapsed if elapsed > 0 else 0.0
        mean_latency = float(np.mean(latencies)) if latencies else 0.0
        p95_latency = float(np.percentile(latencies_sorted, 95)) if latencies else 0.0
        p99_latency = float(np.percentile(latencies_sorted, 99)) if latencies else 0.0
        mean_conf = float(
            np.mean([m.confidence_mean for m in self.metrics if m.confidence_mean > 0])
        )

        return ProfileReport(
            total_frames=len(self.metrics),
            total_time_sec=elapsed,
            fps=fps,
            mean_latency_ms=mean_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            mean_confidence=mean_conf,
            memory_peak_mb=peak_mem,
            metrics=self.metrics,
        )

    @staticmethod
    def _get_memory_mb() -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return float(process.memory_info().rss / 1024 / 1024)
        except ImportError:
            return 0.0

    @staticmethod
    def timeit(func: Callable, *args, **kwargs) -> tuple[float, object]:
        """Time a single function call (returns elapsed_ms, result)."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms, result
