# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `PrivacyGuard.process_frame()` now records real per-frame latency for monitors instead of hardcoded `0ms`.
- `PrivacyGuard.process_image()` and `PrivacyGuard.process_video()` now write real `detections_count` and `processing_time_ms` to `AuditLogger`.
- `RealTimeMonitor.get_stats()` no longer returns `inf` FPS when a frame latency sample is `0ms`.
- `CustomRegionMasker.apply_masks()` no longer crashes on tiny ROIs in `pixelate` mode.
- `EnsembleDetector` now consumes configured regional detectors (text/document/arabic plate) and supports `intersection` mode explicitly.
- `ONNXDetector` NMS is now truly per-class, preventing one class from suppressing another.
- `MetadataStripper.get_safe_filename()` regex patterns now correctly match and sanitize timestamps/device-like names.
- `Profiler.stop()` now returns stable confidence metrics for empty runs and tracks peak memory delta from profiling start.
- Arabic string literals/patterns in `utils/arabic.py` were corrected to valid Arabic text.

### Changed
- Added missing docs linked by README: `QUICKSTART.md`, `ARCHITECTURE.md`, and `BENCHMARKS.md`.
- Synced documentation to implemented behavior (removed stale claims like `in_place` support and CSV/invisible watermark support).
- Updated static README badges to match current local test count and measured coverage baseline.
- Aligned package version metadata (`pyproject.toml`) with runtime package version (`0.2.0`).

### Added
- Regression tests for:
  - monitor latency and audit metric correctness
  - tiny-ROI pixelation safety
  - per-class NMS behavior
  - regional/intersection ensemble behavior
  - metadata filename sanitization
  - profiler empty-run stability

## [0.2.0] - 2025-01-XX

### Added
- Enterprise module: audit logging, batch processing, real-time monitoring, custom region masking.
- Regional Arabic detectors (plate/text/document).
- Ensemble detection support.
- Compliance watermarking.
- Profiling and metadata stripping utilities.

## [0.1.0] - 2025-01-XX

### Added
- Initial release.
- Core ONNX detection + anonymization pipeline.
- Gaussian/pixelate/solid anonymization methods.
- Real-time stream, image, and video processing APIs.
- CLI and examples.
- MIT license.
