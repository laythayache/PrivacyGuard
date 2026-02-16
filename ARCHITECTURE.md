# Architecture

PrivacyGuard is organized as a small set of composable modules:

- `privacyguard.detector.ONNXDetector`: model inference and post-processing.
- `privacyguard.anonymizer.Anonymizer`: region anonymization methods.
- `privacyguard.core.PrivacyGuard`: orchestration for frame/image/video pipelines.
- `privacyguard.stream.VideoStream`: threaded frame capture.
- `privacyguard.enterprise`: optional audit, monitoring, batch, and masking helpers.

## Data Flow

```text
Video/Image Source -> ONNXDetector -> Detection[] -> Anonymizer -> Output
```

## Design Notes

- Detection and anonymization are separated, so detections can be inspected or filtered before masking.
- Processing runs fully local on-device.
- Optional hooks (`postprocess_hook`, `monitor`, `audit_logger`) let integrators add behavior without forking core logic.

## Regional Processing

`privacyguard.detectors` contains specialized detectors for Arabic plate/text/document use cases.
These can be used standalone or via `EnsembleDetector`.
