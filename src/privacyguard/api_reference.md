# PrivacyGuard API Reference

## Main Classes

### PrivacyGuard
- End-to-end privacy de-identification pipeline.
- Args:
  - model_path: Path to ONNX model
  - method: Default anonymization method
  - conf_threshold, iou_threshold, input_size, class_labels, target_classes, providers, padding
  - class_methods: Per-class anonymization (dict)
  - postprocess_hook: Callable for custom post-processing
  - audit_logger: Audit logging integration
  - batch_processor: Batch processing integration
  - monitor: Real-time monitoring integration
- Methods:
  - process_frame(frame): Detect & anonymize single frame
  - detect(frame): Run detection only
  - anonymize(frame, detections): Apply anonymization
  - process_image(input_path, output_path): Anonymize image file
  - process_video(input_path, output_path): Anonymize video file
  - run(source, display, output_path): Live stream pipeline

### Anonymizer
- Applies anonymization to detected regions
- Supports Gaussian, pixelate, solid fill
- Per-class overrides

### ONNXDetector
- Runs inference on ONNX models
- Supports YOLOv8, legacy formats

### EnsembleDetector
- Multi-model ensemble for robust detection
- Union, weighted vote, intersection modes

### AuditLogger
- Records masking operations in a local JSON file
- Provides operational counts and timing summaries
- Does not create a tamper-evident record or establish legal compliance

### BatchProcessor
- Processes directories of images/videos
- Progress tracking, error handling

### RealTimeMonitor
- Tracks FPS, latency, detection rates
- Alerts on performance degradation

### CustomRegionMasker
- Defines custom regions for persistent masking

### ProcessingStatusWatermark
- Adds an informational status label and timestamp
- Does not prove masking success, consent, or legal compliance
- `ComplianceWatermark` and `add_compliance_badge` remain compatibility aliases

## Example Usage

```python
from privacyguard.core import PrivacyGuard
from privacyguard.enterprise import AuditLogger, BatchProcessor, RealTimeMonitor

# Per-class anonymization
class_methods = {0: 'pixelate', 1: 'gaussian'}

# Optional integrations
logger = AuditLogger('audit.json')
batch = BatchProcessor('model.onnx', 'output_dir', audit_logger=logger)
monitor = RealTimeMonitor()

guard = PrivacyGuard(
    model_path='model.onnx',
    method='gaussian',
    class_methods=class_methods,
    audit_logger=logger,
    batch_processor=batch,
    monitor=monitor,
)

# Process image
guard.process_image('input.jpg', 'output.jpg')

# Process video
guard.process_video('input.mp4', 'output.mp4')

# Live stream
guard.run(source=0)
```

## Advanced Features
- Use `postprocess_hook` to add custom overlays or watermark after anonymization.
- Use `CustomRegionMasker` for persistent region masking.
- Use `ProcessingStatusWatermark.add_status_label(frame, text)` for an informational overlay.

---

For full details, see source code and README.
