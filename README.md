<p align="center">
  <h1 align="center">PrivacyGuard</h1>
  <p align="center">
    <strong>Privacy-oriented detection and masking pipeline for edge AI</strong><br>
    <em>Local processing with configurable detection and masking</em>
  </p>
  <p align="center">
    <a href="QUICKSTART.md">Quick Start</a> &nbsp;&bull;&nbsp;
    <a href="ARCHITECTURE.md">Architecture</a> &nbsp;&bull;&nbsp;
    <a href="BENCHMARKS.md">Benchmarks</a> &nbsp;&bull;&nbsp;
    <a href="src/privacyguard/api_reference.md">API Reference</a> &nbsp;&bull;&nbsp;
    <a href="COMMUNITY_FIRST.md">Community</a> &nbsp;&bull;&nbsp;
    <a href="CONTRIBUTING.md">Contributing</a>
  </p>
  <p align="center">
    <img alt="CI" src="https://github.com/laythayache/privacyguard/actions/workflows/ci.yml/badge.svg">
    <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
    <img alt="Code style" src="https://img.shields.io/badge/code%20style-ruff-purple">
  </p>
</p>

---

**PrivacyGuard** is a computer-vision pipeline that detects selected regions and applies configurable masking to images, recorded video, and live streams. It can run locally without sending frames to a cloud inference API.

Masking quality depends on detection quality. Missed or heavily occluded regions cannot be masked, blur is not encryption, and this software does not by itself establish legal compliance. Validate the selected model, labels, thresholds, input conditions, and deployment controls for your own use case.

### Why PrivacyGuard?

| Challenge | Solution |
|-----------|----------|
| **Deployment control** | Process frames locally and choose what leaves the device |
| **Performance** | Benchmark your exact model, input, hardware, runtime, and masking settings; see [Benchmarks](BENCHMARKS.md) |
| **Privacy-oriented processing** | Avoid a cloud inference dependency when the pipeline is configured to run locally |
| **Robustness** | Multi-model ensemble, adaptive blurring, regression-tested |
| **Integration** | 3-line API, CLI tool, easy to embed in existing pipelines |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    PrivacyGuard                       │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ VideoStream  │→│ ONNXDetector  │→│  Anonymizer  │ │
│  │  (threaded)  │  │ (YOLOv8/ONNX)│  │ (blur/pixel)│ │
│  └─────────────┘  └──────────────┘  └─────────────┘ │
│                                                      │
│  Sources:           Models:           Methods:       │
│  • Webcam           • YOLOv8-nano     • Gaussian     │
│  • Video file       • YOLOv8-small    • Pixelate     │
│  • RTSP stream      • Any ONNX        • Solid fill   │
└──────────────────────────────────────────────────────┘
```

## Installation

PrivacyGuard is not currently published as a verified package on PyPI. Install
the public repository from source:

```bash
git clone https://github.com/laythayache/privacyguard.git
cd privacyguard
pip install -e ".[dev]"
```

**GPU acceleration (optional):**

```bash
pip install -e ".[gpu]"
```

### Requirements

- Python 3.9+
- OpenCV 4.8+
- ONNX Runtime 1.16+
- A compatible ONNX detection model (see [Model Setup](#model-setup))

## Quick Start

### 3-Line Live Webcam Masking

```python
from privacyguard import PrivacyGuard

guard = PrivacyGuard("yolov8n-face.onnx")
guard.run(source=0)  # webcam — press 'q' to quit
```

### Process a Single Image

```python
from privacyguard import PrivacyGuard

guard = PrivacyGuard("yolov8n-face.onnx", method="pixelate")
detections = guard.process_image("photo.jpg", "photo_safe.jpg")
print(f"Anonymized {len(detections)} regions")
```

### Process a Video File

```python
guard = PrivacyGuard("yolov8n-face.onnx")
guard.process_video("input.mp4", "output_safe.mp4")
```

### Fine-Grained Control

```python
import cv2
from privacyguard import PrivacyGuard

guard = PrivacyGuard(
    model_path="yolov8n-face.onnx",
    method="gaussian",           # "gaussian" | "pixelate" | "solid"
    conf_threshold=0.5,          # detection confidence
    target_classes=[0],          # only anonymize faces (skip plates)
    padding=10,                  # expand blur region by 10px
)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Option A: one-shot
    result = guard.process_frame(frame)

    # Option B: inspect detections first
    detections = guard.detect(frame)
    for det in detections:
        print(f"{det.label}: {det.confidence:.0%} at ({det.x1},{det.y1})-({det.x2},{det.y2})")
    result = guard.anonymize(frame, detections)

    cv2.imshow("PrivacyGuard", result)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
```

### CLI Usage

```bash
# Live webcam
privacyguard model.onnx

# Video file with pixelation
privacyguard model.onnx -s input.mp4 -m pixelate -o output.mp4

# RTSP stream, no preview
privacyguard model.onnx -s "rtsp://192.168.1.10:554/stream" --no-display -o recording.mp4
```

## Experimental Regional Processing

PrivacyGuard includes experimental modules for Arabic plate, text, document,
and mixed-script workflows. They require deployment-specific models and
validation; their presence in the repository is not evidence of accuracy on a
particular country's plates, documents, scripts, or camera conditions.

### Arabic License Plate Processing
Use a compatible custom model and optional format heuristics for plate regions.

```python
from privacyguard.detectors.arabic_plate import ArabicPlateDetector, PlateConfig

config = PlateConfig(model_path="yolov8-arabic-plates.onnx")
detector = ArabicPlateDetector(config)
detections = detector.detect(frame)
```

The supplied module does not include a validated plate model or a published
regional benchmark. Verify the model labels, format assumptions, and false
negatives against representative footage.

### Arabic Text Detection and Masking
Detect and blur Arabic text regions while preserving visual context (for bilingual documents).

```python
from privacyguard.detectors.text import ArabicTextDetector, TextDetectorConfig

config = TextDetectorConfig(use_paddle_ocr=True)
detector = ArabicTextDetector(config)
result = detector.anonymize_text(frame)
```

**Features:**
- Arabic + Latin + Mixed script detection
- PaddleOCR integration (optional, falls back to contour-based detection)
- Per-script selective blurring
- Document-aware processing

### Identity Document Masking
Selectively blur ID cards, passports, and driving licenses while **preserving face visibility** for recognition.

```python
from privacyguard.detectors.document import DocumentDetector, DocumentConfig

config = DocumentConfig(blur_strategy="selective", preserve_face=True)
detector = DocumentDetector(config)
result = detector.anonymize_frame(frame)
```

**Strategies:**
- `"selective"`: Blur text/numbers, keep face visible
- `"full"`: Blur entire document

### Multi-Script Processing
Process bilingual documents (Arabic-French/English) with different strategies per script.

```python
from privacyguard.detectors.multiscript import MultiScriptProcessor, MultiScriptConfig

processor = MultiScriptProcessor(MultiScriptConfig())
result = processor.process_mixed_document(
    frame,
    arabic_strategy="blur",
    latin_strategy="preserve"
)
```

**Examples:**
- `examples/arabic_plate_detection.py` — Real-time plate detection
- `examples/arabic_text_anonymization.py` — Text region anonymization
- `examples/document_anonymization.py` — Selective document blur

## Operational Helpers

## 📚 API Reference

See [API Reference](src/privacyguard/api_reference.md) for full class/method documentation and advanced usage examples.

### Advanced Features
- Per-class anonymization
- Custom post-processing hooks
- Batch processing, audit logging, real-time monitoring
- Status watermarking and persistent region masking


PrivacyGuard includes optional helpers for logging software operations, batch
processing, runtime monitoring, and fixed-region masking:

### Audit Logging
Record masking operations for operational review:

```python
from privacyguard.enterprise import AuditLogger

logger = AuditLogger("audit_trail.json")
logger.log_anonymization(
    source_file="video.mp4",
    output_file="anonymized.mp4",
    detections_count=42,
    processing_time_ms=33,
    anonymization_method="gaussian",
    model_name="yolov8-face"
)

# Generate an operation summary
report = logger.get_operation_summary()
# → {"total_operations": 1000, "total_detections": 42000, ...}
```

### Batch Processing
Process directories of files with progress tracking:

```python
from privacyguard.enterprise import BatchProcessor

processor = BatchProcessor("model.onnx", output_dir="anonymized/")
results = processor.process_directory("images/", pattern="*.jpg")
# → {"total_files": 500, "successful": 495, "failed": 5, "total_time_sec": 120}
```

### Runtime Monitoring
Monitor FPS, latency, and performance anomalies:

```python
from privacyguard.enterprise import RealTimeMonitor

monitor = RealTimeMonitor("camera_1")

for frame in stream:
    start = time.time()
    detections = guard.detect(frame)
    result = guard.anonymize(frame, detections)
    elapsed_ms = (time.time() - start) * 1000

    monitor.record_frame(elapsed_ms, len(detections))

    stats = monitor.get_stats()
    print(f"FPS: {stats['fps']:.1f}, P95: {stats['p95_latency_ms']:.1f}ms")

    if monitor.should_alert(fps_threshold=20):
        send_alert("Performance degraded!")
```

### Custom Region Masking (Flexibility)
Define zones that should always be masked:

```python
from privacyguard.enterprise import CustomRegionMasker

masker = CustomRegionMasker()
masker.add_region("company_logo", x1=0, y1=0, x2=200, y2=100, method="solid")
masker.add_region("door_sign", x1=500, y1=200, x2=700, y2=400, method="gaussian")

result = masker.apply_masks(frame)
masker.save_config("regions.json")  # Reuse later
```

### Status Watermark
Add a visible processing-status label. A watermark is not legal proof or certification:

```python
from privacyguard.enterprise import ProcessingStatusWatermark

result = ProcessingStatusWatermark.add_status_label(frame, text="MASKING APPLIED")
# → Frame with status label + timestamp
```

---

## 📋 Deployment and Governance

PrivacyGuard is one component in a larger data-processing system. Before deployment, document and review:

- the lawful basis and purpose for processing;
- detection limits, representative evaluation footage, and acceptable failure modes;
- data flows, storage, retention, access controls, and incident handling;
- human review, notices, consent, contracts, and jurisdiction-specific requirements.

Files in the `compliance/` directory are preliminary engineering checklists. They are not legal advice, certification, or evidence that a deployment complies with any law. Compliance depends on the complete system and the organization operating it.

---

## Open-source project

The current public code is available under the MIT License. That makes the
implementation inspectable and permits modification and commercial use subject
to the license; it does not by itself establish security, privacy, adoption, or
fitness for a deployment.

### How You Can Help
- **Use it:** Evaluate it against your own footage, threat model, and deployment requirements
- **Contribute:** Code, documentation, examples, translations
- **Share:** Publish reproducible measurements and documented limitations

**Learn more:** See [COMMUNITY_FIRST.md](COMMUNITY_FIRST.md) for project scope and contribution guidance.

---

## Model Setup

PrivacyGuard works with compatible ONNX object detection models that follow its supported YOLOv8 or SSD output conventions. Example model profiles:

| Model profile | Typical use case | Tradeoff to validate |
|---|---|---|
| YOLOv8n face model | Faces only | Smaller model; verify recall on your footage |
| YOLOv8n custom model | Faces and plates | Requires a compatible label mapping and representative evaluation data |
| YOLOv8s custom model | Higher-capacity detection | Higher compute and memory requirements |

The repository does not publish one universal FPS result. Use [BENCHMARKS.md](BENCHMARKS.md) to record a reproducible measurement for your configuration.

**Export a YOLOv8 model to ONNX:**

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640
```

**Custom class labels:**

```python
guard = PrivacyGuard(
    "custom_model.onnx",
    class_labels={0: "face", 1: "license_plate", 2: "person"},
    target_classes=[0, 1],  # skip "person", only blur faces and plates
)
```

## API Reference

### `PrivacyGuard(model_path, **kwargs)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str \| Path` | *required* | Path to ONNX model |
| `method` | `str` | `"gaussian"` | `"gaussian"`, `"pixelate"`, or `"solid"` |
| `conf_threshold` | `float` | `0.4` | Detection confidence threshold |
| `iou_threshold` | `float` | `0.45` | NMS IoU threshold |
| `input_size` | `tuple` | `(640, 640)` | Model input resolution |
| `target_classes` | `list[int]` | `None` | Class IDs to anonymize (None = all) |
| `padding` | `int` | `0` | Pixels to expand each detection |

**Methods:**

| Method | Description |
|---|---|
| `process_frame(frame)` | Detect + anonymize a single frame (returns copy) |
| `detect(frame)` | Run detection only, returns `list[Detection]` |
| `anonymize(frame, detections)` | Apply anonymization to given detections |
| `process_image(in_path, out_path)` | Read, anonymize, and save an image file |
| `process_video(in_path, out_path)` | Process an entire video file |
| `run(source, display, output_path)` | Real-time processing loop |

### `Detection`

Immutable dataclass returned by `detect()`:

```python
Detection(x1=120, y1=80, x2=220, y2=190, confidence=0.94, class_id=0, label="face")
```

## Examples

See the [`examples/`](examples/) directory:

- **[`webcam_demo.py`](examples/webcam_demo.py)** — Live camera anonymization
- **[`video_file_demo.py`](examples/video_file_demo.py)** — Process a video file
- **[`batch_images.py`](examples/batch_images.py)** — Batch-process a directory of images

## Performance Tips

1. **Start with a small model** when latency and device resources are constrained, then validate detection quality
2. **Reduce `input_size`** to `(320, 320)` to improve throughput at the cost of detection detail
3. **Evaluate an accelerated ONNX Runtime provider** when the deployment hardware supports it
4. **Limit downstream masking work** to the classes the deployment requires

## Contributing

```bash
git clone https://github.com/laythayache/privacyguard.git
cd privacyguard
pip install -e ".[dev]"
pytest                    # run tests
ruff check src/ tests/    # lint
mypy src/privacyguard/    # type check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE) — use it freely in commercial and open-source projects.
