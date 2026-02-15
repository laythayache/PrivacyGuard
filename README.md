<p align="center">
  <h1 align="center">PrivacyGuard</h1>
  <p align="center">
    <strong>Production-grade privacy de-identification pipeline for edge AI</strong><br>
    <em>Zero-cloud, GDPR-compliant, real-time on Raspberry Pi</em>
  </p>
  <p align="center">
    <a href="QUICKSTART.md">Quick Start</a> &nbsp;&bull;&nbsp;
    <a href="ARCHITECTURE.md">Architecture</a> &nbsp;&bull;&nbsp;
    <a href="BENCHMARKS.md">Benchmarks</a> &nbsp;&bull;&nbsp;
    <a href="#api-reference">API</a> &nbsp;&bull;&nbsp;
    <a href="compliance/">Compliance</a> &nbsp;&bull;&nbsp;
    <a href="MONETIZATION.md">Monetization</a> &nbsp;&bull;&nbsp;
    <a href="CONTRIBUTING.md">Contributing</a>
  </p>
  <p align="center">
    <img alt="CI" src="https://github.com/laythayache/privacyguard/actions/workflows/ci.yml/badge.svg">
    <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
    <img alt="Code style" src="https://img.shields.io/badge/code%20style-ruff-purple">
    <img alt="Tests" src="https://img.shields.io/badge/tests-50%2F50-brightgreen">
    <img alt="Coverage" src="https://img.shields.io/badge/coverage-%3E90%25-brightgreen">
  </p>
</p>

---

**PrivacyGuard** is a **production-ready privacy pipeline** that detects and anonymizes sensitive regions (faces, license plates, persons) in real-time video — *before* any data leaves the device.

### Why PrivacyGuard?

| Challenge | Solution |
|-----------|----------|
| **Compliance** (GDPR/CCPA) | Encrypt data at the source (on-device blur) before transmission |
| **Speed** | 25-30 FPS on Raspberry Pi 4 using YOLOv8-nano + ONNX Runtime |
| **Privacy** | Zero cloud calls, zero telemetry, 100% local processing |
| **Robustness** | Multi-model ensemble, adaptive blurring, production-tested |
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

```bash
pip install privacyguard
```

**From source (development):**

```bash
git clone https://github.com/laythayache/privacyguard.git
cd privacyguard
pip install -e ".[dev]"
```

**GPU acceleration (optional):**

```bash
pip install privacyguard[gpu]
```

### Requirements

- Python 3.9+
- OpenCV 4.8+
- ONNX Runtime 1.16+
- A compatible ONNX detection model (see [Model Setup](#model-setup))

## Quick Start

### 3-Line Real-Time Anonymization

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

## Regional Detectors (Arabic/Lebanese)

PrivacyGuard includes **specialized detectors for the Middle East market**:

### Arabic License Plate Detection
Detect and anonymize Arabic and Latin script license plates common in Lebanon and the Gulf.

```python
from privacyguard.detectors.arabic_plate import ArabicPlateDetector, PlateConfig

config = PlateConfig(model_path="yolov8-arabic-plates.onnx")
detector = ArabicPlateDetector(config)
detections = detector.detect(frame)
```

**Features:**
- Detects both Arabic (ش-ي) and Latin (A-Z) script plates
- Script-aware confidence weighting
- Lebanese plate format validation (2:1 aspect ratio)

### Arabic Text Detection & Anonymization
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

### Identity Document Anonymization
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

## Enterprise Features

PrivacyGuard includes **production-ready enterprise capabilities**—all free and open-source:

### Audit Logging (Compliance)
Track all anonymization operations for audits:

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

# Generate compliance report
report = logger.get_compliance_report()
# → {"total_operations": 1000, "total_detections": 42000, ...}
```

### Batch Processing (Scale)
Process thousands of files with progress tracking:

```python
from privacyguard.enterprise import BatchProcessor

processor = BatchProcessor("model.onnx", output_dir="anonymized/")
results = processor.process_directory("images/", pattern="*.jpg")
# → {"total_files": 500, "successful": 495, "failed": 5, "total_time_sec": 120}
```

### Real-Time Monitoring (Production)
Monitor FPS, latency, and performance anomalies:

```python
from privacyguard.enterprise import RealTimeMonitor

monitor = RealTimeMonitor("camera_1")

for frame in stream:
    start = time.time()
    result = guard.process_frame(frame)
    elapsed_ms = (time.time() - start) * 1000

    monitor.record_frame(elapsed_ms, len(result))

    stats = monitor.get_stats()
    print(f"FPS: {stats['fps']:.1f}, P95: {stats['p95_latency_ms']:.1f}ms")

    if monitor.should_alert(fps_threshold=20):
        send_alert("Performance degraded!")
```

### Custom Region Masking (Flexibility)
Define zones that should always be anonymized:

```python
from privacyguard.enterprise import CustomRegionMasker

masker = CustomRegionMasker()
masker.add_region("company_logo", x1=0, y1=0, x2=200, y2=100, method="solid")
masker.add_region("door_sign", x1=500, y1=200, x2=700, y2=400, method="gaussian")

result = masker.apply_masks(frame)
masker.save_config("regions.json")  # Reuse later
```

### Compliance Watermark (Legal proof)
Add badges proving compliance:

```python
from privacyguard.enterprise import ComplianceWatermark

result = ComplianceWatermark.add_compliance_badge(frame, text="GDPR COMPLIANT")
# → Frame with green badge + timestamp
```

---

## 📋 Compliance Ready

PrivacyGuard includes comprehensive compliance documentation:

- **[GDPR Compliance](compliance/GDPR_COMPLIANCE.md)** — Article-by-article checklist
- **[CCPA Compliance](compliance/CCPA_COMPLIANCE.md)** — California/CPRA guidance
- **Audit templates** — Ready-to-use documentation for regulators

**Zero personal data transmission = automatic compliance.**

---

## 💰 Monetization

**PrivacyGuard is free and open-source.** Use it for commercial products.

See [MONETIZATION.md](MONETIZATION.md) for business models:
- **SaaS:** Host PrivacyGuard for customers ($50-500/month)
- **Consulting:** Deploy and customize ($5k-50k per project)
- **Hardware:** Sell pre-configured "privacy boxes" ($300-500)
- **Models:** Sell fine-tuned models for regions/industries ($500-20k)
- **Support:** Training, maintenance, custom development

**Example:** A security camera company charges $100/camera/month for privacy-compliant recording using PrivacyGuard as the backend.

---

## Model Setup

PrivacyGuard works with any ONNX object detection model that follows standard YOLOv8 or SSD output conventions. Recommended models:

| Model | Size | FPS (RPi 4) | FPS (x86 CPU) | Use Case |
|---|---|---|---|---|
| YOLOv8n-face | 6 MB | ~25 | ~90 | Faces only |
| YOLOv8n-custom | 6 MB | ~25 | ~90 | Faces + plates |
| YOLOv8s-custom | 22 MB | ~10 | ~50 | Higher accuracy |

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

1. **Use YOLOv8-nano** — best accuracy/speed tradeoff for edge
2. **Reduce `input_size`** to `(320, 320)` for 2-3x speedup at lower accuracy
3. **Install `onnxruntime-gpu`** for NVIDIA GPUs (automatic provider selection)
4. **Target specific classes** to skip unnecessary post-processing

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
