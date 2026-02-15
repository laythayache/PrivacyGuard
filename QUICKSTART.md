# PrivacyGuard: Quick Start Guide

Get privacy de-identification running in minutes.

---

## Installation

```bash
# Basic installation
pip install privacyguard

# With GPU support
pip install privacyguard[gpu]

# Development (for contributors)
git clone https://github.com/yourname/privacyguard.git
cd privacyguard
pip install -e ".[dev]"
```

---

## Minimal Example (3 lines)

```python
from privacyguard import PrivacyGuard

guard = PrivacyGuard("yolov8n-face.onnx")  # Load model
guard.run(source=0)  # Webcam — press 'q' to quit
```

---

## Image Processing

```python
from privacyguard import PrivacyGuard

guard = PrivacyGuard("yolov8n-face.onnx", method="pixelate")
detections = guard.process_image("photo.jpg", "photo_safe.jpg")
print(f"Anonymized {len(detections)} faces")
```

---

## Video Processing

```python
from privacyguard import PrivacyGuard

guard = PrivacyGuard("yolov8n-face.onnx")
guard.process_video("input.mp4", "output_safe.mp4")
```

---

## Advanced: Ensemble Detection

```python
from privacyguard import EnsembleConfig, EnsembleDetector, Anonymizer, Method

# Configure two models (face + plate)
config = EnsembleConfig(
    face_model="yolov8n-face.onnx",
    plate_model="yolov8n-plate.onnx",
    confidence_threshold=0.4,
)

# Run ensemble
detector = EnsembleDetector(config)
anonymizer = Anonymizer(
    method=Method.GAUSSIAN,
    class_methods={1: Method.PIXELATE},  # Pixelate plates
)

# Process
import cv2
frame = cv2.imread("dashcam.jpg")
detections = detector.detect(frame)
result = anonymizer.apply(frame, detections)
cv2.imwrite("anonymized.jpg", result)
```

---

## Advanced: Performance Profiling

```python
from privacyguard import PrivacyGuard, Profiler
import cv2

guard = PrivacyGuard("yolov8n-face.onnx")
profiler = Profiler()
profiler.start()

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    import time
    t0 = time.perf_counter()
    result = guard.process_frame(frame)
    latency_ms = (time.perf_counter() - t0) * 1000

    dets = guard.detect(frame)
    confidences = [d.confidence for d in dets]
    profiler.record_frame(latency_ms, len(dets), confidences)

report = profiler.stop()
print(report)  # Beautiful ASCII table with FPS, latency, memory
```

---

## Advanced: Adaptive Blurring

High-confidence faces get stronger blur:

```python
from privacyguard import PrivacyGuard, Anonymizer, Method

guard = PrivacyGuard("yolov8n-face.onnx")

for frame_path in ["frame1.jpg", "frame2.jpg"]:
    frame = cv2.imread(frame_path)
    detections = guard.detect(frame)

    for det in detections:
        # Adaptive kernel size: 20 + 20 * confidence
        kernel_size = int(20 + det.confidence * 20)
        if kernel_size % 2 == 0:
            kernel_size += 1

        adaptive_blur = Anonymizer(
            method=Method.GAUSSIAN,
            gaussian_ksize=(kernel_size, kernel_size),
        )
        frame = adaptive_blur.apply(frame, [det])

    cv2.imwrite(f"{frame_path[:-4]}_adaptive.jpg", frame)
```

---

## CLI Usage

```bash
# Webcam real-time
privacyguard model.onnx

# Video file
privacyguard model.onnx -s input.mp4 -o output.mp4 -m pixelate

# RTSP stream (security camera)
privacyguard model.onnx -s "rtsp://192.168.1.10:554/stream" --no-display -o recording.mp4

# Get help
privacyguard --help
```

---

## Getting Models

### Pre-trained YOLOv8

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640
```

This creates `yolov8n.onnx` (6 MB, face detection).

### Custom Models

Train your own using YOLOv8:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="dataset.yaml", epochs=100)
model.export(format="onnx")
```

---

## Troubleshooting

### "ONNX model not found"

```
Ensure the model path is correct:
  python -c "from pathlib import Path; print(Path('yolov8n.onnx').exists())"
```

### "No detections"

1. Lower confidence threshold: `conf_threshold=0.3`
2. Verify model is for faces (not objects, pose, segmentation)
3. Check image quality (very dark/blurry = hard to detect)

### "GPU not used"

```python
# Force CPU only
from privacyguard import PrivacyGuard
guard = PrivacyGuard("model.onnx", providers=["CPUExecutionProvider"])

# Or verify GPU
from privacyguard import PrivacyGuard
import onnxruntime
print(onnxruntime.get_available_providers())
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### "Out of memory"

1. Use nano model instead of large
2. Reduce input size: `input_size=(320, 320)`
3. Process 1 frame at a time (avoid batching)

### "Slow performance"

1. Use YOLOv8-nano: fastest for edge
2. Enable GPU: Install `onnxruntime-gpu`
3. Profile: `Profiler().record_frame()` to find bottleneck

---

## Next Steps

- 📖 **Architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md) for design deep-dive
- 📊 **Benchmarks**: See [BENCHMARKS.md](BENCHMARKS.md) for performance metrics
- 🔬 **Examples**: Check [examples/](examples/) for advanced use cases
- 🤝 **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Common Patterns

### Pattern 1: Batch Processing

```python
from pathlib import Path
from privacyguard import PrivacyGuard

guard = PrivacyGuard("model.onnx")

for image_path in Path("photos/").glob("*.jpg"):
    guard.process_image(image_path, f"photos_safe/{image_path.name}")
```

### Pattern 2: Real-time Display

```python
import cv2
from privacyguard import PrivacyGuard

guard = PrivacyGuard("model.onnx")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    result = guard.process_frame(frame)
    cv2.imshow("PrivacyGuard", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Pattern 3: Metadata Stripping

```python
from privacyguard import PrivacyGuard, MetadataStripper

guard = PrivacyGuard("model.onnx")

# Anonymize
guard.process_image("original.jpg", "temp.jpg")

# Strip metadata
MetadataStripper.strip_image("temp.jpg", "final.jpg")
```

---

**Ready to deploy? Check out [ARCHITECTURE.md](ARCHITECTURE.md) for production setup.**
