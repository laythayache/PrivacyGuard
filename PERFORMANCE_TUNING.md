# Performance Tuning and Benchmarking

PrivacyGuard does not publish a universal FPS claim. Throughput depends on the
complete configuration: hardware, runtime provider, model artifact, model input
size, source resolution and codec, masking method, number and size of detected
regions, display and output encoding, and what the measurement includes.

## Historical result and its boundary

A historical controlled internal run was described as reaching approximately
25–30 FPS. The retained repository and project history do not record enough
information to reproduce that result. In particular, they do not preserve a
signed-off record of the exact device, model file and hash, ONNX Runtime
provider, source footage, warm-up, frame count, masking configuration, output
encoding, or whether capture and display were included.

Treat 25–30 FPS only as a narrow historical implementation result. It is not a
Raspberry Pi 4 guarantee, a result for every YOLOv8-nano model, or a performance
claim for an arbitrary deployment.

## What to record

A useful benchmark report should include:

- CPU, GPU or accelerator model, RAM, operating system, and power mode;
- Python, OpenCV, ONNX Runtime, and PrivacyGuard versions;
- ONNX Runtime execution provider;
- model name, source, file hash, labels, and input size;
- source resolution, codec, frame rate, and representative footage description;
- confidence and IoU thresholds, target classes, padding, and masking method;
- warm-up frames, measured frames, number of runs, and aggregation method;
- whether timing covers inference only, masking, capture, display, and encoding;
- mean, median, p95 latency, throughput, and any dropped frames; and
- a separate detection-quality evaluation on representative footage.

FPS does not measure whether the correct regions were detected or masked.
Performance and masking quality must be evaluated separately.

## Reproducible measurement template

Use real, representative frames and freeze the configuration before reporting a
result. The following example measures detection and masking only; it excludes
camera capture, display, and output encoding.

```python
import hashlib
import platform
import statistics
import time
from pathlib import Path

import cv2
import onnxruntime as ort

from privacyguard import PrivacyGuard

model_path = Path("model.onnx")
video_path = Path("representative-input.mp4")
warmup_frames = 20
measured_frames = 300

guard = PrivacyGuard(
    model_path=model_path,
    input_size=(640, 640),
    method="pixelate",
    conf_threshold=0.4,
    iou_threshold=0.45,
    padding=0,
)

cap = cv2.VideoCapture(str(video_path))
latencies_ms = []

for index in range(warmup_frames + measured_frames):
    ok, frame = cap.read()
    if not ok:
        break

    started = time.perf_counter()
    guard.process_frame(frame)
    elapsed_ms = (time.perf_counter() - started) * 1000

    if index >= warmup_frames:
        latencies_ms.append(elapsed_ms)

cap.release()

if not latencies_ms:
    raise RuntimeError("No frames were measured")

elapsed_seconds = sum(latencies_ms) / 1000
fps = len(latencies_ms) / elapsed_seconds if elapsed_seconds else 0.0
p95_index = max(0, round(0.95 * len(latencies_ms)) - 1)

print({
    "platform": platform.platform(),
    "providers": ort.get_available_providers(),
    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    "frames": len(latencies_ms),
    "input_size": [640, 640],
    "method": "pixelate",
    "mean_ms": statistics.fmean(latencies_ms),
    "median_ms": statistics.median(latencies_ms),
    "p95_ms": sorted(latencies_ms)[p95_index],
    "fps": fps,
})
```

Run the test several times after the device reaches a steady thermal state. Do
not compare results that include different pipeline stages without labelling the
difference.

## Tuning sequence

1. Choose a model whose labels and output format match the deployment.
2. Establish detection-quality acceptance criteria on representative footage.
3. Measure a frozen baseline configuration.
4. Try a smaller model or input size and re-evaluate both speed and false
   negatives.
5. Compare masking methods with the expected number and size of regions.
6. Test the actual capture, display, and encoding path when end-to-end latency
   matters.
7. Re-run the benchmark on the deployment hardware after every material model,
   runtime, or configuration change.

Potential optimizations are configuration-dependent. A smaller input can improve
throughput but may reduce detection detail. A smaller model can reduce inference
cost but may miss difficult regions. GPU or accelerator providers can help only
when the installed runtime, model, and hardware support them.

See [BENCHMARKS.md](BENCHMARKS.md) for the shorter reporting checklist.
