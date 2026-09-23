# Benchmarks

This repository does not ship a fixed benchmark dataset, so absolute FPS numbers depend on:

- model architecture and input size
- hardware (CPU/GPU)
- anonymization method (`gaussian`, `pixelate`, `solid`)
- stream resolution and codec settings

## Historical result

A historical controlled internal run was described as reaching approximately
25–30 FPS. The retained repository does not preserve enough configuration and
measurement detail to reproduce it. Do not attribute that number to Raspberry
Pi 4, a particular model, or an arbitrary deployment.

See [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md) for the missing fields, a
measurement template, and the boundary on that historical observation.

## Reproduce locally

Use the built-in profiler to measure your own workload:

```python
import cv2
from privacyguard import PrivacyGuard
from privacyguard.profiler import Profiler

guard = PrivacyGuard("model.onnx", input_size=(640, 640), method="pixelate")
profiler = Profiler()

cap = cv2.VideoCapture("input.mp4")
profiler.start()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = guard.process_frame(frame)
    # record custom metrics if needed

cap.release()
report = profiler.stop()
print(report)
```

## Reporting guidance

When sharing results, include:

- CPU/GPU model and RAM
- Python version
- ONNX Runtime provider (`CPUExecutionProvider`, `CUDAExecutionProvider`, etc.)
- model name and input size
- anonymization method and confidence threshold
- model file hash and class mapping
- source resolution, codec, and representative footage description
- warm-up, frame count, run count, and whether capture/display/encoding are timed
- latency distribution and dropped frames, not only average FPS
- separate detection-quality results for each protected class
