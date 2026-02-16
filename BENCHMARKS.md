# Benchmarks

This repository does not ship a fixed benchmark dataset, so absolute FPS numbers depend on:

- model architecture and input size
- hardware (CPU/GPU)
- anonymization method (`gaussian`, `pixelate`, `solid`)
- stream resolution and codec settings

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
