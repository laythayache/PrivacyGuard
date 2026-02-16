# Quick Start

## Install

```bash
pip install privacyguard
```

## Real-time webcam anonymization

```python
from privacyguard import PrivacyGuard

guard = PrivacyGuard("model.onnx")
guard.run(source=0)
```

## Process a single image

```python
from privacyguard import PrivacyGuard

guard = PrivacyGuard("model.onnx", method="pixelate")
detections = guard.process_image("input.jpg", "output.jpg")
print(f"Anonymized {len(detections)} regions")
```

## Process a video file

```python
from privacyguard import PrivacyGuard

guard = PrivacyGuard("model.onnx")
guard.process_video("input.mp4", "output.mp4")
```

## CLI

```bash
privacyguard model.onnx -s input.mp4 -o output.mp4
```
