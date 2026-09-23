# Quick Start

## Install from source

```bash
git clone https://github.com/laythayache/privacyguard.git
cd privacyguard
pip install -e .
```

PrivacyGuard is not currently published as a verified package on PyPI.

## Live webcam masking

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
print(f"Masked {len(detections)} detected regions")
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
