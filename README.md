# PrivacyGuard

**PrivacyGuard** is a privacy-first, edge AI Python library for real-time de-identification of faces and license plates in video streams. It uses highly optimized ONNX models (e.g., YOLOv8-nano) to detect and blur sensitive regions before any data leaves the device.

## Features

- Real-time face and license plate blurring on edge devices
- Lightweight ONNX model support for low-latency inference
- Easy integration with OpenCV video pipelines

## Installation

```bash
pip install opencv-python onnxruntime numpy
# (Install your ONNX model separately)
```

## Quick Start

```python
import cv2
from privacyguard import PrivacyGuard

guard = PrivacyGuard("path/to/yolov8_nano.onnx")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    processed = guard.process_frame(frame)
    cv2.imshow("De-identified", processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## Technical Requirements

- Python 3.8+
- OpenCV
- ONNX Runtime
- Pre-trained ONNX detection model (e.g., YOLOv8-nano, trained for faces/plates)

## License

[MIT](LICENSE)
