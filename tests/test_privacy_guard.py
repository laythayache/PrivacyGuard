import cv2
import numpy as np
from privacyguard import PrivacyGuard

def test_blur_on_sample_image():
    model_path = "path/to/model.onnx"
    guard = PrivacyGuard(model_path)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = guard.process_frame(img)
    assert result.shape == img.shape
