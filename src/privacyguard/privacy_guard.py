import cv2
import numpy as np
import onnxruntime as ort

class PrivacyGuard:
    def __init__(self, model_path, conf_threshold=0.3, blur_kernel=(35, 35)):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.blur_kernel = blur_kernel

    def preprocess(self, frame, input_shape=(640, 640)):
        img = cv2.resize(frame, input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]
        return img

    def postprocess(self, outputs, orig_shape, input_shape=(640, 640)):
        detections = []
        scale_x = orig_shape[1] / input_shape[0]
        scale_y = orig_shape[0] / input_shape[1]
        for det in outputs[0]:
            conf = det[4]
            if conf > self.conf_threshold:
                x1, y1, x2, y2 = det[:4]
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                cls = int(det[5])
                detections.append((x1, y1, x2, y2, conf, cls))
        return detections

    def blur_sensitive(self, frame, detections):
        for x1, y1, x2, y2, _, _ in detections:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            blurred = cv2.GaussianBlur(roi, self.blur_kernel, 0)
            frame[y1:y2, x1:x2] = blurred
        return frame

    def process_frame(self, frame):
        input_shape = (640, 640)
        img = self.preprocess(frame, input_shape)
        outputs = self.session.run(None, {self.input_name: img})
        detections = self.postprocess(outputs, frame.shape, input_shape)
        return self.blur_sensitive(frame, detections)
