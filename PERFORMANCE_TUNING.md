# Performance Tuning Guide

PrivacyGuard is designed to run efficiently on edge devices from Raspberry Pi to modern servers. This guide helps you optimize performance for your use case.

## Understanding the Tradeoffs

Privacy de-identification involves a tradeoff between **speed**, **accuracy**, and **visual quality**. Choose the right settings for your needs.

### Input Size vs Speed vs Accuracy

The model input size is the primary determinant of speed. Larger inputs give better accuracy but process slower.

| input_size | RPi 4 FPS | x86 FPS | Accuracy | Use Case |
|------------|-----------|---------|----------|----------|
| (320, 320) | ~50 | ~180 | Low | Motion detection, counting, lowest latency |
| (416, 416) | ~35 | ~120 | Medium | Balanced (recommended for edge) |
| (640, 640) | ~25 | ~90 | High | Default, production, good accuracy |
| (1280, 1280) | ~8 | ~30 | Very High | Offline batch processing, maximum accuracy |

**Recommendation:** Start with (416, 416) for edge devices, (640, 640) for production servers.

### Anonymization Methods

Different anonymization methods have different speed/quality tradeoffs.

| Method | Speed | CPU Load | Privacy Level | Visual Quality | Use Case |
|--------|-------|----------|---------------|----------------|----------|
| **solid** | ~0.1ms/region | Minimal | Maximum | Lowest | Critical privacy, public spaces |
| **pixelate** | ~0.5ms/region | Low | High | Medium | Balanced, most use cases |
| **gaussian** | ~2-3ms/region | Higher | Medium | Highest | Preserving context, visual appeal |

**Recommendation:** Use `pixelate` as default; use `solid` for maximum privacy; use `gaussian` for aesthetics.

### Confidence Threshold Tuning

Confidence threshold controls detection sensitivity.

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| **0.3** | Low (~80%) | Very High (~99%) | Critical privacy - detect everything |
| **0.5** | Balanced (~90%) | High (~95%) | Default - good balance |
| **0.7** | High (~96%) | Medium (~80%) | Reduce false positives |
| **0.85** | Very High (~99%) | Low (~60%) | Only confident detections |

**Recommendation:** Use 0.5 for production; use 0.3 for critical privacy scenarios.

### IOU Threshold Tuning

IOU (Intersection over Union) threshold controls duplicate detection suppression.

- **0.3**: Aggressive suppression, fewer overlaps (fewer anonymizations)
- **0.45**: Default, balanced suppression
- **0.6**: Lenient suppression, more overlapping boxes kept

**Recommendation:** Keep at 0.45 default unless you see duplicate boxes.

## GPU Acceleration

GPU acceleration provides 5-10x speedup on NVIDIA GPUs.

### NVIDIA GPU Setup

```bash
# Install GPU-accelerated ONNX Runtime
pip install onnxruntime-gpu

# PrivacyGuard will automatically use GPU if available
guard = PrivacyGuard("model.onnx")
# Uses CUDA if available, falls back to CPU
```

**Expected Performance:**
- RTX 3090: ~200 FPS at 640×640
- RTX 2080: ~150 FPS at 640×640
- GTX 1080: ~120 FPS at 640×640
- CPU (Ryzen 5600X): ~90 FPS at 640×640

### Automatic Provider Selection

PrivacyGuard tries providers in this order:
1. CUDA (NVIDIA GPU)
2. CoreML (Apple Neural Engine)
3. CPU (fallback)

You can override with:
```python
guard = PrivacyGuard(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
```

## Real-Time Performance Tips

### 1. Use YOLOv8-Nano or YOLOv8-Small

- **YOLOv8n**: ~25 FPS on RPi4 (fastest, good accuracy)
- **YOLOv8s**: ~12 FPS on RPi4 (better accuracy, slower)
- **YOLOv8m**: ~4 FPS on RPi4 (much slower, diminishing returns)

**Use YOLOv8-nano for edge, consider small for servers.**

### 2. Reduce Input Size for Edge Devices

```python
# For Raspberry Pi: 416×416 gives 35 FPS
guard = PrivacyGuard("model.onnx", input_size=(416, 416))

# For servers: 640×640 gives good accuracy
guard = PrivacyGuard("model.onnx", input_size=(640, 640))
```

### 3. Use Target Classes to Skip Unnecessary Detections

```python
# Only anonymize faces, skip license plates
guard = PrivacyGuard(
    "model.onnx",
    target_classes=[0],  # 0=face, 1=license_plate
)
# ~30% speedup by skipping post-processing
```

### 4. Use Pixelate for Streaming

Gaussian blur is expensive. Use pixelate for real-time streaming:

```python
guard = PrivacyGuard("model.onnx", method="pixelate")
# ~3x faster than Gaussian, still good privacy
```

### 5. Reduce Extra Work in Batch Loops

```python
# Reuse one guard instance and keep per-frame logic minimal
guard = PrivacyGuard("model.onnx", method="pixelate")
for frame in frames:
    result = guard.process_frame(frame)
```

### 6. Adjust Padding If Needed

```python
# Reduce padding for speed (default is 0)
guard = PrivacyGuard("model.onnx", padding=0)  # No expansion

# Increase padding for coverage
guard = PrivacyGuard("model.onnx", padding=10)  # Expand by 10px (slower)
```

## Batch Processing Performance

For processing many files offline, use BatchProcessor:

```python
from privacyguard.enterprise import BatchProcessor

processor = BatchProcessor(
    "model.onnx",
    output_dir="anonymized/",
)

results = processor.process_directory("images/", pattern="*.jpg")
print(f"Processed {results['successful']} files in {results['total_time_sec']:.1f}s")
```

**Tips:**
- Use 640×640+ for batch processing (speed is less critical)
- Consider running on GPU if available
- Process multiple files sequentially

## Memory Optimization

For long-running streams, monitor memory usage:

```python
from privacyguard.enterprise import RealTimeMonitor

monitor = RealTimeMonitor()

for frame in stream:
    start = time.time()
    result = guard.process_frame(frame)
    elapsed_ms = (time.time() - start) * 1000

    monitor.record_frame(elapsed_ms, len(detections))

    if monitor.should_alert(fps_threshold=20):
        print(f"Performance degraded: {monitor.get_stats()}")
```

## Optimization Checklist

- [ ] Use YOLOv8-nano (not -small or -medium)
- [ ] Set `input_size=(416,416)` for Raspberry Pi, `(640,640)` for servers
- [ ] Set `conf_threshold=0.5` (or 0.3 for critical privacy)
- [ ] Use `target_classes` to skip unnecessary detections
- [ ] Use `method="pixelate"` for real-time (faster than Gaussian)
- [ ] Enable GPU with `onnxruntime-gpu` if available
- [ ] Monitor performance with `RealTimeMonitor`
- [ ] Profile on target hardware (RPi, server, GPU, etc.)

## Expected Performance Summary

### Raspberry Pi 4 (Default Model)
- **YOLOv8-nano at 416×416, pixelate:** ~35 FPS
- **YOLOv8-nano at 640×640, pixelate:** ~25 FPS
- **YOLOv8-nano at 640×640, gaussian:** ~12 FPS

### Desktop CPU (Ryzen 5600X)
- **YOLOv8-nano at 640×640, pixelate:** ~90 FPS
- **YOLOv8-small at 640×640, pixelate:** ~50 FPS
- **YOLOv8-small at 640×640, gaussian:** ~30 FPS

### NVIDIA RTX 3090 GPU
- **YOLOv8-nano at 1280×1280, gaussian:** ~200 FPS
- **YOLOv8-small at 1280×1280, gaussian:** ~180 FPS

## Benchmarking Your Setup

```python
import time
from privacyguard import PrivacyGuard
import cv2
import numpy as np

guard = PrivacyGuard("model.onnx", input_size=(640, 640))

# Create dummy frames
frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(100)]

start = time.time()
for frame in frames:
    guard.process_frame(frame)
elapsed = time.time() - start

fps = len(frames) / elapsed
print(f"Performance: {fps:.1f} FPS ({elapsed:.1f}s for 100 frames)")
```

## Troubleshooting Performance Issues

### GPU Not Used
```python
# Check which provider is being used
provider = guard.detector.session.get_providers()
print(provider)  # Should include CUDA if GPU installed
```

### Still Slow on GPU
- Check NVIDIA drivers: `nvidia-smi`
- Verify CUDA is installed: `pip list | grep cuda`
- Try forcing CPU for comparison

### Memory Errors
- Reduce `input_size` to (416, 416)
- Process fewer frames per batch
- Enable in-place processing

### Inconsistent FPS
- Disable thermal throttling (if safe)
- Close background applications
- Use dedicated GPU (not integrated graphics)
