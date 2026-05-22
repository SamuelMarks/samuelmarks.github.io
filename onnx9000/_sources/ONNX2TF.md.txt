---
orphan: true
---

# ONNX to TensorFlow Lite (ONNX2TF)

> **Ecosystem Context:** `onnx9000` includes a robust `onnx9000-tflite-exporter` package and `onnx2tf` CLI that converts ONNX models to TFLite format seamlessly, making them ready for Edge TPU, Android NNAPI, and iOS CoreML inference.

## Usage via CLI

```bash
# Convert basic
onnx9000 onnx2tf model.onnx -o model.tflite

# Enable INT8 Quantization
onnx9000 onnx2tf model.onnx --int8 -o quantized.tflite
```

## Interactive Web Demo

.. interactive-demo::
   :initial-source: script
   :initial-target: onnx
