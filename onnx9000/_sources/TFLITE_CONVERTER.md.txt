# TFLite Converter

> **Ecosystem Context:** `onnx9000` provides an advanced, zero-dependency `onnx2tf` converter to translate ONNX models into TFLite format for edge and mobile deployment.

This document describes how to use the TFLite exporter built into ONNX9000.

## Overview

The `onnx9000-tflite-exporter` package translates ONNX Intermediate Representation (IR) graphs directly into TensorFlow Lite FlatBuffers (`.tflite`), without needing heavy `tensorflow` or `onnx` Python packages.

## Features

- **Direct Translation:** Emits TFLite FlatBuffers directly from the ONNX9000 core IR.
- **Quantization:** Supports MIN_MAX INT8 quantization and FP16 downcasting during export.
- **Layout Conversion:** Handles NCHW (ONNX) to NHWC (TFLite) layout transformations automatically.

## Usage

### Python API

```python
from onnx9000.core.parser.core import load
from onnx9000.tflite_exporter.compiler import compile_to_tflite

# Load an ONNX model
graph = load("model.onnx")

# Compile to TFLite
tflite_buffer = compile_to_tflite(graph)

with open("model.tflite", "wb") as f:
    f.write(tflite_buffer)
```

### CLI

You can use the built-in CLI command to convert models from the terminal:

```bash
onnx9000 onnx2tf model.onnx -o model.tflite --int8
```

Options:

- `--int8`: Triggers INT8 quantization.
- `--fp16`: Triggers FP16 quantization.
- `--keep-nchw`: Keeps the NCHW format instead of optimizing for NHWC.
- `--disable-optimization`: Disables layout optimizations.
