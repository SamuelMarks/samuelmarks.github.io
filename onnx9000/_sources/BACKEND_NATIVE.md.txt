---
orphan: true
---

# Native Backend (onnx9000-backend-native)

> **Ecosystem Context:** `onnx9000` is a **Zero-dependency, WASM-First, and WebGPU-Native Polyglot Monorepo**. For local development, server-side inference, and CLI testing, `onnx9000-backend-native` provides high-performance, native execution directly in Python.

## Overview

The `onnx9000-backend-native` package contains pure-Python dispatcher and executor logic along with optimized C/C++ native extensions for the following Execution Providers (EPs):

- **CPU:** The default, reference implementation using optimized BLAS routines when available.
- **Apple Metal:** Direct bindings to Metal Performance Shaders for macOS/iOS hardware acceleration.
- **CUDA:** Bindings to NVIDIA cuDNN/cuBLAS for high-performance server execution.

## Usage

```python
from onnx9000.core import Model
from onnx9000.backends.cpu import CPUExecutionProvider

# The native backend is the default provider for Python environments
model = Model.load('model.onnx', provider=CPUExecutionProvider)

# Inference runs natively
results = model.predict(inputs)
```

For hardware acceleration, use the appropriate command-line flags or explicit provider instantiations (e.g. `onnx9000 apple model.onnx` or `onnx9000 cuda model.onnx`).
