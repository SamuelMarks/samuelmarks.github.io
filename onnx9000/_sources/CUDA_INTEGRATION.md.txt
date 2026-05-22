# CUDA Backend Integration

> **Ecosystem Context:** `onnx9000` integrates tightly with the NVIDIA CUDA ecosystem, providing multiple avenues for high-performance execution on NVIDIA GPUs, including TensorRT compilation and direct CUDA Execution Providers.

This document describes how to execute ONNX9000 graphs on NVIDIA CUDA hardware.

## Overview

For users with NVIDIA GPUs, `onnx9000` allows models to be executed natively using the CUDA Toolkit and cuDNN libraries. This provides massive acceleration for deep learning workloads.

## Usage

### Execution Provider

When running models via the Python API, simply specify the `CUDAExecutionProvider`.

```python
import onnxruntime as ort

options = ort.SessionOptions()
session = ort.InferenceSession("model.onnx", sess_options=options, providers=['CUDAExecutionProvider'])

# Run inference on NVIDIA GPU
result = session.run(None, {"input": input_data})
```

### TensorRT Compilation

For the absolute highest performance on NVIDIA hardware, compile the ONNX graph into a TensorRT engine. TensorRT performs layer fusion, precision calibration, and kernel auto-tuning specifically for your exact GPU architecture.

```bash
onnx9000 tensorrt build model.onnx -o model.trt --fp16
```

### WebGPU over Vulkan/DX12

If using the `@onnx9000/backend-web` package in Node.js or Chrome, the WebGPU abstraction layer can map calls directly down to Vulkan or DirectX 12, which are executed natively on the NVIDIA GPU driver.

```javascript
import { createSession } from '@onnx9000/backend-web';

const session = await createSession('model.onnx', {
  executionProviders: ['webgpu'],
});
```
