# ROCm Backend Integration

> **Ecosystem Context:** `onnx9000` supports native execution on AMD GPUs via the ROCm (Radeon Open Compute) ecosystem, leveraging `onnxruntime-rocm` or our WebGPU abstraction layer over Vulkan/ROCm.

This document describes how to execute ONNX9000 graphs on ROCm hardware.

## Overview

For users with AMD GPUs (e.g., Instinct, Radeon Pro), `onnx9000` provides tight integration with the ROCm software stack. This ensures that models compiled via `onnx9000` achieve maximum throughput on AMD hardware.

## Usage

### Execution Provider

When running models via the Python API, simply specify the `ROCMExecutionProvider`.

```python
import onnxruntime as ort

options = ort.SessionOptions()
session = ort.InferenceSession("model.onnx", sess_options=options, providers=['ROCMExecutionProvider'])

# Run inference on AMD GPU
result = session.run(None, {"input": input_data})
```

### WebGPU over Vulkan/ROCm

If using the `@onnx9000/backend-web` package in Node.js, the backend can map WebGPU calls down to the local Vulkan API, which is highly optimized by ROCm on AMD systems.

```javascript
import { createSession } from '@onnx9000/backend-web';

const session = await createSession('model.onnx', {
  executionProviders: ['webgpu'],
});

// Will natively leverage ROCm -> Vulkan -> WebGPU if run in Node.js
```
