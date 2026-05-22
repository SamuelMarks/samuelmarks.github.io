# Apple Metal Backend Integration

> **Ecosystem Context:** `onnx9000` integrates seamlessly with Apple Silicon (M1, M2, M3, M4) by compiling directly to CoreML or utilizing the Metal Performance Shaders (MPS) via WebGPU.

This document describes how to execute ONNX9000 graphs on Apple Metal hardware.

## Overview

For users on macOS or iOS devices, the highest performance is achieved by utilizing the Metal API. `onnx9000` offers two primary paths to Metal execution:

1. **CoreML Compilation**: AOT compilation to `.mlmodel` or `.mlpackage`.
2. **WebGPU to Metal**: Direct WebGPU API calls mapped to Metal in modern browsers like Safari or via Node.js.

## Usage

### CoreML Export (AOT)

Use the built-in compiler to translate an ONNX model into Apple's native CoreML format.

```bash
onnx9000 coreml export model.onnx -o model.mlpackage
```

### WebGPU / Metal (JIT)

When running in the browser (Safari) or Node.js, request the `webgpu` execution provider. The underlying engine will map computational shaders directly to Apple's Metal API.

```javascript
import { createSession } from '@onnx9000/backend-web';

const session = await createSession('model.onnx', {
  executionProviders: ['webgpu'], // Automatically uses Metal on macOS/iOS
});
```

### MPS in PyTorch

If you used the `onnx9000` PyTorch code generation tools, the resulting PyTorch code is natively compatible with the `mps` device.

```python
import torch
from model_generated import ONNXModel

device = torch.device("mps")
model = ONNXModel().to(device)

result = model(input_data.to(device))
```
