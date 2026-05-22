# WebGPU-Native ONNX Support

`onnx9000` is built for modern hardware. Our **WebGPU-Native** backend (`@onnx9000/backend-web` in JS, or `onnx9000-backend-web` via Pyodide in Python) translates the `onnx9000-core` IR directly into optimized WGSL compute shaders.

> **Ecosystem Context:** As a **zero-dependency, Polyglot Monorepo**, `onnx9000` enables high-performance inference in the browser without relying on heavy C++ runtimes or official ONNX Runtime binaries.

## Supported Operators (Opset 18+)

The following subset of ONNX operators is currently optimized for WebGPU execution:

### Mathematics

- **Add, Sub, Mul, Div, Pow**: Element-wise operations with broadcast support.
- **MatMul, Gemm**: Tiled matrix multiplication optimized for GPU compute groups.
- **Neg, Exp, Log, Sqrt, Reciprocal**: Standard unary operations.

### Neural Network

- **Relu, Sigmoid, Tanh, Gelu, Silu**: Common activation functions.
- **Conv, ConvTranspose**: Highly optimized kernels for image processing.
- **MaxPool, AveragePool, GlobalAveragePool**: Spatial pooling.
- **Softmax, LogSoftmax**: Optimized for numerical stability.

### Tensor Operations

- **Reshape, Transpose, Squeeze, Unsqueeze**: Metadata-only operations (zero-copy when possible).
- **Concat, Slice, Gather, ScatterND**: Data movement kernels.
- **LayerNormalization, BatchNormalization**: Standard normalization layers.

### Control Flow

- **If, Loop, Scan**: Supported via conditional execution and iterative shader dispatch.

## Custom Shaders

`onnx9000` allows for dynamic kernel generation. If an operator is not natively supported, you can provide a custom WGSL kernel:

```typescript
import { registerKernel } from '@onnx9000/backend-web';

registerKernel('MyCustomOp', {
  wgsl: `
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      // Custom WGSL logic here
    }
  `,
});
```

## Interactive Web Demo

.. interactive-demo::
   :initial-source: script
   :initial-target: onnx
