# ONNX to Triton Operator Mapping

`onnx9000` provides a **zero-dependency** kernel generator (`@onnx9000/compiler`) that maps ONNX operators directly to OpenAI Triton (`@triton.jit`) Python source code. This allows for the generation of high-performance GPU kernels without requiring the original model framework or official C++ Protobuf bindings during the compilation phase.

> **Ecosystem Context:** As a **Polyglot Monorepo**, `onnx9000` ensures that models can be optimized and compiled into various formats. The Triton generator is a key part of our **AOT (Ahead-of-Time)** compilation strategy for server-side acceleration.

## Elementwise Operations

The following mappings are used when generating Triton kernels from `onnx9000-core` nodes:

| ONNX Op | Triton Equivalent          | Notes                                    |
| ------- | -------------------------- | ---------------------------------------- |
| Add     | `a + b`                    | Supports broadcasting                    |
| Sub     | `a - b`                    | Supports broadcasting                    |
| Mul     | `a * b`                    | Supports broadcasting                    |
| Div     | `a / (b + 1e-10)`          | Includes epsilon for numerical stability |
| Exp     | `tl.exp(x)`                |                                          |
| Log     | `tl.log(x)`                |                                          |
| Sqrt    | `tl.sqrt(x)`               |                                          |
| Relu    | `tl.maximum(x, 0.0)`       |                                          |
| Sigmoid | `1.0 / (1.0 + tl.exp(-x))` |                                          |
| Tanh    | `tl.math.tanh(x)`          |                                          |
| Pow     | `tl.math.pow(a, b)`        |                                          |

## Reductions

| ONNX Op   | Triton Equivalent      |
| --------- | ---------------------- |
| ReduceSum | `tl.sum(x, axis=0)`    |
| ReduceMax | `tl.max(x, axis=0)`    |
| ArgMax    | `tl.argmax(x, axis=0)` |

## Complex Layers

| Layer     | Implementation Strategy                                           |
| --------- | ----------------------------------------------------------------- |
| MatMul    | Tiled `tl.dot` with `BLOCK_K` accumulation loop and autotuning.   |
| LayerNorm | Two-pass reduction (mean, then variance) for hardware efficiency. |
| Softmax   | Numerically stable `max` subtraction followed by `exp` and `sum`. |

## Usage

You can generate Triton code from a graph using the JS compiler:

```typescript
import { load } from '@onnx9000/core';
import { generateTriton } from '@onnx9000/compiler';

const graph = load(modelBuffer);
const pythonCode = generateTriton(graph);
console.log(pythonCode);
```
