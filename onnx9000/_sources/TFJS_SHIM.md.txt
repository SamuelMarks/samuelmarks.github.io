---
orphan: true
---

# TensorFlow.js Shim (`tfjs-shim`)

ONNX9000 provides `@onnx9000/tfjs-shim`, a drop-in API replacement for `@tensorflow/tfjs` that seamlessly redirects standard TensorFlow.js operations to the optimized `onnx9000` WebGPU backend and memory arena manager.

This allows legacy TFJS codebases to benefit from ONNX9000's zero-malloc performance and advanced compiler passes without requiring extensive refactoring.

## Installation

Replace your existing `@tensorflow/tfjs` import with `@onnx9000/tfjs-shim`:

```bash
npm uninstall @tensorflow/tfjs
npm install @onnx9000/tfjs-shim
```

## Basic Usage

The shim mirrors the standard TFJS API. Simply update your import paths:

```typescript
// Replace: import * as tf from '@tensorflow/tfjs';
import * as tf from '@onnx9000/tfjs-shim';

// All standard operations work natively via ONNX9000's backend
tf.tidy(() => {
  const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
  const b = tf.tensor2d([5, 6, 7, 8], [2, 2]);

  const c = tf.matMul(a, b);
  const d = tf.relu(tf.sub(a, tf.scalar(2)));

  console.log(c.dataSync());
});
```

## Diagnostic Command Line Interface (CLI)

You can verify the active status of the TFJS shim within the ONNX9000 ecosystem using the provided diagnostic command:

```bash
onnx9000 tfjs-shim
```

This ensures your ecosystem paths and WebGPU contexts are correctly configured for shim routing.

## Interactive Web Demo

To interactively see the TFJS shim intercepting commands and rendering them through the ONNX9000 WebGPU engine locally:

```bash
onnx9000 serve
```

And navigate to `/tfjs-shim` in your web browser.
