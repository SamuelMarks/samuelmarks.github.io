---
orphan: true
---

# Tutorial: Building a Zero-Dependency 10KB Image Classifier

> **Ecosystem Context:** `onnx9000` operates as a **Zero-dependency**, **WASM-First**, **WebGPU-Native**, **Polyglot Monorepo**. It enables ultra-lightweight inference by targeting the metal directly.

This tutorial shows how to use `onnx9000-core` (Python) and `@onnx9000/core` (JS) to compile and run a standard ResNet or MobileNet model as a 10KB standalone file that uses WebGPU directly, bypassing heavy runtime libraries.

## 1. Get an ONNX Model

Download a standard ONNX vision model from the ONNX Model Zoo or export one using `onnx9000-core`.

## 2. Compile for Zero-Dependency Inference

Use the `@onnx9000/compiler` to generate a minimal inference payload.

```bash
npx @onnx9000/compiler compile mobilenet.onnx --target-backend=@onnx9000/backend-web --optimize-level=O3
```

## 3. Run Inference with @onnx9000/core

The resulting payload is designed to be used with the `@onnx9000/core` runtime, which is a **WASM-First**, **Zero-dependency** library.

```javascript
import { Model } from '@onnx9000/core';

const model = await Model.load('mobilenet.bin');
const input = new Float32Array(224 * 224 * 3);
const output = await model.predict(input);

console.log('Top-1 Class:', output.argmax());
```

## 4. Why Zero-Dependency?

Traditional runtimes like ONNX Runtime or TensorFlow.js can be tens of megabytes. By using `onnx9000-core` and the `@onnx9000/compiler` pipeline, you get:

- **Minimal Footprint:** Only the code needed for your specific model is included.
- **WASM-First:** High-performance execution on the CPU when GPU is unavailable.
- **WebGPU-Native:** Direct access to GPU acceleration without overhead.
- **No External Dependencies:** No need for heavy Protobuf libraries or complex C++ bindings.

You now have a production-ready, ultra-lightweight image classifier!

.. interactive-demo::
   :initial-source: script
   :initial-target: onnx
