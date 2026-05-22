# Extended Format Converters

> **Ecosystem Context:** `onnx9000` is a **Zero-dependency, WASM-First, and WebGPU-Native Polyglot Monorepo**. Through `@onnx9000/converters`, it supports reading and transforming deep learning models from various legacy and exotic frameworks directly into the unified ONNX9000 IR.

The `@onnx9000/converters` package (and its Python equivalent `onnx9000-converters`) provides MMDNN-inspired multi-framework conversion natively. 

## Supported Frameworks

- Keras (v2 / v3)
- PyTorch (FX, TorchScript, AOTAutograd)
- JAX (ClosedJaxpr, Flax)
- TensorFlow (SavedModel, TFJS)
- Apple CoreML
- MXNet
- PaddlePaddle
- XGBoost, LightGBM, CatBoost
- Scikit-Learn
- SparkML
- H2O, LibSVM
- Caffe, CNTK, Darknet, NCNN

## CLI Usage

The universal `onnx9000 convert` tool handles translation between frameworks.

```bash
# Convert a Keras H5 model to ONNX
onnx9000 convert --src keras --dst onnx model.h5

# Convert PyTorch to CoreML
onnx9000 convert --src pytorch --dst coreml model.pt

# Batch convert a directory of Darknet weights
onnx9000 convert --src darknet --dst onnx ./yolo_weights/
```

## JavaScript / Web API Usage

For Web applications, `@onnx9000/converters` exposes the `convert` API, which works with standard Web `Blob` / `File` objects. This allows full in-browser parsing without a backend server.

```typescript
import { mmdnn } from '@onnx9000/converters';
const { convert } = mmdnn;

async function onFileUpload(files: File[]) {
  // Translate MXNet symbols/weights directly to ONNX9000 IR in the browser
  const onnxGraph = await convert('mxnet', 'onnx', files, { verbose: true });
  console.log("Converted Graph:", onnxGraph);
}
```

This translates models via the `onnx9000` Core IR before emitting the target format.
