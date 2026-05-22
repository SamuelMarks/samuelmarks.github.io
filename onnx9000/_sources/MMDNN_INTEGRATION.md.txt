# MMdnn Integration

> **Ecosystem Context:** `onnx9000` operates as a **Zero-dependency**, **WASM-First**, **WebGPU-Native**, **Polyglot Monorepo**. The MMdnn module provides N-to-N converter functionality between various frameworks directly in the browser and via CLI.

This guide covers the integration and usage of the MMdnn conversion tools within the ONNX9000 ecosystem.

## Overview

MMdnn provides a comprehensive, open-source, N-to-N converter and framework. Historically a heavy Python-based toolset, `onnx9000.mmdnn` reimagines this universal translator as a **client-side, browser-native conversion tool**.

### Core Features

- **ONNX as the Universal IR**: Instead of using a proprietary MMdnn IR, `onnx9000` uses standard ONNX as the absolute source of truth. Every legacy format is converted _to_ ONNX, and every export target is generated _from_ ONNX.
- **Code Generation**: Generates raw PyTorch or TensorFlow.js code from an ONNX file, allowing developers to mathematically recreate models natively in modern frameworks.
- **Client-Side Parsing**: Translates Caffe, MXNet, Keras, CoreML, and CNTK models directly in the browser or via Node.js CLI without native dependencies.

## Usage

### JavaScript / TypeScript API

The MMdnn tools are available in the `@onnx9000/converters` package:

```typescript
import { mmdnn } from '@onnx9000/converters';

// Convert a Caffe model to ONNX
const onnxGraph = await mmdnn.convert({
  sourceFormat: 'caffe',
  sourcePath: 'model.prototxt',
  weightsPath: 'model.caffemodel',
  targetFormat: 'onnx',
});

// Or convert directly to PyTorch code
const pytorchCode = await mmdnn.convert({
  sourceFormat: 'mxnet',
  sourcePath: 'model.json',
  targetFormat: 'pytorch_code',
});
```

### CLI

You can also use the `onnx9000 convert` CLI command to access MMdnn features:

```bash
onnx9000 convert model.prototxt --weights model.caffemodel --from caffe --to onnx -o output.onnx
```

## Supported Frameworks

- **Caffe**: Read-only
- **MXNet**: Read-only
- **CNTK**: Read-only
- **Keras**: Read-only (HDF5 and Keras v3)
- **Darknet**: Read-only
- **NCNN**: Read-only
- **PaddlePaddle**: Read-only
- **PyTorch**: Export (Code Generation)
- **TFJS**: Export (Code Generation)
- **ONNX**: Read & Write (Universal IR)
