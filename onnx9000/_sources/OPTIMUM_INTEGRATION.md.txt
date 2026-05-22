# Optimum Integration

> **Ecosystem Context:** `onnx9000` integrates with Hugging Face's Optimum library to streamline the export and quantization of Transformer models.

This guide explains how to use Optimum features via the ONNX9000 CLI and APIs.

## Overview

Hugging Face Optimum provides tools to optimize transformer models for hardware like ONNX Runtime, OpenVINO, and more. `onnx9000` wraps these capabilities, providing a cohesive interface to export models from the Hugging Face Hub directly into ONNX format with advanced quantization (like GPTQ) enabled.

## Usage

### CLI

The `onnx9000 optimum` command group provides several utilities.

**Exporting a Model**

Export a Hugging Face model directly to ONNX:

```bash
onnx9000 optimum export "distilbert-base-uncased-finetuned-sst-2-english" --task text-classification
```

**Optimizing a Model**

Optimize an existing ONNX model (e.g., fusing nodes, optimizing for size):

```bash
onnx9000 optimum optimize model.onnx --level 1 --optimize-size
```

**Quantizing a Model**

Quantize an ONNX model (e.g., using GPTQ or W4A16):

```bash
onnx9000 optimum quantize model.onnx --quantize gptq --gptq-bits 4 --gptq-group-size 128
```

### JS / TS Support

You can also leverage Optimum export tools through the Node.js API:

```typescript
import { exportModel } from '@onnx9000/optimum';

await exportModel('bert-base-uncased', {
  task: 'feature-extraction',
  opset: 14,
  device: 'cpu',
});
```
