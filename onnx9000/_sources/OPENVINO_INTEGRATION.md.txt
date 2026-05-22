# OpenVINO Integration

> **Ecosystem Context:** `onnx9000` features an integrated, zero-dependency OpenVINO IR compiler, translating ONNX models to Intel's OpenVINO XML/BIN format for high-performance execution on Intel architectures.

This guide provides instructions on converting models to OpenVINO format.

## Overview

Intel's OpenVINO toolkit optimizes AI workloads across Intel hardware (CPU, GPU, VPU). The `onnx9000-openvino` package translates the ONNX9000 Intermediate Representation into the `XML` and `BIN` files required by the OpenVINO inference engine natively, bypassing the need for the heavyweight `openvino-dev` Python suite.

## Usage

### CLI

You can use the built-in CLI to export an ONNX model to OpenVINO:

```bash
onnx9000 openvino export model.onnx -o openvino_model_dir
```

Additional options:

- `--fp16`: Downcast all weights to FP16 to improve performance.
- `--shape input:[1,3,224,224]`: Override input shapes.
- `--dynamic-batch`: Sets the batch size to dynamic (`-1`).

### Python API

```python
from onnx9000.core.parser.core import load
from onnx9000.openvino.exporter import export_openvino

graph = load("model.onnx")

# Exports the model as .xml and .bin files in the target directory
export_openvino(graph, "openvino_model_dir", fp16=True)
```

## Features

- **Direct XML/BIN Generation:** Avoids intermediate steps and Python dependencies.
- **Precision Options:** Automatic downcasting to FP16.
- **Bin Packing:** Compiles weights directly into the required binary format mapping.
