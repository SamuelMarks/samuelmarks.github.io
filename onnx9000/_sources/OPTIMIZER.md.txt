---
orphan: true
---

# Graph Optimization

> **Ecosystem Context:** `onnx9000-optimizer` provides an extensible pass manager for modifying ONNX IR graphs prior to compilation or export. It forms the backbone of `onnx9000`'s zero-dependency quantization and structural graph fusion tooling.

## Usage via CLI

```bash
# Optimize with default passes
onnx9000 optimize model.onnx -o model_opt.onnx

# Run specific optimization passes
onnx9000 optimize model.onnx --passes "fuse_bn_into_conv,eliminate_deadend" -o model_opt.onnx
```

## Available Passes

- `fuse_bn_into_conv`: Fuses BatchNormalization operations into preceding Convolutions to save parameters and memory bandwidth.
- `eliminate_deadend`: Removes nodes whose outputs do not contribute to the final graph outputs.
- `constant_folding`: Evaluates constant expressions statically.

## Interactive Web Demo

.. interactive-demo::
   :initial-source: script
   :initial-target: onnx
