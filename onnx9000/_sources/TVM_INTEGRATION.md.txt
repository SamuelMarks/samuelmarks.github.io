---
orphan: true
---

# Apache TVM Integration (`tvm`)

ONNX9000 integrates with Apache TVM through native export paths, mapping ONNX IR AST to Relay IR for compilation targets like WebGPU, WASM, or LLVM.

## CLI Usage

```bash
onnx9000 tvm my_model.onnx
```

## Interactive Web Demo

Run `onnx9000 serve` and navigate to `/tvm`.
