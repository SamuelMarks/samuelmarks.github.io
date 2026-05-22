# GGUF Integration (onnx2gguf)

> **Ecosystem Context:** `onnx9000` provides tools to convert ONNX models to the GGUF (GPT-Generated Unified Format) standard used by llama.cpp and other fast inferencing engines.

This guide explains how to use the ONNX to GGUF conversion features.

## Overview

The `onnx9000-onnx2gguf` package translates ONNX graph definitions and tensor weights into the GGUF format. This is particularly useful when you have a general-purpose ONNX LLM and want to run it efficiently on CPU/GPU via llama.cpp or its web ports.

## Features

- **Direct Conversion:** Translates ONNX nodes to GGUF architecture definitions.
- **Quantization Compatibility:** Prepares weights for subsequent GGUF quantization (e.g., Q4_0, Q8_0).
- **Web-Native Integration:** Can operate in browser environments to create client-side model pipelines.

## Usage

### CLI

You can use the built-in CLI command to convert an ONNX model to GGUF:

```bash
onnx9000 onnx2gguf model.onnx -o model.gguf
```

Additional options:

- `--architecture llama`: Overrides the inferred architecture type.
- `--tokenizer tokenizer.json`: Merges a HuggingFace tokenizer into the resulting GGUF file.
- `--outtype f16`: Specifies the data format for output weights.

### Python API

```python
from onnx9000.core.parser.core import load
from onnx9000.onnx2gguf.compiler import compile_to_gguf

graph = load("model.onnx")

# Returns the raw bytes of the GGUF file
gguf_bytes = compile_to_gguf(graph)

with open("model.gguf", "wb") as f:
    f.write(gguf_bytes)
```
