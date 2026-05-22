# ONNX Checker

> **Ecosystem Context:** `onnx9000` provides a 100% Pure TS/Python Web-Native Schema Validator to check the correctness of ONNX models without needing the official C++ bindings.

This guide explains how to use the ONNX Checker within the ecosystem.

## Overview

The ONNX Checker ensures that an ONNX model adheres to the official ONNX schema. Historically, this required the `onnx` Python package, which relies on a compiled C++ backend. `onnx9000` implements a zero-dependency, isomorphic checker that works across Python, Node.js, and the Browser.

## Features

- **Schema Validation**: Validates nodes against standard ONNX operator schemas.
- **Type Checking**: Ensures inputs and outputs match the expected types.
- **Shape Inference Validation**: Checks that shapes propagate correctly and legally.
- **Web-Native**: Runs completely in the browser via TypeScript or in Python via a pure implementation.

## Usage

### JavaScript / TypeScript

```typescript
import { check_model } from '@onnx9000/core/checker';
import { load } from '@onnx9000/core/parser';

const graph = await load('model.onnx');
const isValid = check_model(graph);

if (isValid) {
  console.log('Model is valid according to ONNX schema!');
}
```

### Python API

```python
from onnx9000.core.parser.core import load
from onnx9000.core.checker import check_model

graph = load("model.onnx")
check_model(graph) # Raises ValidationError if invalid
```

### UI Integration

The checker is natively integrated into the local Visual Modifier UI (`onnx9000 edit`) and provides real-time schema validation when editing models visually.
