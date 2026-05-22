# C Compiler (onnx2c)

> **Ecosystem Context:** `onnx9000` provides a Web-Native TinyML & Embedded C99 Generator based on the `onnx2c` (deepC) architecture. It allows you to compile ONNX models directly into zero-dependency C code.

This guide explains how to generate C code from ONNX models.

## Overview

The C Compiler (`onnx9000-c-compiler`) package transpiles the ONNX Intermediate Representation into static, alloc-free C99 code. This is extremely useful for edge devices, microcontrollers (TinyML), or embedded systems where deploying a heavy runtime engine like ONNX Runtime or TensorFlow Lite Micro is not feasible.

## Features

- **Zero Malloc:** Generates C code that uses statically allocated memory buffers, perfect for memory-constrained embedded devices.
- **C99 Compliant:** The output is pure C99 code without external dependencies.
- **Web-Native Execution:** The compiler itself can run within the browser or Node.js via the JS API, allowing for instant model-to-C translation in web apps.

## Usage

### CLI

You can generate C code directly from the command line:

```bash
onnx9000 export model.onnx --format c -o model.c
```

### Python API

```python
from onnx9000.core.parser.core import load
from onnx9000.c_compiler import compile_to_c

graph = load("model.onnx")
c_source, h_source = compile_to_c(graph, prefix="model_")

with open("model.c", "w") as f:
    f.write(c_source)

with open("model.h", "w") as f:
    f.write(h_source)
```

### Compiling the generated C code

The generated C code exposes a single entrypoint function (e.g. `model_run(input_array, output_array)`).

```c
#include "model.h"
#include <stdio.h>

int main() {
    float input[256] = {0.0f};
    float output[10] = {0.0f};

    // Set up inputs...
    input[0] = 1.0f;

    // Execute inference
    model_run(input, output);

    printf("Result: %f\n", output[0]);
    return 0;
}
```

You can compile this code with any standard C compiler:

```bash
gcc -std=c99 -o main main.c model.c -lm
```
