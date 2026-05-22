# JSON Extraction (`json-extract`)

ONNX9000 allows you to extract the full topology, metadata, and structural graph of an ONNX file into a standard, human-readable JSON format.

This functionality is available via the CLI, the Python SDK, the JS SDK, and an interactive Web Demo.

## Command Line Interface (CLI)

Use the `json-extract` command to parse an `.onnx` model and output its JSON representation:

```bash
# Print JSON to standard output
onnx9000 json-extract my_model.onnx

# Write JSON to a file
onnx9000 json-extract my_model.onnx -o output.json
```

## Python SDK

In Python, the `onnx9000-json-extract` package provides a straightforward way to extract a JSON string representation from a loaded AST `Graph`. It automatically drops massive tensor buffers to keep the output readable and lightweight.

```python
from onnx9000.core.parser.core import load
from onnx9000.json_extract import extract_json

# 1. Load the ONNX model into a Graph
graph = load("my_model.onnx")

# 2. Extract JSON (automatically replaces raw buffers)
json_data = extract_json(graph, indent=2)

print(json_data)
```

## JavaScript/TypeScript SDK

In the browser or Node.js, the `@onnx9000/json-extract` package provides a programmatic way to serialize the loaded AST while safely handling `BigInt` values and dropping large `ArrayBuffer` items for efficiency.

```typescript
import { load } from '@onnx9000/core';
import { extractJson } from '@onnx9000/json-extract';
import * as fs from 'fs';

const arrayBuffer = fs.readFileSync('my_model.onnx').buffer;
const graph = await load(arrayBuffer);

// By default, drops buffers and replaces them with a summary like '[Buffer: 1024 bytes]'
const jsonString = extractJson(graph, {
  dropBuffers: true,
  spaces: 2,
});

console.log(jsonString);
```

### Advanced TypeScript Options

You can customize how buffers are dropped by supplying a custom `bufferReplacer`:

```typescript
import { extractJson, createOnnxJsonReplacer } from '@onnx9000/json-extract';

const jsonString = extractJson(graph, {
  dropBuffers: true,
  bufferReplacer: (val) => `<Dropped ${val.byteLength} bytes for UI brevity>`,
});
```

Or you can use `createOnnxJsonReplacer()` directly if you want to perform your own `JSON.stringify`:

```typescript
import { createOnnxJsonReplacer } from '@onnx9000/json-extract';

const replacer = createOnnxJsonReplacer({ dropBuffers: true });
const customString = JSON.stringify(graph, replacer, 4);
```

## Interactive Web Demo

To try JSON extraction locally via our web interface, run:

```bash
onnx9000 serve
```

And navigate to `/json-extract` in your web browser.
