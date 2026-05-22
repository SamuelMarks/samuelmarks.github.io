# ONNX9000 Model Zoo & Safetensors

The Model Zoo integration provides robust tools for fetching, inspecting, and extracting models from the Hugging Face Hub (or compatible registries) directly into the ONNX9000 ecosystem.

Central to this is the native **Safetensors** implementation.

## Key Features

- **Zero-Copy Memory Mapping:** In Python and Node.js environments, safetensors are mmap'd directly to prevent memory duplication.
- **Progressive Streaming:** In browser environments, byte-range requests are used to fetch chunks of large weights incrementally, providing a progress-bar friendly experience.
- **No PyTorch Dependency:** Read metadata and weights directly via HTTP without requiring the `torch` or `safetensors` pip packages.
- **Python Sync:** `onnx9000-zoo` provides a native CLI and API for syncing entire catalogs.

## Web / JS Usage

```typescript
import { fetchSafetensorsHeader, loadTensors } from '@onnx9000/core';

const url = 'hf://huggingface/co/bert-base-uncased/resolve/main/model.safetensors';

// Fetch only the JSON header using HTTP Range Requests
const { headerObj, headerSize } = await fetchSafetensorsHeader(url);

console.log(Object.keys(headerObj));

// Progressively stream chunks
for await (const tensor of loadTensors(url)) {
  console.log(`Loaded ${tensor.name} with shape ${tensor.info.shape}`);
}
```

## Python CLI Usage

```bash
# Sync a model from Hugging Face
onnx9000 zoo pull HuggingFaceTB/SmolLM-135M

# Convert a downloaded model to GGUF or ONNX
onnx9000 convert --from zoo --id HuggingFaceTB/SmolLM-135M -o smollm.onnx
```
