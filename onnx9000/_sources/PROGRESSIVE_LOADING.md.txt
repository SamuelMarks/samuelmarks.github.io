# WASM-First: Progressive Model Loading

`onnx9000` is designed for the modern web. Our **WASM-First** architecture (`@onnx9000/backend-web`) supports **Progressive Loading**, allowing models to begin execution before they are fully downloaded. By leveraging HTTP `Range` requests, we can surgically fetch only the metadata and initial layers needed for a fast "Time to First Token."

> **Ecosystem Context:** As a **zero-dependency, Polyglot Monorepo**, `onnx9000` eliminates the need for heavy C++ runtimes, making progressive loading even more effective by keeping the total application footprint minimal.

## How it Works

1. **Metadata Fetch**: `@onnx9000/core` fetches the first few KB of a model (ONNX or Safetensors) to parse the graph structure and weight offsets.
2. **Layer-on-Demand**: The engine requests specific weight ranges from the server only when the execution reaches that layer.
3. **Weight Streaming**: Weights are streamed directly into WebGPU buffers, bypassing main-thread memory overhead.

## Server Configuration (Nginx)

To enable progressive loading, your server must support HTTP Range requests and CORS headers.

```nginx
server {
    listen 80;
    server_name models.onnx9000.ai;

    location / {
        root /var/www/models;

        # Enable CORS for browser access
        add_header 'Access-Control-Allow-Origin' '*';
        add_header 'Access-Control-Allow-Methods' 'GET, OPTIONS';
        add_header 'Access-Control-Allow-Headers' 'Range, Content-Type';
        add_header 'Access-Control-Expose-Headers' 'Content-Length, Content-Range, Accept-Ranges';

        # Allow multiple byte-range requests
        max_ranges 10;

        # Optional: Disable Gzip for large binary weights to ensure Range requests work correctly
        gzip off;
    }
}
```

## Client-Side Usage

```typescript
import { loadProgressive } from '@onnx9000/backend-web';

// Only fetches metadata initially
const session = await loadProgressive('https://models.onnx9000.ai/llama3-8b.onnx', {
  maxChunkSize: 1024 * 1024, // 1MB chunks
});

// Weights are fetched lazily during the first run
const result = await session.run(inputs);
```

.. interactive-demo::
   :initial-source: script
   :initial-target: onnx
