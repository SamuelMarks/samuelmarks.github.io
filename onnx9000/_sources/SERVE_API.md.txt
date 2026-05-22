# ONNX9000 Serverless Edge Serving

The `@onnx9000/serve` package provides a high-performance, lightweight inference server specifically designed for serverless environments (Bun, Cloudflare Workers, Deno) and web browsers. It acts as a drop-in replacement for NVIDIA Triton Inference Server and the OpenAI API.

## Key Features

- **KServe / Triton Protocol:** Fully implements the standard V2 Inference Protocol.
- **OpenAI Compatible:** Offers an `/v1/chat/completions` endpoint for text-generation models.
- **Zero-Dependency Edge Routing:** Native `Request` -> `Response` fetch handlers.
- **Dynamic Batching:** Includes robust streaming batching pipelines.

## Browser / Serverless Usage

Because the server is implemented purely in TypeScript and uses standard web `Request` and `Response` objects, it can run entirely inside a browser service worker or a Cloudflare worker.

```typescript
import { createServer } from '@onnx9000/serve';

const server = createServer();

// In Cloudflare Workers or Bun:
export default {
  fetch(req: Request) {
    return server.fetch(req);
  },
};
```

## CLI Usage

You can launch the edge server directly from the command line, exposing it over local HTTP:

```bash
# Start the local visualizer and API server on port 8080
onnx9000 serve

# Or point it directly to a model
onnx9000 serve my_model.onnx
```

Once running, you can send requests to `http://localhost:8080/v2/models/my_model/infer` just as you would with Triton Server.
