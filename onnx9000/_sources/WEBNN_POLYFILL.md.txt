# ONNX9000 WebNN API Polyfill

The WebNN API is an emerging W3C standard for hardware-accelerated machine learning on the web. However, browser support is currently limited and fragmented.

The `@onnx9000/webnn-polyfill` package provides a fully compliant WebNN API surface that transparently falls back to the native `ONNX9000 WebGPU` and `WASM` engines when native WebNN is unavailable.

## Key Features

- **W3C Compliant:** Implements the full `MLContext` and `MLGraphBuilder` specification.
- **Transparent Fallback:** Uses `navigator.ml` if available; otherwise, automatically injects the polyfill.
- **Hardware Accelerated:** Leverages ONNX9000's WebGPU kernels under the hood for maximum performance on unsupported browsers.

## Usage

Simply importing the package is enough to polyfill the environment.

```typescript
import '@onnx9000/webnn-polyfill';

async function run() {
  // navigator.ml is guaranteed to exist
  const context = await navigator.ml.createContext({ deviceType: 'gpu' });
  const builder = new MLGraphBuilder(context);

  // Define graph
  const x = builder.input('x', { dataType: 'float32', dimensions: [1, 2] });
  const w = builder.constant(
    { dataType: 'float32', dimensions: [2, 2] },
    new Float32Array([1, 2, 3, 4]),
  );
  const y = builder.matmul(x, w);

  // Compile and run
  const graph = await builder.build({ y });

  const inputs = { x: new Float32Array([1, 1]) };
  const outputs = { y: new Float32Array(2) };

  await context.compute(graph, inputs, outputs);
  console.log(outputs.y);
}
```

## Supported Operations

The polyfill maps all `MLGraphBuilder` operations directly to the equivalent ONNX nodes within the `ONNX9000 Core IR`. All mathematical primitives and ML ops defined by the WebNN spec are supported.
