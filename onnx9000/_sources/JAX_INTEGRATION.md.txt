# ONNX9000 JAX & Flax Integration

The `onnx9000-converters` (JS) and `onnx9000-optimizer` (Python) packages provide exhaustive integration for Google's JAX and Flax frameworks.

Unlike traditional PyTorch conversion which traces `nn.Module` execution, ONNX9000 translates JAX directly from its lowest-level Intermediate Representation: `ClosedJaxpr`.

## Key Features

- **Perfect Equivalence:** Because `ClosedJaxpr` is a purely functional, static graph of primitives, translation to ONNX is mathematically lossless.
- **Flax Module Unrolling:** Automatically extracts weight dictionaries from `flax.linen` models and binds them as static constants in the resulting ONNX graph.
- **Client-Side Translation:** Using the JS SDK, a JAX model serialized to JSON can be parsed and mapped to `ONNX9000 Core IR` entirely within the browser.

## Browser / JS Usage

When JAX is exported to a JSON representation of `ClosedJaxpr`, you can parse it directly in the frontend:

```typescript
import { parseJaxpr } from '@onnx9000/converters';

const jaxprJson = `...`;

// Translate jaxpr into an internal representation
const parsed = parseJaxpr(jaxprJson);

console.log(parsed.eqns); // Displays all JAX primitives (e.g. dot_general, exp, reduce_sum)
```

## Python CLI Usage

Convert a standard Flax/JAX checkpoint to ONNX using the Python CLI:

```bash
onnx9000 convert --from jax --model my_jax_model --out output.onnx
```
