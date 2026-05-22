# ONNX9000 Array API

The `@onnx9000/array` package provides a Web-Native, Numpy-like eager execution API that wraps the underlying ONNX9000 computation engine.

It allows you to write standard mathematical logic natively in JavaScript or Python, automatically building ONNX graphs and executing them locally via WebGPU/WASM or PyTorch/CUDA natively—without requiring a massive C++ dependency.

## Key Features

- **Eager & Lazy Execution:** Build graphs immediately, or switch to lazy mode for delayed, optimized JIT compilation.
- **NumPy-like Interface:** Drop-in replacement for standard scientific computing methods.
- **Zero-Copy Transfers:** Built heavily around `ArrayBufferViews` and web standards.
- **Polyglot:** Symmetrical Python (`import onnx9000_array as np`) and JS (`import * as np from '@onnx9000/array'`) SDKs.

## JavaScript Example

```typescript
import * as np from '@onnx9000/array';

// Eager mode execution
const a = np.array([1, 2, 3]);
const b = np.array([4, 5, 6]);

const c = np.add(a, b);
console.log(c.numpy()); // [5, 7, 9]

// Matrix multiplication
const mat1 = np.array([
  [1, 2],
  [3, 4],
]);
const mat2 = np.array([
  [5, 6],
  [7, 8],
]);
const mat3 = np.matmul(mat1, mat2);

console.log(mat3.numpy()); // [[19, 22], [43, 50]]

// Lazy / Computational Graph Mode
np.lazy_mode(true);

const lazyA = np.array([10, 20]);
const lazyB = np.array([30, 40]);
const lazyC = np.add(lazyA, lazyB);

console.log(lazyC.opType); // 'Add'
```

## Python Example

```python
import onnx9000_array as np

a = np.array([1, 2, 3], dtype="float32")
b = np.array([4, 5, 6], dtype="float32")
c = np.add(a, b)

print(c.numpy())

# Or use lazy mode
np.lazy_mode(True)
lazy_a = np.array([10, 20])
lazy_b = np.array([30, 40])
lazy_c = np.add(lazy_a, lazy_b)
```

## CLI Usage

You can directly run a script using the Array API from the CLI:

```bash
onnx9000 array my_script.py
```

To run in lazy mode:

```bash
onnx9000 array my_script.py --lazy
```
