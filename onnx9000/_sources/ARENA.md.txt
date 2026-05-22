# Memory Arena and Profiler

The ONNX9000 Profiler and Memory Arena tools allow developers to analyze memory allocations, operator bottlenecks, and memory blocks utilized during ONNX inference.

## Web Arena Demo

A standalone web user interface is available to inspect and simulate memory blocks. 

To run the demo:
```bash
cd apps/demo-arena
npm run dev
```

## Python SDK

The Python SDK includes `onnx9000_profiler` to trace memory and computational peaks.

```python
from onnx9000_profiler import Profiler

profiler = Profiler(model_path="model.onnx")
profiler.run()
print(profiler.get_peak_memory())
```

## JS SDK

The JS SDK includes `@onnx9000/profiler` for in-browser metrics.

```javascript
import { Profiler } from '@onnx9000/profiler';

const profiler = new Profiler("model.onnx");
await profiler.run();
console.log(profiler.peakMemory);
```

## CLI Usage

You can invoke the profiler directly from the CLI.

```bash
onnx9000 profiler model.onnx --show-arena
```
