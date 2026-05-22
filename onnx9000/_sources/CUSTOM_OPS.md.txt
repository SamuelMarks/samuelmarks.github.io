# Custom Operations (Custom Ops)

ONNX9000 allows for extending standard ONNX runtime capabilities with Custom Operations. This allows you to implement missing or specialized mathematical functions directly in Python or JavaScript.

## Web Custom Ops Demo

A standalone web UI is available to simulate registering a custom operation.
```bash
cd apps/demo-custom-ops
npm run dev
```

## Python SDK

```python
from onnx9000_custom_ops import registry

def my_activation():
    return "activated!"

registry.register("MyActivation", my_activation)
print(registry.list_ops())
```

## JS SDK

```javascript
import { registry } from '@onnx9000/custom-ops';

registry.register('MyActivation', () => 'activated!');
console.log(registry.listOps());
```

## CLI Usage

```bash
onnx9000 custom-ops my_ops_definitions.json
```
