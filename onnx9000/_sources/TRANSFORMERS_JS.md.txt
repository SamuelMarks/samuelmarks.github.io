# ONNX9000 Transformers.js Pipelines

The `@onnx9000/transformers` package is a high-level API designed to mirror Hugging Face's `transformers.js`.

It enables 1:1 migration for developers accustomed to Hugging Face's `pipeline()` syntax, while bringing the full weight of ONNX9000's WebGPU kernels, lazy execution, and JIT compilation to the frontend.

## Supported Pipelines

- `text-classification`
- `token-classification`
- `question-answering`
- `zero-shot-classification`
- `text-generation`
- `image-classification`
- `object-detection`
- `automatic-speech-recognition`
- _...and many more standard HF pipelines._

## Usage in the Browser (TypeScript)

```typescript
import { pipeline } from '@onnx9000/transformers';

async function run() {
  // 1. Initialize the pipeline
  // It will automatically fetch weights via the ONNX9000 Zoo API using Safetensors progressive loading
  const classifier = await pipeline(
    'text-classification',
    'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
  );

  // 2. Run inference
  const result = await classifier('I absolutely love ONNX9000!');

  console.log(result);
  // [{ label: 'POSITIVE', score: 0.999 }]
}

run();
```

## Usage via CLI

You can easily test models right from the terminal using the `onnx9000 transformers` command:

```bash
onnx9000 transformers text-classification "This is amazing!"

# Or text generation
onnx9000 transformers text-generation "Once upon a time"
```
