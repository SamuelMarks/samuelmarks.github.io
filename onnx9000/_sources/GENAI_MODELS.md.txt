---
orphan: true
---

# Generative AI Models

ONNX9000 provides native, zero-dependency support for large generative AI models directly within its ecosystem. This includes state-of-the-art architectures like **Whisper** (for automatic speech recognition) and **LLaMA** (for large language model inference).

## Command Line Interface (CLI)

You can run these models locally using dedicated CLI commands provided by both the Python and JavaScript tools.

### Whisper LLM (`whisper-llm`)

Run transcription on an audio file using an ONNX-exported Whisper model:

```bash
# Print transcription to standard output
onnx9000 whisper-llm my_whisper_model.onnx my_audio.wav

# Save transcription to a file
onnx9000 whisper-llm my_whisper_model.onnx my_audio.wav -o transcript.txt
```

### LLaMA Web (`llama-web`)

Run text generation using an ONNX-exported LLaMA model:

```bash
# Print generated text to standard output
onnx9000 llama-web my_llama_model.onnx --prompt "Write a short poem about ONNX"

# Save generated text to a file
onnx9000 llama-web my_llama_model.onnx --prompt "Write a short poem about ONNX" -o output.txt
```

## Python SDK

In Python, instantiate the model classes directly from the `onnx9000.core.models` namespace:

```python
from onnx9000.core.models.whisper import Whisper
from onnx9000.core.models.llama import LLaMA

# Initialize Whisper
whisper_model = Whisper(d_model=512)

# Initialize LLaMA
llama_model = LLaMA(vocab_size=32000)
```

## JavaScript/TypeScript SDK

In the browser or Node.js, these models are identically structured and natively supported for WebAssembly/WebGPU execution:

```typescript
import { Whisper, LLaMA } from '@onnx9000/core';

// Initialize Whisper
const whisperModel = new Whisper(512);

// Initialize LLaMA
const llamaModel = new LLaMA(32000);
```

## Interactive Web Demos

For a fully interactive, in-browser execution experience, run the unified web server:

```bash
onnx9000 serve
```

- **Whisper**: Navigate to `/whisper-llm`
- **LLaMA**: Navigate to `/llama-web`
