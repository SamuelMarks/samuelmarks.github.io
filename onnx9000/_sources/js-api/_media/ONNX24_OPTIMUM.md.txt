# ONNX21: HuggingFace Optimum (Web-Optimized Export & Quantization)

## Original Project Description

HuggingFace Optimum is an extension of the `transformers` library designed to bridge the gap between high-level model code and hardware-accelerated execution backends. It provides dedicated tools to export PyTorch/TensorFlow models to ONNX (via `optimum-cli`), apply hardware-specific graph optimizations, and perform advanced quantization (Dynamic Int8, Static Int8, GPTQ, AWQ) to maximize inference speed and minimize memory footprints. Optimum typically targets backend SDKs like ONNX Runtime, Intel OpenVINO, Nvidia TensorRT, and Habana Gaudi.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

Instead of catering to heavy, server-centric C++ hardware SDKs (like TensorRT or OpenVINO), `onnx9000.optimum` acts as the definitive build, optimization, and quantization toolchain targeting **WebAssembly, WebGPU, and WebNN**.

- **Web-Centric Quantization:** Focuses heavily on W4A16 (4-bit weights, 16-bit activations) and sub-byte packing tailored specifically for WebGPU storage buffers and WASM SIMD, prioritizing payload size reduction over pure theoretical FLOPs.
- **Universal Tooling:** Replaces the Python-only `optimum` toolchain with a universal framework. Developers can export, optimize, and quantize models using pure Node.js or even directly within the browser, avoiding massive PyTorch environments when simply converting an existing model for web delivery.
- **Integrated Optimization:** Standard HF Optimum relies on external ONNX Runtime tools for optimization. `onnx9000` applies these graph mutations internally via its own AST/Graph rewriting engine, emitting highly pruned web-ready payloads.
- **Web Inference Wrappers:** Provides equivalent `ORTModelForX` wrapper classes that are intimately aware of WebGPU memory management and WASM threading.

---

## Exhaustive Implementation Checklist

### Phase 1: Exporter CLI & Core Architectures (`optimum-cli`)

- [x] 1.  Implement `onnx9000 optimum` base CLI command structure.
- [x] 2.  Implement `onnx9000 optimum export` sub-command.
- [x] 3.  Implement `onnx9000 optimum optimize` sub-command.
- [x] 4.  Implement `onnx9000 optimum quantize` sub-command.
- [x] 5.  Support `--model <model_id>` fetching from HuggingFace Hub.
- [x] 6.  Support `--task <task>` flag for explicit export paths.
- [x] 7.  Auto-detect task from `config.json` if `--task` is omitted.
- [x] 8.  Support `--opset <version>` flag for specific ONNX opset targeting.
- [x] 9.  Implement `--device` flag targeting `cpu`, `wasm`, `webgpu`, `webnn`.
- [x] 10. Support `--cache_dir` for downloading HuggingFace weights.
- [x] 11. Support `--monolith` vs `--external-data` flag for weight storage.
- [x] 12. Implement `--atol` and `--rtol` flags for post-export validation.
- [x] 13. Parse specific `transformers` model architectures dynamically.
- [x] 14. Support export of `past_key_values` inputs/outputs automatically.
- [x] 15. Handle `use_cache=True` configuration during export tracing.
- [x] 16. Support creating dummy inputs for ONNX JIT tracing.
- [x] 17. Support dynamic axes declaration during export mapping.
- [x] 18. Handle multiple graph outputs automatically mapping to dictionary keys.
- [x] 19. Warn users on unsupported PyTorch ops with fallback suggestions.
- [x] 20. Implement export progress bars (Tqdm equivalent in Python/JS).
- [x] 21. Provide Node.js equivalent API: `import { exportModel } from 'onnx9000/optimum'`.
- [x] 22. Save resulting `model.onnx` alongside `config.json` and `tokenizer.json`.
- [x] 23. Generate a `generation_config.json` on export for GenAI models.
- [x] 24. Extract preprocessor configs (e.g., `preprocessor_config.json`) during export.
- [x] 25. Support `--split` flag to partition massive graphs (e.g., separating Encoder/Decoder).

### Phase 2: Base Graph Optimizations (O1 & O2 Levels)

- [x] 26. Implement Optimization Level 1 (O1): Basic Graph Topology Optimization.
- [x] 27. Implement constant folding across the entire graph.
- [x] 28. Implement redundant node elimination (e.g., double transposes).
- [x] 29. Implement Cast insertion/removal for mixed precision graph cleanup.
- [x] 30. Implement Identity node removal.
- [x] 31. Fuse `MatMul` + `Add` into `Gemm` operation.
- [x] 32. Fuse `Conv` + `BatchNormalization`.
- [x] 33. Fuse `Conv` + `Add` + `Relu`.
- [x] 34. Implement reshape/transpose propagation.
- [x] 35. Implement Optimization Level 2 (O2): Extended Fusions.
- [x] 36. Fuse `LayerNormalization` from underlying Add/ReduceMean/Sub/Pow/Div ops.
- [x] 37. Fuse `Gelu` from Erf/Add/Mul/Div ops.
- [x] 38. Fuse `FastGelu` from Tanh approximation ops.
- [x] 39. Fuse `SkipLayerNormalization` (Add + LayerNorm).
- [x] 40. Fuse `Attention` mechanisms (standard Multi-Head Attention).
- [x] 41. Handle masked attention fusions.
- [x] 42. Identify and fuse `RotaryEmbedding` (RoPE) subgraph structures.
- [x] 43. Support overriding optimization behavior via `--disable-fusion` flags.
- [x] 44. Track FLOPs reduction and report optimization statistics post-build.
- [x] 45. Ensure O1/O2 passes maintain strict floating-point parity with raw export.
- [x] 46. Eliminate dead initializer memory from the model proto.
- [x] 47. Perform ONNX model shape inference statically before saving.
- [x] 48. Deduplicate identical initializers (weights) referenced multiple times.
- [x] 49. Strip `doc_string` metadata from ONNX nodes to reduce file size.
- [x] 50. Strip debug tensor names based on an `--optimize-size` flag.

### Phase 3: Web-Native Advanced Optimizations (O3 & O4 Levels)

- [x] 51. Implement Optimization Level 3 (O3): Hardware-aware layout transformations.
- [x] 52. Perform NCHW to NHWC layout conversions explicitly for WebGPU targeting.
- [x] 53. Apply specific SIMD padding alignment for WebAssembly memory boundaries.
- [x] 54. Fuse grouped Query/Key/Value projections into unified linear layers.
- [x] 55. Fuse SwiGLU activations (commonly used in Llama/Mistral).
- [x] 56. Fuse GeGLU activations.
- [x] 57. Replace standard Softmax with numerically stable/fast approximations for WASM.
- [x] 58. Implement Optimization Level 4 (O4): Precision mapping and Web Mixed-Precision.
- [x] 59. Cast entire graph weights to FP16 (`--fp16`).
- [x] 60. Exclude `LayerNorm` and `Softmax` from FP16 casting to prevent overflow/NaNs.
- [x] 61. Provide a `webgpu_strict` graph pass (replacing WebGPU unsupported ops).
- [x] 62. Implement custom `onnx9000.DynamicQuantizeLinear` fusion for smaller payload.
- [x] 63. Support rewriting FlashAttention-like nodes natively recognized by `onnx9000` web runtimes.
- [x] 64. Optimize model subgraph partitioning specifically for asynchronous WebGPU passes.
- [x] 65. Inject explicit Web Worker memory boundaries into the graph metadata.
- [x] 66. Build an interactive HTML report of the optimized graph vs original graph.
- [x] 67. Support `--disable-gelu-fusion` for legacy browser support.
- [x] 68. Perform static allocation planning and save arena layouts into model metadata.
- [x] 69. Replace `Gather` operations with specific dictionary lookups where weights are constant.
- [x] 70. Generate a topological execution schedule as a JSON sidecar file.

### Phase 4: Basic Quantization Engine (Int8 / FP16)

- [x] 71. Implement Dynamic Int8 Quantization core engine.
- [x] 72. Support dynamic quantization for `MatMul` nodes.
- [x] 73. Support dynamic quantization for `Attention` nodes.
- [x] 74. Implement asymmetric Int8 quantization (Zero-point + Scale).
- [x] 75. Implement symmetric Int8 quantization (Scale only, Zero-point = 0).
- [x] 76. Implement MinMax quantization calibration algorithm.
- [x] 77. Implement Entropy (KL-Divergence) calibration algorithm.
- [x] 78. Implement Percentile calibration algorithm.
- [x] 79. Support Per-Tensor quantization configuration.
- [x] 80. Support Per-Channel quantization configuration.
- [x] 81. Add Python API: `Quantizer.quantize(model, config)`.
- [x] 82. Add Node.js API: `quantizer.quantize(model, config)`.
- [x] 83. Support `ORTConfig` mapping for backwards compatibility with HF Optimum.
- [x] 84. Implement Static Int8 Quantization engine.
- [x] 85. Provide APIs to ingest calibration datasets for static quantization.
- [x] 86. Expose `--quantize dynamic` in the CLI.
- [x] 87. Expose `--quantize static` in the CLI.
- [x] 88. Prevent quantization of embedding layers to maintain output quality.
- [x] 89. Allow selective node exclusion from quantization via regex/node name.
- [x] 90. Convert specific nodes directly to Int8, emitting `QuantizeLinear` and `DequantizeLinear`.

### Phase 5: Advanced Web-Quantization (GPTQ, AWQ, W4A16)

- [x] 91. Implement **GPTQ** (Accurate Post-Training Quantization) algorithm.
- [x] 92. Compute Hessian matrices over calibration data for GPTQ.
- [x] 93. Perform greedy/Cholesky inverse weight updates for GPTQ.
- [x] 94. Support `--gptq-bits` parameter (e.g., 4, 3, 2).
- [x] 95. Support `--gptq-group-size` (e.g., 32, 64, 128).
- [x] 96. Implement **AWQ** (Activation-aware Weight Quantization) algorithm.
- [x] 97. Scale salient weights dynamically based on activation distributions.
- [x] 98. Implement **SmoothQuant** algorithm.
- [x] 99. Perform activation smoothing (migrating difficulty from activations to weights).
- [x] 100.  Implement W4A16 (4-bit weights, 16-bit activations) packing engine.
- [x] 101.  Pack two 4-bit weights into a single UInt8 initializer (crucial for web payloads).
- [x] 102.  Pack eight 4-bit weights into a single UInt32 initializer (optimal for WebGPU buffers).
- [x] 103.  Emit specialized `onnx9000.Dequantize4Bit` nodes.
- [x] 104.  Implement Block-wise quantization structures to prevent accuracy loss.
- [x] 105.  Support storing quantization scales and zero-points in separate packed tensors.
- [x] 106.  Handle custom grouping strategies in WASM quantization.
- [x] 107.  Integrate with `safetensors` to stream quantized weights efficiently.
- [x] 108.  Support automatic fallback to FP16 if a specific layer degrades too much during 4-bit quant.
- [x] 109.  Support INT4 calibration in the Node.js/Browser environment.
- [x] 110.  Expose an API to evaluate perplexity degradation post-quantization.

### Phase 6: Calibration & Data Processing

- [x] 111.  Define base `CalibrationDataReader` interface.
- [x] 112.  Implement dataset loaders for text datasets (WikiText, C4).
- [x] 113.  Implement dataset loaders for image datasets (ImageNet miniset).
- [x] 114.  Connect to HuggingFace `datasets` library for fetching calibration data.
- [x] 115.  Expose specific formatting functions to map raw datasets to ONNX inputs.
- [x] 116.  Support caching calibration intermediate activations to disk to save memory.
- [x] 117.  Implement random subsetting of datasets for faster calibration.
- [x] 118.  Handle variable sequence lengths during calibration (padding/truncation).
- [x] 119.  Export calibration data to `.pb` or JSON format for cross-platform debugging.
- [x] 120.  Provide a built-in dummy data generator for "blind" quantization (when no data is available).
- [x] 121.  Support multi-modal calibration data (Image + Text paired batches).
- [x] 122.  Implement calibration metrics tracking (MSE, Cosine Similarity of weights).
- [x] 123.  Allow user-defined evaluation hooks to monitor metric drops step-by-step.
- [x] 124.  Build an interactive progress monitor during the lengthy GPTQ/AWQ process.
- [x] 125.  Implement early stopping in calibration if degradation threshold is exceeded.

### Phase 7: Specialized Task Exporters (NLP architectures)

- [x] 126.  Create custom ONNX config mapping for **BERT** architecture.
- [x] 127.  Create custom ONNX config mapping for **RoBERTa** architecture.
- [x] 128.  Create custom ONNX config mapping for **DistilBERT** architecture.
- [x] 129.  Create custom ONNX config mapping for **T5** architecture (Encoder/Decoder split).
- [x] 130.  Create custom ONNX config mapping for **BART** architecture.
- [x] 131.  Create custom ONNX config mapping for **GPT-2** architecture.
- [x] 132.  Create custom ONNX config mapping for **LLaMA** architecture (1, 2, and 3).
- [x] 133.  Create custom ONNX config mapping for **Mistral** architecture.
- [x] 134.  Create custom ONNX config mapping for **Gemma** architecture.
- [x] 135.  Create custom ONNX config mapping for **Phi** architecture (1.5, 2, 3).
- [x] 136.  Create custom ONNX config mapping for **Qwen** architecture.
- [x] 137.  Create custom ONNX config mapping for **LlamaVision** (Multimodal LLM).
- [x] 138.  Ensure `past_key_values` dynamically resolve `num_attention_heads` from HF config.
- [x] 139.  Automate extraction of `eos_token_id` and `pad_token_id` into the ONNX graph metadata.
- [x] 140.  Support exporting models with custom `rotary_dim` sizes.
- [x] 141.  Ensure sliding window attention parameters are successfully encoded during Mistral export.
- [x] 142.  Support Mixture of Experts (MoE) topologies (Mixtral) export mappings.
- [x] 143.  Map `GatedCrossEntropyLoss` or other specialized MoE outputs if requested.
- [x] 144.  Handle dynamic position IDs generation internally if not provided by the input.
- [x] 145.  Automatically fix missing dummy inputs for complex custom NLP topologies.

### Phase 8: Specialized Task Exporters (Vision, Audio, Multimodal)

- [x] 146.  Create custom ONNX config mapping for **ViT** (Vision Transformer).
- [x] 147.  Create custom ONNX config mapping for **CLIP** (Text and Image Encoders split).
- [x] 148.  Create custom ONNX config mapping for **DETR** (Object Detection).
- [x] 149.  Create custom ONNX config mapping for **YOLOS**.
- [x] 150.  Create custom ONNX config mapping for **Stable Diffusion** (UNet).
- [x] 151.  Create custom ONNX config mapping for **Stable Diffusion** (VAE Encoder/Decoder).
- [x] 152.  Create custom ONNX config mapping for **Stable Diffusion** (Text Encoder).
- [x] 153.  Create custom ONNX config mapping for **Whisper** (Encoder/Decoder split).
- [x] 154.  Create custom ONNX config mapping for **Wav2Vec2**.
- [x] 155.  Create custom ONNX config mapping for **SpeechT5**.
- [x] 156.  Handle sequence-length scaling factors dynamically in Whisper export.
- [x] 157.  Export image preprocessing normalization constants strictly into ONNX graph initializers.
- [x] 158.  Resolve dynamic height/width parameters for CNN-based vision architectures.
- [x] 159.  Support specific feature extractors configuration serialization.
- [x] 160.  Create mapping rules for 3D convolution networks (Video processing).
- [x] 161.  Handle complex tuple-based return types from vision transformers.
- [x] 162.  Map attention masks for audio spectrogram inputs securely.
- [x] 163.  Map raw waveform inputs dynamically scaling `chunk_size`.
- [x] 164.  Implement specific graph optimizations to fuse vision PatchEmbedding layers natively.
- [x] 165.  Add warnings for audio models if exported without caching mechanisms.

### Phase 9: Model Web-Inference Wrappers (`ORTModelForX`)

- [x] 166.  Implement base `ORTModel` wrapper for browser execution environments.
- [x] 167.  Implement `ORTModelForSequenceClassification`.
- [x] 168.  Implement `ORTModelForTokenClassification`.
- [x] 169.  Implement `ORTModelForQuestionAnswering`.
- [x] 170.  Implement `ORTModelForCausalLM` (Integrated heavily with ONNX19 GenAI APIs).
- [x] 171.  Implement `ORTModelForMaskedLM`.
- [x] 172.  Implement `ORTModelForSeq2SeqLM`.
- [x] 173.  Implement `ORTModelForImageClassification`.
- [x] 174.  Implement `ORTModelForObjectDetection`.
- [x] 175.  Implement `ORTModelForSpeechSeq2Seq`.
- [x] 176.  Implement `ORTModelForSemanticSegmentation`.
- [x] 177.  Provide `from_pretrained()` loading directly from `.onnx` files or Hub URLs.
- [x] 178.  Integrate configuration parsing seamlessly inside `from_pretrained`.
- [x] 179.  Support asynchronous `await ORTModelForCausalLM.from_pretrained(...)`.
- [x] 180.  Pass specific `onnx9000` session configuration parameters transparently.
- [x] 181.  Ensure inputs strictly match ONNX expected types (auto-casting standard JS arrays to Float32Array).
- [x] 182.  Implement generation wrapper passing arguments correctly to the KV Cache state engine.
- [x] 183.  Map standard `transformers` output dataclasses (e.g., `CausalLMOutputWithPast`).
- [x] 184.  Support retrieving hidden states if `--output_hidden_states` was flagged during export.
- [x] 185.  Support retrieving attentions if `--output_attentions` was flagged during export.

### Phase 10: Web-Native Optimization Extensions (BetterTransformer equivalent)

- [x] 186.  Port "BetterTransformer" concept to WebAssembly/WebGPU fast paths.
- [x] 187.  Implement AST pass: Replace PyTorch native `nn.MultiheadAttention` with optimized `onnx9000.FlashAttention`.
- [x] 188.  Support sparsity-aware execution routing in models (executing only non-zero blocks if identified).
- [x] 189.  Strip dropout layers permanently from the exported graph to speed up inference.
- [x] 190.  Implement specific graph rewrites to utilize WebGPU subgroup operations (when available).
- [x] 191.  Apply constant folding recursively until graph size stabilizes.
- [x] 192.  Replace `Pow(x, 2)` with `Mul(x, x)` automatically to save WGSL shader instructions.
- [x] 193.  Analyze memory lifecycle graphs to pre-allocate minimum VRAM boundaries for WebGPU.
- [x] 194.  Handle explicit `int64` downcasting to `int32` globally, as WebGPU natively lacks `int64` support.
- [x] 195.  Implement sub-byte unpacking WGSL shaders tightly bound to the W4A16 nodes.
- [x] 196.  Detect sequence-length limitations statically and throw early web warnings.
- [x] 197.  Add `--web-safe` CLI flag to ensure 100% strict compliance with base WebGL/WebGPU specs.
- [x] 198.  Support generating separate WASM vs WebGPU optimized ONNX binaries in a single CLI run.
- [x] 199.  Compile static shape variations of a model if dynamic shapes cause massive overhead on some GPUs.
- [x] 200.  Minify the ONNX graph structure by renaming long internal node names to short alphabetic identifiers.

### Phase 11: Export Tooling Validation & Parity

- [x] 201.  Create a validation suite comparing PyTorch outputs vs ONNX exported outputs.
- [x] 202.  Measure max absolute error (MAE) across all output tensors post-export.
- [x] 203.  Measure cosine similarity across all output tensors post-export.
- [x] 204.  Validate O1/O2 optimizations do not drop cosine similarity below 0.999.
- [x] 205.  Validate INT8 dynamic quantization keeps cosine similarity > 0.95.
- [x] 206.  Validate INT4 (W4A16) quantization keeps cosine similarity > 0.90.
- [x] 207.  Run automated integration tests exporting 50+ HuggingFace popular models.
- [x] 208.  Implement a specific check ensuring `past_key_values` are functionally identical across loops.
- [x] 209.  Export models with mixed precisions and validate boundary casts.
- [x] 210.  Provide an HTML-based export summary report (showing layer-by-layer size reduction).
- [x] 211.  Establish automated benchmarking suite tracking ONNX binary size over time.
- [x] 212.  Ensure memory usage during the export process itself stays below 8GB limits for standard CI runners.
- [x] 213.  Expose debug flags (`--debug-nodes`) to pinpoint exactly which layer loses precision during quantization.
- [x] 214.  Create automated fixes for common ONNX exporter bugs in native PyTorch versions.
- [x] 215.  Validate correct exporting of complex control flow structures (If/Loop) if present.

### Phase 12: HuggingFace Hub Integration & Publishing

- [x] 216.  Implement `onnx9000 optimum push_to_hub` CLI.
- [x] 217.  Handle chunked file uploads for ONNX files > 2GB.
- [x] 218.  Generate appropriate `README.md` model cards tagging `onnx9000` and `webgpu`.
- [x] 219.  Ensure `safetensors` format is preferred over `onnx_data` external binaries when pushing to Hub.
- [x] 220.  Maintain repository metadata linking back to the original PyTorch model.
- [x] 221.  Implement fetching/saving API Tokens from the local environment (`HF_TOKEN`).
- [x] 222.  Parse branch and PR structures directly from the CLI.
- [x] 223.  Validate that generated models pass HF's security scanners natively.
- [x] 224.  Bundle optimization metadata into `optimum_config.json` inside the repository.
- [x] 225.  Expose an API to check if a model repository already contains a web-optimized ONNX variant.

### Phase 13: Specialized Node.js Export Tooling (Browser Context)

- [x] 226.  Ensure the graph optimization engine (AST rewriting) is 100% written in TS/JS.
- [x] 227.  Enable a user to upload a `.onnx` file in a browser, optimize it, and download it locally.
- [x] 228.  Provide a Web Worker wrapper for heavy JS-based optimization passes.
- [x] 229.  Expose dynamic quantization directly in JS (quantizing a model purely on the client side).
- [x] 230.  Manage ArrayBuffer lifecycle carefully in JS to prevent memory leaks during massive graph rewrites.
- [x] 231.  Use IndexedDB to stage large model files during browser-based export.
- [x] 232.  Display visual graph pruning statistics in a UI component.
- [x] 233.  Enable JS-based model slicing (e.g., extracting just the Text Encoder from a full pipeline).
- [x] 234.  Parse and edit ONNX protobuf structures natively in TS without reliance on Python Protobuf compilers.
- [x] 235.  Provide simple APIs: `const optimizedBlob = await optimize(onnxBlob, { level: 'O3' })`.

### Phase 14: LoRA and Adapters Integration

- [x] 236.  Support exporting PyTorch models with fused PEFT/LoRA adapters.
- [x] 237.  Implement logic to extract LoRA weights as a standalone `.onnx_adapter` file.
- [x] 238.  Optimize base model to support dynamic injection of exported LoRA weights.
- [x] 239.  Ensure quantization engine correctly handles models with injected LoRAs.
- [x] 240.  Validate generation equivalence when applying LoRAs natively vs dynamically in `onnx9000`.
- [x] 241.  Provide CLI support: `onnx9000 optimum export --model <base> --lora <lora_id>`.
- [x] 242.  Build specialized WebGPU kernels for fast LoRA addition `(W + A*B)*x`.
- [x] 243.  Implement support for multiple active LoRAs during web-inference.
- [x] 244.  Optimize loading speed of small adapter files.
- [x] 245.  Validate LoRA rank scaling factors are correctly serialized.

### Phase 15: Telemetry, Logs, and Error Handling

- [x] 246.  Implement highly descriptive parsing errors if the user's HF model structure is unrecognized.
- [x] 247.  Produce a comprehensive debug log (`onnx9000_export.log`) tracking every graph mutation.
- [x] 248.  Provide clear "How to fix" suggestions when encountering unsupported PyTorch dynamic control flows.
- [x] 249.  Integrate `pino` or standard Python `logging` to standardize output formats.
- [x] 250.  Warn explicitly when the user exports an fp32 model and targets WASM (suggesting quantization).
- [x] 251.  Catch WebGPU OOM errors during O3/O4 simulation and warn the user before deployment.
- [x] 252.  Handle graceful interrupts (Ctrl+C) cleaning up temporary heavy export directories.
- [x] 253.  Prevent overwriting existing model folders without `--overwrite` confirmation.
- [x] 254.  Support tracing memory allocations during the JIT export to identify bloated operators.
- [x] 255.  Wrap obscure ONNX protobuf parse errors into human-readable TS exceptions.

### Phase 16: Security & System Integration

- [x] 256.  Strictly sanitize any custom python code present in Hub `trust_remote_code=True` instances.
- [x] 257.  Verify checksums of downloaded HuggingFace models prior to executing export logic.
- [x] 258.  Ensure the exported `.onnx` does not embed local file paths or PII from the export machine.
- [x] 259.  Strip user environment metadata from ONNX `producer_name` or `doc_string` fields.
- [x] 260.  Implement integration tests to run the CLI successfully within isolated Docker containers.
- [x] 261.  Release Node.js CLI to NPM (`npm i -g @onnx9000/optimum`).
- [x] 262.  Release Python CLI to PyPI (`pip install onnx9000-optimum`).
- [x] 263.  Add GitHub actions automating model optimization on pull requests (e.g., checking size reduction).
- [x] 264.  Support configuration files (`optimum.yaml`) to standardize export recipes for organizations.
- [x] 265.  Document the complete mapping architecture for adding a new model to the ecosystem.

### Phase 17: Extended Calibration and Evaluation Options

- [x] 266.  Integrate BLEU score evaluation directly post-quantization for translation models.
- [x] 267.  Integrate ROUGE score evaluation for summarization models.
- [x] 268.  Integrate WER (Word Error Rate) evaluation for ASR models.
- [x] 269.  Allow saving evaluation metrics alongside the `optimum_config.json`.
- [x] 270.  Create interactive confusion matrix visualizations post-calibration.
- [x] 271.  Compare generated web artifacts directly against the HF Space standard implementation.
- [x] 272.  Implement custom token-wise perplexity charts during GPTQ calibration.
- [x] 273.  Establish a standard format for sharing community quantization recipes.
- [x] 274.  Implement fallback to standard JIT if Dynamo/TorchScript export fails.
- [x] 275.  Expand test coverage to handle edge case models with sparse attention patterns.

### Phase 18: Specific Kernel Tuning for Web Deployment

- [x] 276.  Generate WebGPU specific memory alignment metadata directly during export.
- [x] 277.  Tag specific nodes for execution on WASM vs WebGPU in heterogenous setups.
- [x] 278.  Export WebNN specific hint metadata for NPU offloading capabilities.
- [x] 279.  Support INT4 quantization of 1D tensors (e.g., biases) if specifically requested to maximize compression.
- [x] 280.  Compile specific math polynomials into look-up tables (LUTs) during export.
- [x] 281.  Replace complex trigonometric sequences with fast approximations if `--fast-math` is flagged.
- [x] 282.  Auto-tune chunk sizes for streaming audio models depending on the `--device` target.
- [x] 283.  Strip `Shape` operations and hardcode shapes if `--static-shapes` is strictly provided.
- [x] 284.  Pre-compute and bake invariant positional embeddings directly into the model graph to save runtime execution.
- [x] 285.  Support externalizing weights into independent files chunks for HTTP range requests.

### Phase 19: Comprehensive Examples & Ecosystem

- [x] 286.  Provide `examples/export_llama_webgpu.sh` script.
- [x] 287.  Provide `examples/quantize_whisper_wasm.sh` script.
- [x] 288.  Provide Jupyter notebook detailing the AWQ calibration process step-by-step.
- [x] 289.  Provide TS/Node.js script demonstrating how to optimize a model without Python.
- [x] 290.  Maintain a "Supported Models Tracker" matching the original HF Optimum page.
- [x] 291.  Host a live gallery of web-optimized models utilizing the resulting `ORTModelForX` classes.
- [x] 292.  Hook up with `transformers.js` to automatically utilize exported models from this pipeline.
- [x] 293.  Add comprehensive API documentation for the optimization AST classes.
- [x] 294.  Create video tutorials showing the size difference before and after W4A16 packing.
- [x] 295.  Write a migration guide for users switching from `optimum-cli` to `onnx9000 optimum`.

### Phase 20: Final Polish and Release Readiness

- [x] 296.  Verify 100% test coverage over graph mutating functions to prevent silent corruption.
- [x] 297.  Ensure binary deterministic outputs (identical inputs + seeds = byte-for-byte identical `.onnx`).
- [x] 298.  Perform a final audit on memory limits to ensure massive models (e.g., 70B parameter models) do not crash the export CLI.
- [x] 299.  Ensure graceful error messaging when users run out of disk space during massive file serialization.
- [x] 300.  Release v1.0 feature complete certification for `onnx9000.optimum`.
