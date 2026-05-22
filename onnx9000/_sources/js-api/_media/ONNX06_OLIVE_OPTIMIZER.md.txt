# Olive Optimizer Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of the `Olive Optimizer` framework within the `onnx9000` ecosystem.
Microsoft's `Olive` is a powerful hardware-aware model optimization tool, but it heavily relies on complex C++ native tools, massive external ML frameworks (PyTorch, ONNX Runtime), and heavy build toolchains to quantize, prune, and optimize models.
Our `onnx9000` reimplementation operates completely in pure Python. It leverages our internal `GraphSurgeon` and `safetensors` modules to perform state-of-the-art model compression (INT8, INT4 quantization, weight packing, operator fusion, and sparsity pruning) entirely in-memory. This zero-dependency architecture means a multi-GB transformer can be downloaded, dynamically quantized, and executed strictly within a browser (Pyodide/WASM) or optimized instantly on an edge device without installing massive pip packages.

## Exhaustive Parity Checklist

### 1. Core Optimization Architecture & Passes (35+ items)

- [x] Implement `OliveModel` abstraction natively
- [x] Implement pure-Python `Pass` base class
- [x] Implement `PassContext` tracking intermediate optimization states
- [x] Implement `QuantizationPass`
- [x] Implement `DynamicQuantizationPass`
- [x] Implement `StaticQuantizationPass` (with calibration logic)
- [x] Implement `WeightOnlyQuantizationPass`
- [x] Implement `PruningPass` (Sparsity)
- [x] Implement `GraphFusionPass` (leveraging internal GraphSurgeon)
- [x] Implement `MixedPrecisionPass` (FP16 / BFloat16)
- [x] Implement `LayoutConversionPass` (NCHW <-> NHWC)
- [x] Implement `OrtPerfTuningPass` (Thread/EP tuning suggestions)
- [x] Implement `OrtTransformerOptimizationPass` (Attention/Gelu fusion)
- [x] Implement Hardware-Targeting logic (CPU, WebGPU, CoreML, TensorRT)
- [x] Implement Pass sequence orchestration (`AutoOptimizer`)
- [x] Support conditional Pass execution based on Target hardware limits
- [x] Evaluate Pass impact on Graph FLOPs (using internal `onnx-tool`)
- [x] Evaluate Pass impact on Graph Memory (using internal `onnx-tool`)
- [x] Evaluate Pass impact on mathematical accuracy (Tolerance checking)
- [x] Track End-to-End latency metrics explicitly during optimization
- [x] Generate comprehensive JSON Optimization Reports (Metrics before/after)
- [x] Catch Sub-graph topology issues and disable passes dynamically
- [x] Provide an interactive visualization of the applied passes (via `Netron` links)
- [x] Serialize strictly compliant `.onnx` models post-optimization
- [x] Pack massive optimized constants into `.safetensors` cleanly
- [x] Ensure the Python package size is under 5MB for Cloudflare Worker deployments
- [x] Provide simple CLI `onnx9000 optimize model.onnx --target=webgpu`
- [x] Implement caching mechanisms for Intermediate Models
- [x] Expose native `OrtSessionOptions` emulation directly to the optimizer
- [x] Support explicit multi-threading controls during execution benchmarking
- [x] Bypass ONNX Runtime dependencies completely during pure-Python optimization phases
- [x] Run mathematical constant folding explicitly before quantization
- [x] Strip Identity nodes explicitly before quantization
- [x] Strip un-used initializers explicitly before quantization
- [x] Extract symbolic shapes to validate layout transformations safely

### 2. Quantization & Weight Compression (40+ items)

- [x] Map FP32 to `Int8` dynamically (`DynamicQuantizeLinear` injection)
- [x] Map FP32 to `Uint8` dynamically
- [x] Map `MatMul` -> `DynamicQuantizeMatMul` automatically
- [x] Map `MatMul` -> `MatMulInteger` (if inputs are statically quantized)
- [x] Map `Conv` -> `QLinearConv`
- [x] Map `Add` -> `QLinearAdd`
- [x] Support Block-wise quantization natively
- [x] Implement K-Means based weight clustering compression
- [x] Implement INT4 (4-bit) quantization explicitly
- [x] Implement AWQ (Activation-aware Weight Quantization) natively in Python
- [x] Implement GPTQ (Generative Pre-trained Transformer Quantization) emulation logic
- [x] Pack two INT4 weights into a single INT8 tensor natively (Bitwise operations)
- [x] Pack INT4 weights into Uint32 / Uint8 buffers aligned for WebGPU
- [x] Extract minimum and maximum weight bounds cleanly in Python memory
- [x] Calculate `scale` (Float32) and `zero_point` (Int8/Uint8) exactly
- [x] Support symmetric quantization (`zero_point` = 0 explicitly)
- [x] Support asymmetric quantization
- [x] Prevent explicit division-by-zero during scale calculation (e.g. all-zero weights)
- [x] Expose explicitly `reduce_range` logic to prevent Int8 overflow in `QLinearConv`
- [x] Track calibration metrics across 1D, 2D, and ND tensors
- [x] Support Histogram based calibration for Static Quantization
- [x] Support Entropy (KL Divergence) based calibration for Static Quantization
- [x] Support MinMax based calibration for Static Quantization
- [x] Inject `QuantizeLinear` and `DequantizeLinear` boundaries safely (Fake Quantization)
- [x] Fold `QuantizeLinear` -> `DequantizeLinear` -> `QuantizeLinear` effectively
- [x] Fuse `BatchNormalization` natively into `Conv` weights BEFORE quantization
- [x] Quantize explicit `Constant` nodes into `Initializer` payloads
- [x] Expose `per_channel` quantization logic for `Conv` (Axis 0 or Axis 1)
- [x] Expose `per_channel` quantization logic for `MatMul`
- [x] Verify `per_tensor` vs `per_channel` limits natively against ONNX Opset specifications
- [x] Handle explicit PyTorch `qint8` and `quint8` translation flawlessly
- [x] Ensure specific biases (e.g. `Conv` bias) are maintained in Int32 / FP32 for precision
- [x] Extract FP16 scale factors implicitly
- [x] Extract BF16 scale factors implicitly
- [x] Validate quantized output mathematically against FP32 original output (PSNR calculation)
- [x] Highlight completely non-quantizable ops natively
- [x] Handle fallback to FP32 cleanly for subgraphs that fail precision tests
- [x] Automatically apply FP16 mixed precision to ops surrounding INT8 boundaries
- [x] Support INT8 -> FP32 Dequantize boundaries for Softmax and Sigmoid
- [x] Inject specific WebGPU friendly shader unpacking logic dynamically if targeted

### 3. Graph Fusions & Pattern Optimization (35+ items)

- [x] Implement `GraphSurgeon` driven `Conv` + `Relu` fusion
- [x] Implement `Conv` + `Clip` fusion
- [x] Implement `Conv` + `Sigmoid` fusion
- [x] Implement `MatMul` + `Add` -> `Gemm` fusion
- [x] Implement `Gemm` + `Relu` fusion
- [x] Implement `MatMul` + `Add` + `Relu` -> `Gemm(Relu)`
- [x] Implement `LayerNormalization` exact math-pattern matching fusion
- [x] Implement `SkipLayerNormalization` (Residual + LayerNorm) fusion
- [x] Implement `FastGelu` exact pattern matching (Erf emulation)
- [x] Implement `BiasGelu` pattern matching
- [x] Implement `MultiHeadAttention` (MHA) pattern extraction and fusion
- [x] Support PyTorch standard Scaled Dot Product Attention (SDPA) to ONNX MHA fusion
- [x] Implement Rotary Positional Embedding (RoPE) fusion natively
- [x] Implement `EmbedLayerNormalization` pattern matching
- [x] Detect implicit Reshape -> Transpose -> Reshape bottlenecks (Memory-bound)
- [x] Optimize Reshape + Transpose sequences statically if constants allow
- [x] Cancel out identity `Cast` operations (e.g. FP32 -> FP32)
- [x] Cancel out redundant `Squeeze` -> `Unsqueeze` sequences
- [x] Cancel out redundant `Split` -> `Concat` sequences
- [x] Collapse nested `Slice` operations mathematically
- [x] Collapse sequential `Add` operations containing pure constants (`Add(C1) + Add(C2) -> Add(C1+C2)`)
- [x] Collapse sequential `Mul` operations containing constants
- [x] Distribute scalar `Mul` across `Add` mathematically if profitable
- [x] Evaluate constant `Shape` subgraphs into explicit arrays
- [x] Deduplicate identical `Constant` Initializers to save VRAM
- [x] Pack `Constant` nodes specifically to align with target Execution Providers
- [x] Support generating `NhwcConv` specialized operators
- [x] Support generating `NhwcMaxPool` specialized operators
- [x] Convert `Dropout` ops to `Identity` unconditionally (eval mode assumed)
- [x] Strip out strictly non-functional nodes (`Identity`)
- [x] Propagate shapes strictly to validate topological assumptions after fusion
- [x] Ensure fusions are mathematically strictly equivalent (or bounded by extreme tolerances)
- [x] Export detailed JSON log of every single fusion performed
- [x] Allow explicit masking of fusions (disabling specific rules via dict configs)
- [x] Support dynamic fusion logic for CustomOps (HuggingFace tokenizers)

### 4. Hardware-Aware Target Tuning (WebGPU, WASM, Apple) (30+ items)

- [x] Expose `Target.WebGPU` enumeration
- [x] Expose `Target.WASM_SIMD` enumeration
- [x] Expose `Target.Accelerate` (Apple) enumeration
- [x] Expose `Target.CoreML` enumeration
- [x] Convert NCHW networks to NHWC natively when targeting WebGPU (Texture cache optimization)
- [x] Un-transpose NHWC networks natively when targeting Accelerate (cblas optimizations)
- [x] Force mixed-precision FP16 dynamically across all math operations for `WebGPU`
- [x] Downcast Float64 natively to Float32 when targeting WebAssembly
- [x] Evaluate specific WGSL storage buffer memory alignment constraints statically
- [x] Quantize constants safely into Uint8/Uint32 matrices for `WebGPU` DP4A unpacking
- [x] Remove completely dynamic sequence lengths (`Loop` / `If`) if compiling to strict WASM pipelines
- [x] Translate unsupported `Mod` operations into `Div`/`Floor` sequences for `WebGPU`
- [x] Extract memory footprint estimates strictly targeting the 256MB WebGPU max buffer limits
- [x] Chunk large `MatMul` operations topologically into multiple smaller `MatMul` + `Add` to bypass memory limits
- [x] Export weights completely externalized (`.bin`) for HTTP streaming environments (Web targets)
- [x] Ensure 100% of operators within the target model are WebGPU compatible
- [x] Inject explicit Javascript/TypeScript specific tensor descriptors
- [x] Extract explicit CoreML compatibility limitations natively
- [x] Map 1D `Conv` to 2D `Conv` (with padding) automatically for strict backends
- [x] Force explicit sequence length batching (`[1, 128]` -> `[1, 256]`) for fixed-size compilers
- [x] Pad constant tensors dynamically to multiples of 4 bytes (WebGPU uniform restrictions)
- [x] Optimize 64-bit comparisons (e.g. `Int64` -> `Int32`) for Javascript BigInt overhead reduction
- [x] Expose dynamic compilation flags to Emscripten explicitly (`-Os`, `-msimd128`)
- [x] Estimate execution time per operator heuristically based on target TFLOPS
- [x] Warn dynamically if an optimization fundamentally changes the output data type from float to int
- [x] Benchmark compiled models across multiple backend execution providers explicitly
- [x] Test the memory mapping overhead dynamically on standard macOS targets
- [x] Support strict INT8/FP32 fallback policies if certain targets fail to support specific quantized ops
- [x] Inject native WebWorker threading policies into WASM environments
- [x] Auto-tune Thread counts (e.g. `IntraOpNumThreads=4`) based on browser logical cores

### 5. Pruning & Sparsity Strategies (25+ items)

- [x] Implement Global Magnitude Pruning (Unstructured)
- [x] Implement Block-wise Magnitude Pruning (Structured)
- [x] Support explicit sparsity targets (e.g., `sparsity=0.75` / 75% zeros)
- [x] Evaluate specific L1/L2 norm bounds natively in Python memory
- [x] Modify `Constant` tensors physically in memory (Setting values < Threshold to `0.0`)
- [x] Support NxM Sparse Block packing (e.g. 2:4 sparsity for Nvidia Ampere GPUs)
- [x] Emit explicit Sparse ONNX Tensors (`SparseTensorProto`)
- [x] Compress large, highly sparse matrices explicitly into external formats to save disk space
- [x] Detect implicit sparsity within pre-trained weights natively (Reporting the % of zeros)
- [x] Implement N:M sparsity mask generation natively
- [x] Calculate explicit theoretical FLOP reduction after pruning
- [x] Evaluate Drop-in accuracy impact (Tolerance matching) explicitly after applying masks
- [x] Export `SparseTensor` -> `DenseTensor` decompression nodes selectively if Target lacks sparse ops
- [x] Highlight completely dead channels (All-zero Conv filters)
- [x] Prune completely dead Conv channels explicitly (Removing them from the model topology and updating subsequent biases)
- [x] Identify and prune explicit dead-ends (Subgraphs outputting purely to zeroed out matrices)
- [x] Track the resulting dimension modifications recursively through the entire topological sorted graph
- [x] Update `Reshape` / `Shape` constants explicitly after structured pruning reduces channel sizes
- [x] Implement layer-specific sparsity targets (e.g. pruning Attention less heavily than FFNs)
- [x] Output a rich Markdown/JSON Sparsity Report
- [x] Calculate compressed ZIP/GZIP file size equivalents for the pruned model natively
- [x] Provide explicit hooks for hardware-accelerated Sparse `MatMul` in WebGPU
- [x] Provide hooks for Sparse `Conv` in standard WASM targets
- [x] Implement dynamic random pruning algorithms natively
- [x] Catch explicitly un-prunable operations gracefully (e.g. strict positional embeddings)

### 6. Calibration & Accuracy Evaluation Loop (25+ items)

- [x] Parse explicitly User-Provided datasets (`numpy`, `json`, `csv`) for calibration runs
- [x] Support PyTorch `DataLoader` emulation purely natively in Python
- [x] Iterate dynamically across batches running explicit forward passes natively in Python
- [x] Extract minimum and maximum activation values for every topological intermediate node
- [x] Capture average activation histograms natively
- [x] Measure exact KL Divergence (Entropy) across the distribution bins natively
- [x] Execute completely locally (Zero RPC/Network calls) during calibration
- [x] Prevent Memory Leaks during multi-batch calibration by aggressively garbage collecting intermediate graphs
- [x] Compare `Float32` Baseline vs `Int8` Quantized model using Top-1 / Top-5 Accuracy Logic
- [x] Compare Mean Squared Error (MSE) dynamically
- [x] Compare Cosine Similarity dynamically
- [x] Compare Peak Signal to Noise Ratio (PSNR) dynamically
- [x] Provide a fallback pass (reverting specific nodes back to `Float32`) if accuracy drops below threshold
- [x] Expose an automated binary search (bisecting nodes) to identify exactly which node caused the precision drop
- [x] Highlight "Sensitive" nodes graphically to the user (e.g. `LayerNormalization`, `Softmax` bounds)
- [x] Automatically enforce FP16 / FP32 precision strictly on the identified sensitive nodes
- [x] Profile peak memory allocation exactly during the calibration sequence
- [x] Validate dynamic shapes (`-1`) function flawlessly across all batch variations during calibration
- [x] Serialize standard `CalibrationTable.json`
- [x] Import pre-existing `CalibrationTable.json`
- [x] Handle explicit multi-input models seamlessly (e.g. `input_ids`, `attention_mask`)
- [x] Evaluate generative models natively by accumulating multi-step probabilities
- [x] Support explicit metric logging callbacks (`tqdm` / `logging`)
- [x] Bypass calibration completely if `StaticQuantization` falls back to `DynamicQuantization`
- [x] Test the entire calibration loop inside a Pyodide web environment (WASM Memory Bounds)

### 7. Testing, Edge Cases & Tooling Parity (25+ items)

- [x] Unit Test: Quantize standard ResNet50 (FP32 -> INT8) seamlessly
- [x] Unit Test: Prune standard BERT (75% sparsity) natively
- [x] Unit Test: Optimize massive Whisper topology (MHA / Gelu Fusions) seamlessly
- [x] Unit Test: Convert generic CNN (NCHW) to WebGPU-friendly (NHWC)
- [x] Validate structural equality check against official Microsoft Olive ONNX outputs
- [x] Verify execution exactly matches ORT (Tolerance `atol=1e-2` for INT8)
- [x] Profile memory usage of the optimization script natively (ensure it doesn't OOM on large models)
- [x] Test 2:4 Structured Sparsity generation logic accurately
- [x] Catch explicitly unsupported ONNX operations and exclude them from fusions cleanly
- [x] Prevent topological loops from infinitely freezing graph traversal scripts
- [x] Extract massive multi-gigabyte Constants efficiently via `.safetensors` memory mapping
- [x] Convert `.bin` ONNX external data seamlessly into `.safetensors` natively during optimization
- [x] Emulate Olive's `System` abstraction seamlessly (LocalSystem vs Azure/Docker) purely using native OS calls
- [x] Provide interactive CLI: `onnx9000 optimize config.json` mimicking Olive's workflow
- [x] Map Python decorators securely to allow users to inject custom Metric calculations
- [x] Provide detailed debug verbosity mapping specific rule application success/failures
- [x] Support auto-detection of model architecture (`BERT`, `GPT`, `YOLO`) to auto-select optimal pass pipelines
- [x] Catch explicitly invalid input arrays gracefully (e.g. wrong type)
- [x] Export TypeScript / JS bindings for the JSON optimization reports
- [x] Execute `pytest` across all permutations of FP32 / INT8 / FP16 natively
- [x] Extract Subgraphs natively before optimizing massive graphs iteratively
- [x] Verify Javascript `Number` limitations (preventing 64-bit integer corruption) safely
- [x] Stream JSON log events progressively to stdout
- [x] Clean up temporary calibration directories explicitly automatically
- [x] Emulate Microsoft Olive's strict directory structure (`models`, `metrics`, `passes`) internally

### 8. Exhaustive Operator Fusion Passes (40+ items)

- [x] Implement `AttentionFusion` (Standard PyTorch `Q`, `K`, `V` -> `Attention`)
- [x] Implement `AttentionFusion` (with `Mask` injection)
- [x] Implement `AttentionFusion` (with `Past_Key`, `Past_Value` routing)
- [x] Implement `AttentionFusion` (with `Present_Key`, `Present_Value` outputs)
- [x] Implement `AttentionFusion` (Cross Attention explicitly)
- [x] Implement `AttentionFusion` (FlashAttention optimization fallback if supported)
- [x] Implement `EmbedLayerNormFusion` (Standard BERT embedding)
- [x] Implement `EmbedLayerNormFusion` (with Word, Position, and Token type embeddings)
- [x] Implement `SkipLayerNormFusion` (Bias + Add + LayerNorm)
- [x] Implement `FastGeluFusion` (Erf approximation sequence)
- [x] Implement `FastGeluFusion` (Tanh approximation sequence)
- [x] Implement `BiasGeluFusion` (Add + Gelu)
- [x] Implement `BiasDropoutFusion` (Add + Dropout + Add) (Removes Dropout statically)
- [x] Implement `NhwcConvFusion` (WebGPU / TensorRT spatial optimization)
- [x] Implement `NhwcMaxPoolFusion`
- [x] Implement `ConvAddFusion` (Folding Bias natively into Conv)
- [x] Implement `ConvMulFusion` (Folding Scale natively into Conv)
- [x] Implement `ConvBatchNormFusion` (Mathematical weight/bias update)
- [x] Implement `MatMulAddFusion` (Creating `Gemm`)
- [x] Implement `GemmReluFusion` (Appends activation to Gemm)
- [x] Implement `MatMulAddReluFusion` (Creating `Gemm` with Relu)
- [x] Implement `ReshapeTransposeReshapeFusion` (Memory bandwidth bottleneck resolution)
- [x] Implement `ConcatSplitFusion` (Canceling out redundant splits)
- [x] Implement `SqueezeUnsqueezeFusion` (Canceling out dimension toggles)
- [x] Implement `PadSliceFusion` (Canceling out spatial extensions)
- [x] Implement `CastCastFusion` (Canceling redundant precision changes)
- [x] Implement `ConstantFolding` (Pre-calculating pure math nodes)
- [x] Implement `ShapeConstantFolding` (Pre-calculating static shapes)
- [x] Implement `SliceConstantFolding`
- [x] Implement `GatherConstantFolding`
- [x] Implement `ConcatConstantFolding`
- [x] Implement `TransposeConstantFolding` (Baking transposed weights explicitly)
- [x] Implement `ReshapeConstantFolding` (Baking reshaped weights explicitly)
- [x] Implement `SplitConstantFolding`
- [x] Implement `TileConstantFolding`
- [x] Implement `ExpandConstantFolding`
- [x] Detect and apply `RotaryPositionalEmbedding` (RoPE) exact mathematical pattern
- [x] Detect and apply `GroupNorm` mathematical pattern
- [x] Detect and apply `LayerNorm` mathematical pattern
- [x] Prevent fusions dynamically if the resulting node violates Execution Provider bounds

### 9. LLM Specific Optimizations (30+ items)

- [x] Implement `LoRAMergePass` (Folding LoRA adapters A and B statically into Master Weights)
- [x] Detect standard LLM layer hierarchies (e.g. `layers.0.attention.wq`) natively
- [x] Strip out unused tokenizer subgraphs if `TextDecoder` outputs are not requested
- [x] Force KV-Cache (`past_key_values`) generation dynamically if the model lacks them
- [x] Support generating `BeamSearch` / `GreedySearch` wrappers natively wrapping the LLM
- [x] Extract generation configuration (`max_length`, `temperature`) into ONNX defaults
- [x] Compress 100GB+ LLM weights strictly using `safetensors` external data
- [x] Implement GPTQ 4-bit unpacking operations dynamically using WebGPU WGSL or PyCUDA natively
- [x] Implement AWQ (Activation-aware Weight Quantization) 4-bit unpacking operations dynamically
- [x] Inject `Int4` to `Float16` unpacking layers efficiently at runtime
- [x] Inject `UInt4` (unsigned) to `Float16` unpacking layers efficiently
- [x] Verify `Float16` precision across the entire Attention mechanism (avoiding NaNs)
- [x] Map `BFloat16` correctly to standard `Float32` explicitly if the target (e.g. WebGPU) lacks BF16
- [x] Extract exact prompt padding rules (`left` vs `right`) into the model `AttributeProto`
- [x] Generate specific `<|endoftext|>` stopping criteria nodes recursively
- [x] Provide vocabulary compression (removing strictly unused tokens from the embedding matrix)
- [x] Remove `Dropout` completely from the transformer blocks statically
- [x] Optimize explicitly `SwiGLU` (Silu) activation patterns natively
- [x] Optimize explicitly `GatedLinearUnit` (GLU) activation patterns natively
- [x] Map LLaMA specific `RMSNorm` (Root Mean Square Normalization) pattern fusion
- [x] Map Mistral specific `SlidingWindowAttention` natively if bounded
- [x] Provide explicit quantization overrides (e.g. "Do not quantize the final `lm_head`")
- [x] Extract embedding arrays natively as separate `.safetensors` to stream into memory first
- [x] Profile memory bounds exactly (LLM parameter footprint + KV-cache dynamically expanding footprint)
- [x] Provide warning boundaries if maximum KV-cache size exceeds 4GB (WebAssembly limits)
- [x] Generate static memory offsets (Memory Arena) explicitly for the LLM layers
- [x] Generate standard ONNX `ValueInfoProto` for all dynamic Sequence axes
- [x] Optimize scalar additions (e.g. LayerNorm epsilon) directly into WGSL constants if WebGPU targeted
- [x] Map specific execution providers (`tensorrt`, `openvino`) dynamically based on OS heuristics
- [x] Export LLM fully compatible with `onnxruntime-web` streaming execution

### 10. Advanced Hardware Bounds & Diagnostics (25+ items)

- [x] Calculate `ExecutionProvider` fallback latency penalties heuristically (Memcpy overhead)
- [x] Recommend explicit `IntraOpNumThreads` optimizations based on CPU Core counts
- [x] Recommend explicit `InterOpNumThreads` optimizations
- [x] Simulate Apple Metal (Accelerate) specific transpose penalties (row vs column major)
- [x] Simulate WebGPU specific `StorageBuffer` limitations (128MB/256MB max chunks)
- [x] Partition graph dynamically if a single tensor exceeds the WebGPU `StorageBuffer` limit
- [x] Warn if WebGL textures exceed max texture dimensions (usually 4096x4096 or 8192x8192)
- [x] Simulate Android / iOS specific memory eviction policies
- [x] Expose native CPU `AVX512` vs `AVX2` specific block-padding for `Conv` weights
- [x] Expose native ARM `NEON` / `SVE` block-padding recommendations
- [x] Trace latency across explicitly injected memory boundaries (`MemcpyToHost`)
- [x] Output a rich JSON diagnostic outlining the Top-10 most computationally expensive nodes
- [x] Output a rich JSON diagnostic outlining the Top-10 most memory expensive nodes
- [x] Highlight any nodes causing precision loss natively (FP32 -> FP16 bounds checking)
- [x] Highlight un-fused elementwise operations that are heavily memory-bound (e.g. `Add` + `Mul`)
- [x] Verify `Safetensors` header size limits are respected during external data dumping
- [x] Provide strict structural topology tests (No cycles, no dead ends) before serialization
- [x] Execute completely synchronously if requested (no async barriers during optimization)
- [x] Validate WASM `SharedArrayBuffer` Thread counts natively (fallback to 1 if COOP/COEP fail)
- [x] Generate detailed `chrome://tracing` compatible `.json` execution profiles
- [x] Simulate TensorRT Engine building memory limits (Workspace sizes) natively
- [x] Test the latency of `DynamicQuantizeLinear` itself (it can be slower than FP32 if improperly fused)
- [x] Highlight zero-variance channels (Conv filters that are identical)
- [x] Simulate CPU L1/L2/L3 cache misses explicitly based on layout (NCHW vs NHWC)
- [x] Expose an interactive CLI `--debug` flag to step through every single applied optimization graphically
