# onnx-tool Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `onnx-tool` within the `onnx9000` ecosystem.
The original `onnx-tool` is an excellent diagnostic utility for profiling MACs, FLOPs, parameter counts, and static memory allocations. However, it often relies on heavy system environments or is used purely as a CLI script.
Our `onnx9000` reimplementation integrates these advanced profiling and symbolic shape inference capabilities natively into our pure-Python, WASM-compatible Intermediate Representation. This means you can profile the exact memory bounds and compute intensity of a multi-GB transformer or vision model instantly in the browser or on a cold serverless node without executing the model or installing `onnxruntime`.

## Exhaustive Parity Checklist

### 1. Shape Inference & Symbolic Math (50+ items)

- [x] Implement zero-dependency static shape inference
- [x] Implement symbolic shape inference (e.g., tracking `batch_size`, `seq_len`)
- [x] Support dynamic symbolic variables recursively across subgraphs
- [x] Resolve symbolic math natively (e.g., `seq_len * 2`)
- [x] Evaluate `Reshape` dynamic dimensions (`-1`) via volume preservation
- [x] Evaluate `MatMul` output shapes symbolically `[..., M, K] x [..., K, N] -> [..., M, N]`
- [x] Evaluate `Conv` output shapes statically (with strides, dilations, pads)
- [x] Evaluate `ConvTranspose` output shapes statically
- [x] Evaluate `MaxPool` output shapes statically
- [x] Evaluate `AveragePool` output shapes statically
- [x] Evaluate `GlobalAveragePool` output shapes statically
- [x] Evaluate `Gather` output shapes dynamically based on indices length
- [x] Evaluate `Slice` output shapes statically (when bounds are constant)
- [x] Evaluate `Slice` output shapes symbolically (when bounds are dynamic variables)
- [x] Evaluate `Concat` output shapes (summing across specified axis)
- [x] Evaluate `Split` output shapes
- [x] Evaluate `Tile` output shapes
- [x] Evaluate `Expand` output shapes (broadcasting rules)
- [x] Evaluate `Pad` output shapes (constant padding values)
- [x] Evaluate `TopK` output shapes
- [x] Evaluate `ArgMax` / `ArgMin` output shapes
- [x] Evaluate `NonZero` output shapes (as fully dynamic / undefined bounds)
- [x] Evaluate `Where` output shapes (broadcasting rules)
- [x] Evaluate `Shape` node output values symbolically
- [x] Evaluate `Size` node output values symbolically
- [x] Handle explicit PyTorch symbolic naming (e.g., `SymInt`)
- [x] Handle implicit broadcasting rules across all elementwise arithmetic (`Add`, `Mul`, `Div`, `Sub`)
- [x] Handle implicit broadcasting rules across all logical ops (`And`, `Or`, `Equal`, `Less`)
- [x] Propagate shape inference deeply through `If` subgraphs (ensuring branch shape equality)
- [x] Propagate shape inference deeply through `Loop` subgraphs (resolving loop state dimensions)
- [x] Validate loop body outputs match loop body inputs dimensionality
- [x] Extract sequence length bounds dynamically for `SequenceConstruct`
- [x] Extract sequence length bounds dynamically for `SplitToSequence`
- [x] Implement custom shape inference for recognized custom operators (e.g., FlashAttention)
- [x] Fallback gracefully when encountering unrecognized CustomOps (marking outputs as `Unknown`)
- [x] Infer `Cast` output data types
- [x] Propagate `dtype` inference alongside shape inference explicitly
- [x] Handle type promotion rules strictly across math ops (e.g. Int32 + Float32 -> Float32)
- [x] Handle float16 / bfloat16 propagation safely
- [x] Deduplicate symbolically equivalent shapes (e.g. `dim0` vs `batch`)
- [x] Allow manual user overrides for specific symbolic dimensions (e.g. `batch=1, seq=128`)
- [x] Strip undefined dimensions to bounded sizes for profiling
- [x] Track minimum bounds for dynamic shapes
- [x] Track maximum bounds for dynamic shapes
- [x] Inject `ValueInfo` natively back into the Graph after inference
- [x] Remove explicitly redundant `ValueInfo` metadata to save space
- [x] Throw explicit exceptions on shape validation failures (e.g. non-broadcastable adds)
- [x] Support `Shape` operator constant folding via symbolic evaluation
- [x] Support `Reshape` -> `Reshape` cancellations mathematically

### 2. MACs & FLOPs Computation Profiling (40+ items)

- [x] Profile MACs (Multiply-Accumulates) for `MatMul`
- [x] Profile FLOPs (Floating-Point Operations) for `MatMul`
- [x] Profile MACs for `Conv` (Standard 2D/3D)
- [x] Profile MACs for `Conv` (Depthwise / Grouped)
- [x] Profile FLOPs for `Conv`
- [x] Profile MACs for `ConvTranspose`
- [x] Profile FLOPs for `ConvTranspose`
- [x] Profile MACs for `Gemm`
- [x] Profile FLOPs for `Gemm`
- [x] Profile FLOPs for `BatchNormalization`
- [x] Profile FLOPs for `LayerNormalization`
- [x] Profile FLOPs for `InstanceNormalization`
- [x] Profile FLOPs for Elementwise Math (`Add`, `Sub`, `Mul`, `Div`)
- [x] Profile FLOPs for Transcendental Math (`Exp`, `Log`, `Sin`, `Cos`)
- [x] Profile FLOPs for Activations (`Relu`, `Sigmoid`, `Tanh`, `Gelu`)
- [x] Profile FLOPs for `Softmax`
- [x] Profile FLOPs for `ReduceMean`, `ReduceSum`, `ReduceMax`, `ReduceMin`
- [x] Profile Memory Bandwidth limits (Bytes Read/Written) for memory-bound ops (`Reshape`, `Transpose`)
- [x] Support profiling dynamic shapes via symbolic mathematical formulas (`MACs = batch * seq * 1024`)
- [x] Support profiling dynamic shapes via explicit overrides (`MACs(batch=1)`)
- [x] Aggregate Total MACs globally
- [x] Aggregate Total FLOPs globally
- [x] Output per-node MACs summary
- [x] Output per-node FLOPs summary
- [x] Handle dynamic branching: average MACs across `If` subgraph branches
- [x] Handle dynamic branching: worst-case MACs across `If` subgraph branches
- [x] Handle dynamic looping: multiply loop body MACs by static loop iterations
- [x] Handle dynamic looping: symbolic loop iterations (`MACs = N * body`)
- [x] Profile Transformer architectures cleanly (Attention FLOPs tracking)
- [x] Profile CNN architectures cleanly (spatial dimensionality tracking)
- [x] Profile CustomOps (if custom FLOPs formula provided)
- [x] Distinguish INT8 MACs vs FP32 MACs natively
- [x] Distinguish INT4 MACs natively
- [x] Distinguish FP16/BF16 MACs natively
- [x] Ignore FLOPs for routing/index ops (`Gather`, `Scatter`, `NonZero`)
- [x] Account for sparsity automatically in `SparseTensor` profiled ops (if structural)
- [x] Print top-K most compute-intensive nodes (Bottleneck analysis)
- [x] Print MACs/FLOPs distribution pie-chart data points
- [x] Expose API to query cumulative FLOPs up to a specific node layer
- [x] Provide ratio of Compute vs Memory-bound characteristics per node

### 3. Static Parameter & Memory Footprint Profiling (40+ items)

- [x] Profile Total Parameter count
- [x] Profile Total Constant Memory footprint (MB/GB)
- [x] Distinguish trainable parameters (`Parameter` inputs) vs frozen constants
- [x] Provide precise memory sizes based on `dtype` (FP32=4B, FP16=2B, INT8=1B)
- [x] Profile activation memory footprint (peak memory during inference)
- [x] Estimate working-set RAM requirement statically via topological simulation
- [x] Calculate activation lifecycle boundaries (when tensors can be freed/reused)
- [x] Simulate greedy memory arena allocation natively (calculating exact contiguous buffers)
- [x] Simulate offset-based static memory allocation (for edge device targeting)
- [x] Support dynamic shape overrides for activation footprint calculations
- [x] Profile individual node memory ingestion (Input tensor sizes)
- [x] Profile individual node memory generation (Output tensor sizes)
- [x] Map parameter counts to specific sub-architectures (e.g., `layer1.attention` has X parameters)
- [x] Identify redundant constant sizes (`Tile` -> `Constant` bloat detection)
- [x] Track memory fragmentation in the simulated arena
- [x] Identify shared initializers directly
- [x] Ignore zero-size arrays natively
- [x] Calculate total disk-size footprint vs RAM-size footprint (e.g. sparse formats vs dense expansion)
- [x] Highlight un-fused nodes leading to excessive activation memory (e.g. `Conv` + `Add` separate tensors)
- [x] Output layer-by-layer memory trajectory graph data
- [x] Extract peak memory bottleneck node specifically
- [x] Analyze attention mask activation memory (quadratic expansion profiling)
- [x] Evaluate grouped convolution memory reductions
- [x] Profile ONNX Sequence memory usage
- [x] Support profiling specific INT4 packed weight models
- [x] Provide precise memory savings report comparing FP32 -> FP16 -> INT8
- [x] Check external data alignment requirements natively
- [x] Detect huge attributes that should be Initializers
- [x] Validate `Float64` usage to suggest downcasting for memory savings
- [x] Validate `Int64` usage to suggest downcasting
- [x] Aggregate overall "Model Compute Intensity" (FLOPs / Byte Ratio)
- [x] Output pie-chart data points for Parameter distribution
- [x] Output pie-chart data points for Activation distribution
- [x] Provide heuristic bounds for WebGPU buffer limits (128MB max) vs Graph structure
- [x] Expose API to get exact byte-offset of any tensor in a simulated arena
- [x] Report estimated latency given theoretical hardware TOPS (TeraOps per second)
- [x] Report estimated latency given theoretical hardware memory bandwidth (GB/s)

### 4. Graph Topology Optimization Checks (30+ items)

- [x] Detect missing `ConstantFolding` opportunities automatically
- [x] Detect redundant `Transpose` operations natively
- [x] Detect redundant `Cast` operations
- [x] Detect missing `BatchNorm` fusion opportunities
- [x] Detect missing `Scale` fusion opportunities
- [x] Detect missing `Gelu` fusion opportunities
- [x] Detect un-fused `MatMul` + `Add` structures
- [x] Identify deeply nested `If` subgraphs that can be flattened
- [x] Identify scalar math chains that can be analytically simplified
- [x] Highlight completely unused initializers
- [x] Highlight completely unused global inputs
- [x] Detect Identity/No-Op chains
- [x] Analyze sparsity of weight constants to suggest pruning optimizations
- [x] Suggest `int4` quantization if weight distributions are highly uniform
- [x] Detect dynamic ops (`NonZero`) driving massive downstream dynamic allocations
- [x] Suggest replacing older ONNX constructs with modern variants (Opset suggestions)
- [x] Flag operations known to be slow on GPUs (e.g., dynamic `Loop`)
- [x] Flag operations unsupported by common WebGPU backends (e.g., complex numbers)
- [x] Profile tree-ensemble transpilation complexity statically
- [x] Generate detailed "Optimization Opportunities" text report
- [x] Highlight layout conflicts (mixing NCHW and NHWC implicitly)
- [x] Highlight data type bottlenecks (e.g., FP16 -> FP32 -> FP16 sandwiches)
- [x] Expose programmatic JSON list of all identified optimizations
- [x] Expose automated apply functions for the identified optimizations (via GraphSurgeon)

### 5. Detailed Layer/Module Analysis & Grouping (30+ items)

- [x] Implement smart node-grouping based on naming conventions (e.g., `model.layer.0.*`)
- [x] Group MACs/FLOPs recursively by namespace
- [x] Group Memory recursively by namespace
- [x] Group Parameters recursively by namespace
- [x] Export hierarchical JSON profile based on namespaces
- [x] Collapse namespaces graphically in console output
- [x] Recognize standard PyTorch export names natively (`aten::conv2d`)
- [x] Recognize standard TensorFlow export names natively
- [x] Handle unrecognized namespaces by clustering connected components
- [x] Provide API to explicitly define grouping tags manually
- [x] Analyze layer-by-layer sparsity
- [x] Profile sequence expansion boundaries globally
- [x] Map profiled stats directly back to Python PyTorch `nn.Module` names if metadata exists
- [x] Highlight highly repetitive sub-structures (e.g., 24 identical transformer layers)
- [x] Summarize average metrics per transformer block
- [x] Summarize total attention head parameters vs feed-forward parameters
- [x] Analyze CNN depthwise vs pointwise compute distribution
- [x] Provide text-based terminal UI (TUI) hierarchical folding tables
- [x] Emit CSV files mapping Node -> Layer -> Stats
- [x] Emit Pandas DataFrame compatible dictionaries internally

### 6. Zero-Dependency & Lightweight Runtime Integrations (30+ items)

- [x] Run profiling logic purely via `onnx9000` Python API (no native binaries)
- [x] Support profiling >10GB LLMs directly via memory-mapped IO (without OOM)
- [x] Execute completely within Pyodide/WASM limits
- [x] Integrate with `Netron` visualizer as the backend profiling engine
- [x] Expose WebWorker compatible async profiling functions
- [x] Run instant architecture analysis in serverless functions (AWS Lambda)
- [x] Provide CLI utility: `onnx9000 profile model.onnx`
- [x] Output results in rich Markdown
- [x] Output results in rich JSON
- [x] Output results in strict CSV
- [x] Print beautiful ASCII tables (Rich/Colorama styled) dynamically
- [x] Track profiling execution time itself (should be sub-100ms for massive models)
- [x] Handle disconnected graphs natively
- [x] Generate HTML report templates
- [x] Seamless integration directly into the `onnx9000` Graph Surgeon API (`graph.profile()`)
- [x] Deployable instantly via CDN (JS transpiled profile logic)
- [x] Allow streaming of results as graph is being analyzed (for extremely large models)

### 7. Extensive Profiling Edge Cases & Validations (30+ items)

- [x] Unit Test: Profile MACs on standard ResNet50
- [x] Unit Test: Profile FLOPs on standard ResNet50
- [x] Unit Test: Profile Mem on standard ResNet50
- [x] Unit Test: Profile standard BERT (Attention mechanism scaling)
- [x] Unit Test: Profile MobileNet (Depthwise convolutions specifically)
- [x] Unit Test: Profile Dynamic sequence lengths on Llama 3
- [x] Unit Test: Ensure `Slice` operations handle symbolic dimensions cleanly
- [x] Unit Test: Validate output equivalence with official `onnx-tool` counts (atol=1%)
- [x] Unit Test: Validate peak memory allocation simulations against PyTorch native traces
- [x] Validate `GatherND` memory access counts
- [x] Validate `ScatterND` memory access counts
- [x] Prevent recursion limits on extremely deep models (e.g., 1000 layers)
- [x] Optimize symbolic resolution equations (algebraic simplification internally)
- [x] Check `Reshape` product validity (e.g. `1 * 3 * 224 * 224 == batch * x * y * z`)
- [x] Profile integer multiplications differently from floating point natively (if requested)
- [x] Profile specific ONNX quantization operators (QLinearConv MACs)
- [x] Profile dequantize -> math -> quantize cycles efficiently
- [x] Check symbolic shape stability (variables failing to resolve completely)
- [x] Prevent division by zero mathematically during FLOP division equations
- [x] Validate execution against opset versions 1-21 structurally
- [x] Fallback gracefully when encountering mathematically undefined subgraphs (e.g., RNG nodes)

### 8. External Data & Advanced Deployment Profiling (50+ items)

- [x] Accurately profile ONNX `.bin` external data sizes natively without parsing
- [x] Detect broken external data links during profiling
- [x] Expose HTTP byte-range overhead analysis for streamed models
- [x] Estimate load time given generic Network Bandwidth speeds (e.g., 10MB/s)
- [x] Simulate chunked loading memory footprints
- [x] Profile WebGL uniform buffer limit conflicts natively
- [x] Profile WebGL texture size limit conflicts natively
- [x] Profile WebGPU storage buffer alignment mismatches natively
- [x] Analyze matrix row-major vs column-major transpose penalties explicitly
- [x] Provide specific WASM SIMD128 memory alignment checks
- [x] Warn on Float64 usage specifically targeting WASM limits
- [x] Warn on dynamic memory allocations targeting WASM limits
- [x] Flag operators that force synchronous CPU fallbacks natively
- [x] Validate model fits securely inside Pyodide 2GB RAM limits natively
- [x] Analyze specific graph topologies known to cause JS GC pressure (excessive small objects)
- [x] Analyze memory-arena pre-allocation sizes specifically for JS TypedArrays

### 9. Operator-Specific FLOP/MAC Definitions (40+ items)

- [x] Define precise FLOPs for `Einsum` natively based on equation strings
- [x] Define precise FLOPs for `ConvInteger`
- [x] Define precise FLOPs for `MatMulInteger`
- [x] Define precise FLOPs for `QLinearConv`
- [x] Define precise FLOPs for `QLinearMatMul`
- [x] Define precise FLOPs for `LSTM` (per step and unrolled)
- [x] Define precise FLOPs for `GRU` (per step and unrolled)
- [x] Define precise FLOPs for `RNN` (per step and unrolled)
- [x] Define precise FLOPs for `Multinomial`
- [x] Define precise FLOPs for `RandomNormal`
- [x] Define precise FLOPs for `RandomUniform`
- [x] Define precise FLOPs for `GridSample`
- [x] Define precise FLOPs for `RoiAlign`
- [x] Define precise FLOPs for `MaxRoiPool`
- [x] Define precise FLOPs for `Resize` (Nearest)
- [x] Define precise FLOPs for `Resize` (Bilinear/Linear)
- [x] Define precise FLOPs for `Resize` (Cubic)
- [x] Define precise FLOPs for `SpaceToDepth` / `DepthToSpace` (Zero FLOPs, memory bound)
- [x] Define precise FLOPs for `SpaceToDepth` memory bandwidth
- [x] Define precise FLOPs for `DepthToSpace` memory bandwidth
- [x] Define precise FLOPs for `Pad`
- [x] Define precise FLOPs for `Hardmax`
- [x] Define precise FLOPs for `LogSoftmax`
- [x] Define precise FLOPs for `HardSigmoid`
- [x] Define precise FLOPs for `HardSwish`
- [x] Define precise FLOPs for `Shrink`
- [x] Define precise FLOPs for `PRelu`
- [x] Define precise FLOPs for `CumSum`
- [x] Define precise FLOPs for `ReverseSequence`
- [x] Define precise FLOPs for `BitShift`
- [x] Define precise FLOPs for `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot`
- [x] Define precise FLOPs for `Round`
- [x] Define precise FLOPs for `IsInf`, `IsNaN`
- [x] Define precise FLOPs for `SequenceConstruct`, `SequenceAt`, `SequenceEmpty`, etc.
- [x] Differentiate memory bandwidth for `Concat` (copying) vs `Split` (viewing if supported)

### 10. Advanced Dynamic Range & Data Type Analysis (20+ items)

- [x] Profile min/max value bounds for `Constant` tensors natively
- [x] Profile sparsity percentage (zeros) for `Constant` tensors natively
- [x] Warn if `Float32` constants are strictly integers visually
- [x] Warn if `Float32` constants fit safely within `Float16` bounds without underflow
- [x] Warn if `Int64` constants fit safely within `Int32` or `Int8` bounds
- [x] Profile string lengths explicitly for `String` tensors
- [x] Evaluate `BFloat16` distribution ranges specifically
- [x] Highlight subnormal (denormal) values in `Float32` constants (can cause severe performance drops)
- [x] Highlight NaNs or Infs explicitly baked into constants
- [x] Analyze dynamic value ranges of `Shape` / `Size` constants if evaluated
- [x] Provide distribution histograms of weight parameters internally (Text UI output)
- [x] Quantify theoretical memory savings if entire graph is cast to `Float16`
- [x] Quantify theoretical memory savings if entire graph is cast to `Int8`
- [x] Quantify theoretical memory savings if sparsity is leveraged natively

### 11. Custom Memory Planning & Arena Simulation (20+ items)

- [x] Simulate First-Fit contiguous memory allocation scheme
- [x] Simulate Best-Fit contiguous memory allocation scheme
- [x] Simulate explicit buffer reuse lifetimes based on topological traversal
- [x] Provide explicit buffer offsets for every tensor in the simulated arena
- [x] Support generating C/C++ header arrays describing the static memory plan
- [x] Simulate in-place operation memory optimization natively (e.g. `Relu` modifying input)
- [x] Simulate shared tensor memory for `Reshape` / `Flatten` / `Squeeze` (Zero-copy views)
- [x] Calculate Peak Arena fragmentation natively
- [x] Simulate multi-arena planning (e.g. separate arena for weights vs activations)
- [x] Export memory plan directly to `GraphSurgeon` attributes for compiled backends

### 12. Command Line & Developer Ergonomics (10+ items)

- [x] Output interactive HTML Flamegraphs for memory/compute profiles
- [x] Generate D3.js TreeMaps representing hierarchical parameter distribution
- [x] Expose native Python decorators `@onnx9000.profile` for instant model logging
- [x] Support diffing two ONNX models natively (`onnx9000 profile A.onnx B.onnx --diff`)
- [x] Export diff report showing exact changes in FLOPs/Params
- [x] Implement colorized terminal outputs using standard ANSI escapes
- [x] Graceful fallback for massive models on low-RAM machines (chunked profiling)

### 13. Advanced Hardware & Web Profiling Targets (25+ items)

- [x] Simulate Apple Metal specific thread-group memory alignment
- [x] Simulate WebGPU specific Workgroup alignment (e.g. multiples of 64 or 256)
- [x] Simulate WebGL specific texture packing limits (e.g. RGBA packing overhead)
- [x] Profile padding limits for `Conv2d` implicitly required by certain ML libraries
- [x] Account for padding required by `int4` block unpacking logic in memory sizes
- [x] Analyze execution overhead of implicit dimension broadcasts (broadcasting `[1,3,1,1]` to `[N,3,H,W]`)
- [x] Analyze FLOP overhead of `Expand` when explicitly applied
- [x] Detect implicit padding penalties for `MatMul` blocks (e.g. `K=1023` -> `1024` on TensorCores)
- [x] Profile explicit memory usage of attention masks inside `MultiHeadAttention`
- [x] Test symbolic inference engine limits natively via stress testing
- [x] Unit Test: Profile `ResNet` family with explicit memory-reuse
- [x] Unit Test: Profile `VGG` family without memory-reuse
- [x] Unit Test: Profile `YOLO` family (dynamic bounds tracking)
- [x] Unit Test: Profile `MobileNet` family (depthwise separation math)
- [x] Unit Test: Profile `Transformer` (GPT-2 style dynamic sequence limits)
- [x] Unit Test: Verify `Einsum` equations are parsed perfectly for FLOP extraction
