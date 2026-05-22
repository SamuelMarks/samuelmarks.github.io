# ONNX Runtime (Native Exec) Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of the core `ONNX Runtime` (ORT) native execution engine within the `onnx9000` ecosystem.
Unlike the original Microsoft ORT project, which compiles to a massive 150MB+ C++ shared library depending heavily on Protobuf and CMake, our `onnx9000` engine is written in pure Python. It acts as a lightweight dynamic dispatcher. Instead of bundling math libraries, it parses the ONNX graph into memory and natively dispatches tensor operations via `ctypes` directly to the host OS's accelerated libraries (e.g., Apple Accelerate on macOS, OpenBLAS on Linux, or cuBLAS for CUDA). This achieves native C-level performance with instant, zero-build startup and a microscopic disk footprint.

## Exhaustive Parity Checklist

### 1. Core Session API & Environment (35+ items)

- [xx] Implement `InferenceSession` class
- [xx] Implement `SessionOptions` class
- [xx] Implement `RunOptions` class
- [xx] Support loading models from file path (`InferenceSession(path)`)
- [xx] Support loading models from raw bytes (`InferenceSession(bytes)`)
- [xx] Implement `InferenceSession.run(output_names, input_feed)`
- [xx] Implement `InferenceSession.get_inputs()` returning `NodeArg` abstractions
- [xx] Implement `InferenceSession.get_outputs()` returning `NodeArg` abstractions
- [xx] Implement `InferenceSession.get_overridable_initializers()`
- [xx] Implement `InferenceSession.get_providers()`
- [xx] Implement `InferenceSession.get_provider_options()`
- [xx] Implement `InferenceSession.set_providers()` (dynamic Execution Provider fallback)
- [xx] Support `SessionOptions.graph_optimization_level` (`ORT_DISABLE_ALL`)
- [xx] Support `SessionOptions.graph_optimization_level` (`ORT_ENABLE_BASIC`)
- [xx] Support `SessionOptions.graph_optimization_level` (`ORT_ENABLE_EXTENDED`)
- [xx] Support `SessionOptions.graph_optimization_level` (`ORT_ENABLE_ALL`)
- [xx] Support `SessionOptions.optimized_model_filepath` (dumping optimized graph)
- [xx] Support `SessionOptions.enable_profiling`
- [xx] Support `SessionOptions.profile_file_prefix`
- [xx] Support `SessionOptions.execution_mode` (`ORT_SEQUENTIAL`)
- [xx] Support `SessionOptions.execution_mode` (`ORT_PARALLEL`)
- [xx] Support `SessionOptions.inter_op_num_threads`
- [xx] Support `SessionOptions.intra_op_num_threads`
- [xx] Support `SessionOptions.log_severity_level`
- [xx] Support `SessionOptions.logid`
- [xx] Support `SessionOptions.add_session_config_entry`
- [xx] Support `SessionOptions.register_custom_ops_library`
- [xx] Implement `RunOptions.log_severity_level`
- [xx] Implement `RunOptions.log_verbosity_level`
- [xx] Implement `RunOptions.logid`
- [xx] Implement `RunOptions.run_tag`
- [xx] Implement `RunOptions.terminate` (ability to cancel a running execution)
- [xx] Implement `RunOptions.only_execute_path_to_fetches`
- [xx] Implement environment initialization (`Environment` singleton parity)
- [xx] Expose native `get_device()` capabilities

### 2. Native Memory Management & I/O Binding (25+ items)

- [xx] Implement `OrtValue` / `onnx9000.Tensor` abstraction
- [xx] Implement `IOBinding` class for pre-allocated memory handling
- [xx] Support `IOBinding.bind_input()`
- [xx] Support `IOBinding.bind_output()`
- [xx] Support `IOBinding.bind_ortvalue_input()`
- [xx] Support `IOBinding.bind_ortvalue_output()`
- [xx] Support `IOBinding.synchronize_inputs()`
- [xx] Support `IOBinding.synchronize_outputs()`
- [xx] Support execution via `InferenceSession.run_with_iobinding()`
- [xx] Implement static memory arena planning for inference (pre-allocating max bounds)
- [xx] Eliminate dynamic memory allocation during `InferenceSession.run()` (except for dynamic ops)
- [xx] Provide explicit memory reuse (buffer sharing) across disjoint topological paths
- [xx] Implement zero-copy mapping of NumPy `ndarray` to `OrtValue`
- [xx] Implement explicit PyTorch tensor zero-copy mapping (via `__dlpack__`)
- [xx] Implement explicit JAX tensor zero-copy mapping (via `__dlpack__`)
- [xx] Map PyTorch CUDA pointers natively to ONNX CUDA Execution Provider buffers
- [xx] Support explicit device allocator configurations
- [xx] Catch Out-Of-Memory (OOM) explicitly and surface python `MemoryError`
- [xx] Ensure proper garbage collection of intermediate tensors in pure Python
- [xx] Provide explicit memory offset calculations for contiguous blob generation
- [xx] Support pre-allocating strings and sequence tensors safely
- [xx] Support memory-mapped (mmap) weights loading specifically for massive `Initializers`
- [xx] Extract memory layout metadata cleanly
- [xx] Support passing raw ctypes pointers directly as inputs
- [xx] Generate raw ctypes pointers directly from outputs without copying

### 3. Execution Provider (EP) Abstraction & Routing (20+ items)

- [xx] Implement `ExecutionProvider` base class interface
- [xx] Implement Node partitioning algorithm based on EP capabilities
- [xx] Support fallback cascading (e.g. `CUDAExecutionProvider` -> `CPUExecutionProvider`)
- [xx] Insert memory copy nodes (`MemcpyToHost`, `MemcpyToDevice`) automatically at EP boundaries
- [xx] Support querying specific Node support per EP dynamically
- [xx] Manage device-specific execution streams (e.g., CUDA streams)
- [xx] Expose EP options strictly formatted as dictionaries
- [xx] Implement pure-Python fallback CPU EP (`CPUExecutionProvider`)
- [xx] Implement `AccelerateExecutionProvider` (macOS native BLAS/vDSP)
- [xx] Implement `CUDAExecutionProvider` (Linux/Windows via `ctypes` cuBLAS/cuDNN)
- [xx] Support `OpenVINOExecutionProvider` (via OpenVINO python bindings)
- [xx] Support `TensorrtExecutionProvider` (via TensorRT python APIs)
- [xx] Ensure deterministic graph partitioning identical to standard ORT
- [xx] Highlight un-supported subgraphs forcing a fallback to CPU visually
- [xx] Inject `Cast` nodes implicitly if EP requires specific precisions (e.g. FP16 TensorCores)
- [xx] Allow users to explicitly disable the CPU fallback (forcing an error on missing ops)
- [xx] Validate topological order after EP sub-graph partitioning
- [xx] Merge adjacent nodes belonging to the same EP into continuous Execution Subgraphs
- [xx] Trace latency across EP boundaries to profile memory transfer bottlenecks
- [xx] Abstract synchronous vs asynchronous EP execution cleanly

### 4. Math, Logical & Reduction Operators Parity (45+ items)

- [xx] Dispatch `Add` (broadcasting)
- [xx] Dispatch `Sub` (broadcasting)
- [xx] Dispatch `Mul` (broadcasting)
- [xx] Dispatch `Div` (broadcasting)
- [xx] Dispatch `MatMul` (2D, 3D, ND batched)
- [xx] Dispatch `Gemm` (with alpha, beta, transA, transB)
- [xx] Dispatch `Abs`
- [xx] Dispatch `Acos`, `Acosh`
- [xx] Dispatch `Asin`, `Asinh`
- [xx] Dispatch `Atan`, `Atanh`
- [xx] Dispatch `Cos`, `Cosh`
- [xx] Dispatch `Sin`, `Sinh`
- [xx] Dispatch `Tan`, `Tanh`
- [xx] Dispatch `Ceil`
- [xx] Dispatch `Floor`
- [xx] Dispatch `Round`
- [xx] Dispatch `Clip`
- [xx] Dispatch `Exp`
- [xx] Dispatch `Log`
- [xx] Dispatch `Pow`
- [xx] Dispatch `Sqrt`
- [xx] Dispatch `Erf`
- [xx] Dispatch `Sign`
- [xx] Dispatch `Mod`
- [xx] Dispatch `IsInf`, `IsNaN`
- [xx] Dispatch `Equal`
- [xx] Dispatch `Greater`, `GreaterOrEqual`
- [xx] Dispatch `Less`, `LessOrEqual`
- [xx] Dispatch `And`, `Or`, `Not`, `Xor`
- [xx] Dispatch `BitShift`, `BitwiseAnd`, `BitwiseNot`, `BitwiseOr`, `BitwiseXor`
- [xx] Dispatch `ReduceMax`
- [xx] Dispatch `ReduceMin`
- [xx] Dispatch `ReduceMean`
- [xx] Dispatch `ReduceSum`
- [xx] Dispatch `ReduceProd`
- [xx] Dispatch `ReduceL1`
- [xx] Dispatch `ReduceL2`
- [xx] Dispatch `ReduceLogSum`
- [xx] Dispatch `ReduceLogSumExp`
- [xx] Dispatch `ReduceSumSquare`
- [xx] Dispatch `Einsum` (Native loop translation or fallback to np.einsum)
- [xx] Dispatch `Cast` (strictly typed array conversions)
- [xx] Dispatch `CastLike`

### 5. Neural Network Layers & Activations Parity (40+ items)

- [xx] Dispatch `Conv` (1D, 2D, 3D)
- [xx] Dispatch `Conv` with dilations and grouped convolutions
- [xx] Dispatch `ConvTranspose`
- [xx] Dispatch `MaxPool` (1D, 2D, 3D)
- [xx] Dispatch `AveragePool` (1D, 2D, 3D)
- [xx] Dispatch `GlobalAveragePool`
- [xx] Dispatch `GlobalMaxPool`
- [xx] Dispatch `GlobalLpPool`
- [xx] Dispatch `MaxRoiPool`
- [xx] Dispatch `RoiAlign`
- [xx] Dispatch `BatchNormalization` (Inference mode)
- [xx] Dispatch `LayerNormalization`
- [xx] Dispatch `InstanceNormalization`
- [xx] Dispatch `LRN` (Local Response Normalization)
- [xx] Dispatch `Relu`
- [xx] Dispatch `LeakyRelu`
- [xx] Dispatch `PRelu`
- [xx] Dispatch `Elu`
- [xx] Dispatch `Selu`
- [xx] Dispatch `Sigmoid`
- [xx] Dispatch `HardSigmoid`
- [xx] Dispatch `Softmax`
- [xx] Dispatch `LogSoftmax`
- [xx] Dispatch `Softplus`
- [xx] Dispatch `Softsign`
- [xx] Dispatch `Hardmax`
- [xx] Dispatch `HardSwish`
- [xx] Dispatch `Mish`
- [xx] Dispatch `Gelu` (Tanh and Erf variants)
- [xx] Dispatch `Shrink`
- [xx] Dispatch `Dropout` (No-op in inference mode)
- [xx] Dispatch `RNN` (Dynamic unrolling)
- [xx] Dispatch `LSTM` (Dynamic unrolling)
- [xx] Dispatch `GRU` (Dynamic unrolling)
- [xx] Dispatch `GridSample`
- [xx] Dispatch `Pad` (Constant, Reflect, Edge)
- [xx] Dispatch `Resize` (Nearest, Linear, Cubic)
- [xx] Dispatch `SpaceToDepth`
- [xx] Dispatch `DepthToSpace`

### 6. Tensor Manipulation & Shape Operators Parity (35+ items)

- [xx] Dispatch `Reshape` (Zero-copy logical remap)
- [xx] Dispatch `Transpose` (Physical memory remap or logical stride adjustment)
- [xx] Dispatch `Flatten` (Zero-copy)
- [xx] Dispatch `Squeeze` (Zero-copy)
- [xx] Dispatch `Unsqueeze` (Zero-copy)
- [xx] Dispatch `Concat` (N-Dimensional)
- [xx] Dispatch `Split`
- [xx] Dispatch `Slice` (Handling dynamic steps and bounds)
- [xx] Dispatch `Gather`
- [xx] Dispatch `GatherElements`
- [xx] Dispatch `GatherND`
- [xx] Dispatch `Scatter` / `ScatterElements`
- [xx] Dispatch `ScatterND`
- [xx] Dispatch `ConstantOfShape` (Memset)
- [xx] Dispatch `Tile`
- [xx] Dispatch `Expand` (Broadcasting scalar/vector to tensor)
- [xx] Dispatch `Shape` (Extract dimensions)
- [xx] Dispatch `Size` (Extract volume)
- [xx] Dispatch `NonZero` (Dynamic memory bounds)
- [xx] Dispatch `Where` (Condition based routing)
- [xx] Dispatch `TopK`
- [xx] Dispatch `Unique`
- [xx] Dispatch `CumSum`
- [xx] Dispatch `ReverseSequence`
- [xx] Dispatch `Compress`
- [xx] Dispatch `Trilu`
- [xx] Dispatch `Col2Im`
- [xx] Dispatch `SequenceConstruct`
- [xx] Dispatch `SequenceAt`
- [xx] Dispatch `SequenceEmpty`
- [xx] Dispatch `SequenceErase`
- [xx] Dispatch `SequenceInsert`
- [xx] Dispatch `SequenceLength`
- [xx] Dispatch `SplitToSequence`
- [xx] Dispatch `ConcatFromSequence`

### 7. Control Flow & Dynamic Execution Parity (15+ items)

- [xx] Dispatch `If` (Sub-graph conditional execution)
- [xx] Manage isolated execution contexts for `If` branch bodies
- [xx] Dispatch `Loop` (Standard dynamic loop execution)
- [xx] Track iteration variables and state variables implicitly during `Loop`
- [xx] Dispatch `Scan` (Sequence loop over input tensors)
- [xx] Implement `Scan` state concatenation optimizations
- [xx] Support nested sub-graphs gracefully (e.g. `Loop` inside an `If`)
- [xx] Share static Constants from the parent graph into the sub-graph context seamlessly
- [xx] Ensure sub-graph output shapes are reconciled statically if possible
- [xx] Enforce max iteration limits dynamically for `Loop` to prevent infinite hangs
- [xx] Prevent memory leaks inside deeply nested dynamic sub-graphs
- [xx] Support Execution Provider transitions between Parent and Sub-graph
- [xx] Profile Sub-graph node execution times independently
- [xx] Map Sub-graph `Yield` values to final Output Tensors correctly
- [xx] Validate `If` branch output types match exactly at runtime

### 8. Accelerate Execution Provider (Apple macOS) (20+ items)

- [xx] Load `Accelerate.framework` via `ctypes.util.find_library`
- [xx] Bind `cblas_sgemm` directly for Float32 `MatMul` / `Gemm`
- [xx] Bind `cblas_dgemm` directly for Float64 `MatMul` / `Gemm`
- [xx] Handle `transA` and `transB` via CBLAS transposition flags natively
- [xx] Bind `vDSP_vadd` for vectorized Float32 `Add`
- [xx] Bind `vDSP_vmul` for vectorized Float32 `Mul`
- [xx] Bind `vDSP_vdiv` for vectorized Float32 `Div`
- [xx] Bind `vforce_vexp` for vectorized `Exp`
- [xx] Bind `vforce_vlog` for vectorized `Log`
- [xx] Bind `vDSP_maxv` for `ReduceMax`
- [xx] Bind `vDSP_minv` for `ReduceMin`
- [xx] Bind `vDSP_sve` for `ReduceSum`
- [xx] Prevent memory copies by mapping NumPy `.ctypes.data_as` directly into `cblas`/`vDSP`
- [xx] Provide multi-threading controls native to Apple Accelerate (via GCD / env vars)
- [xx] Validate stride alignments specifically for macOS ARM64 (Apple Silicon) requirements
- [xx] Check matrix size boundaries for `cblas` integers (preventing 32-bit overflow)
- [xx] Fallback to pure Python loops if dimensions do not meet `cblas` expectations
- [xx] Trace `Accelerate` provider execution latency cleanly
- [xx] Unit Test: Verify BLAS MatMul outputs match NumPy precisely
- [xx] Integrate AMX (Apple Matrix Coprocessor) hidden instructions automatically via Accelerate

### 9. CUDA Execution Provider (NVIDIA) (30+ items)

- [xx] Load `libcublas.so` / `cublas.dll` dynamically via `ctypes`
- [xx] Load `libcudart.so` / `cudart.dll` dynamically via `ctypes`
- [xx] Implement GPU memory allocation (`cudaMalloc`, `cudaFree`) via `ctypes`
- [xx] Implement CPU <-> GPU memory transfers (`cudaMemcpy`)
- [xx] Map `CUDAExecutionProvider` device IDs explicitly
- [xx] Initialize `cublasCreate_v2` handle safely per session
- [xx] Bind `cublasSgemm` for Float32 `MatMul`
- [xx] Bind `cublasHgemm` for Float16 `MatMul`
- [xx] Bind `cublasDgemm` for Float64 `MatMul`
- [xx] Bind `cublasSgemmStridedBatched` for 3D/ND Batched `MatMul`
- [xx] Handle column-major (cuBLAS) vs row-major (ONNX) transposition mappings automatically
- [xx] Load `libcudnn.so` / `cudnn.dll` dynamically
- [xx] Initialize `cudnnCreate` handle safely
- [xx] Bind `cudnnConvolutionForward` for highly optimized CNN execution
- [xx] Implement `cudnnSetTensorNdDescriptor` logic natively in Python
- [xx] Handle CUDA streams (`cudaStreamCreate`, `cudaStreamSynchronize`)
- [xx] Enable Tensor Cores explicitly via `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)`
- [xx] Execute `CUDA` operations asynchronously, synchronizing only on `MemcpyToHost`
- [xx] Catch `cudaError_t` error codes and throw explicitly as Python exceptions
- [xx] Manage a static CUDA memory arena natively in Python to avoid `cudaMalloc` overhead
- [xx] Free CUDA memory cleanly upon `InferenceSession` garbage collection
- [xx] Provide zero-copy DLPack mapping directly to `torch.cuda.Tensor`
- [xx] Support `CUDAExecutionProvider` arena extension options
- [xx] Support `CUDAExecutionProvider` explicit workspace size limits (for cuDNN algorithms)
- [xx] Evaluate cuDNN heuristic search (`cudnnGetConvolutionForwardAlgorithm_v7`) for optimal CNN speeds
- [xx] Bind vectorized math ops via cuBLAS (e.g., `cublasSscal` for scaling)
- [xx] Write minimal custom `.ptx` kernels for unsupported ops and load via PyCUDA/CuPy natively
- [xx] Support Numba CUDA JIT kernels as a dynamic fallback for specific layer fusions
- [xx] Warn user cleanly if CUDA driver / libraries are completely missing
- [xx] Test FP16 evaluation numerical stability across Tensor Cores

### 10. Graph Optimizations (Level 1, Level 2, Level 3) (30+ items)

- [xx] Level 1: Implement `ConstantFolding` (Pre-calculating math ops)
- [xx] Level 1: Implement `DeadCodeElimination` (Removing unused nodes)
- [xx] Level 1: Implement `IdentityElimination` (Removing no-ops)
- [xx] Level 1: Implement `CastElimination` (Removing redundant casts)
- [xx] Level 1: Implement `DropoutElimination` (Bypassing dropouts)
- [xx] Level 1: Implement `SliceElimination` (Redundant slice drops)
- [xx] Level 1: Implement `SqueezeElimination`
- [xx] Level 1: Implement `UnsqueezeElimination`
- [xx] Level 2: Implement `Conv` + `BatchNormalization` Fusion
- [xx] Level 2: Implement `Conv` + `Add` (Bias) Fusion
- [xx] Level 2: Implement `MatMul` + `Add` -> `Gemm` Fusion
- [xx] Level 2: Implement `Conv` + `Relu` Fusion (Using cuDNN / OpenVINO specialized fusions)
- [xx] Level 2: Implement `Gemm` + `Relu` Fusion
- [xx] Level 2: Implement `Reshape` + `Reshape` cancellations
- [xx] Level 2: Implement `Transpose` + `Transpose` cancellations
- [xx] Level 2: Implement `Gelu` pattern matching and fusion (Erf approximation)
- [xx] Level 2: Implement `LayerNormalization` pattern matching and fusion
- [xx] Level 3: Implement `MultiHeadAttention` Fusion (Transformer optimization)
- [xx] Level 3: Implement `FastGelu` Fusion
- [xx] Level 3: Implement `SkipLayerNormalization` Fusion
- [xx] Level 3: Implement `EmbedLayerNormalization` Fusion
- [xx] Level 3: Implement `RotaryPositionalEmbedding` (RoPE) Fusion
- [xx] Layout Transformer: Optimize `NCHW` to `NHWC` selectively based on Execution Provider
- [xx] Apply symbolic shape inference iteratively between optimization passes
- [xx] Expose explicitly disabled optimizers (`SessionOptions.add_free_dimension_override_by_name`)
- [xx] Expose an API to serialize the completely optimized graph to `.onnx` for debugging
- [xx] Support offline optimization mode (optimize and exit without initializing execution providers)
- [xx] Generate detailed optimization log reports
- [xx] Validate topological graph integrity after every single optimization pass
- [xx] Ensure optimizations do not change the ultimate output shape/type semantics

### 11. Testing, Profiling & Opset Compliance (25+ items)

- [xx] Execute `onnx` repository standard backend tests: Opset 7
- [xx] Execute `onnx` repository standard backend tests: Opset 8
- [xx] Execute `onnx` repository standard backend tests: Opset 9
- [xx] Execute `onnx` repository standard backend tests: Opset 10
- [xx] Execute `onnx` repository standard backend tests: Opset 11
- [xx] Execute `onnx` repository standard backend tests: Opset 12
- [xx] Execute `onnx` repository standard backend tests: Opset 13
- [xx] Execute `onnx` repository standard backend tests: Opset 14
- [xx] Execute `onnx` repository standard backend tests: Opset 15
- [xx] Execute `onnx` repository standard backend tests: Opset 16
- [xx] Execute `onnx` repository standard backend tests: Opset 17
- [xx] Execute `onnx` repository standard backend tests: Opset 18
- [xx] Execute `onnx` repository standard backend tests: Opset 19
- [xx] Execute `onnx` repository standard backend tests: Opset 20
- [xx] Execute `onnx` repository standard backend tests: Opset 21
- [xx] Generate exact ONNX Runtime compatible trace profiles (`.json` Chrome Trace format)
- [xx] Track Node execution start time, end time, and thread ID natively
- [xx] Track Execution Provider boundary overheads natively
- [xx] Ensure 100% test coverage of all pure-Python dispatch handlers
- [xx] Compare `onnx9000` execution latency against native `onnxruntime` C++ (target: within 5% via ctypes overhead)
- [xx] Compare `onnx9000` memory allocation footprint against native `onnxruntime` (target: lower peak RAM)
- [xx] Benchmark: ResNet50 Inference (Batch 1, Float32)
- [xx] Benchmark: BERT-Base Inference (Batch 1, Float32)
- [xx] Provide detailed mismatch exception logs containing exact tensor deltas when tests fail
- [xx] Fallback to simple NumPy evaluation explicitly for ops lacking optimized ctypes bindings
