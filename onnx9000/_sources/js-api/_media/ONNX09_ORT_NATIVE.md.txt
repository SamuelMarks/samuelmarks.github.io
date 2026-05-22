# ORT Native Exec & FFI Dispatcher Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of the native hardware Execution Providers (EP) originally found in `onnxruntime`, within the `onnx9000` ecosystem.
Instead of relying on a massive C++ framework to coordinate hardware execution, `onnx9000` acts as a zero-overhead pure-Python dynamic dispatcher. It maps ONNX `Tensor` structures (backed by zero-copy DLPack or NumPy contiguous memory) directly to OS-native hardware libraries via `ctypes` and `cffi`.
This covers the exhaustive integration of **Apple Accelerate (vDSP/BNNS)**, **Apple Metal (MPS)**, **NVIDIA CUDA (cuBLAS/cuDNN)**, and **OpenBLAS/MKL** natively from Python. This architecture allows an ONNX model to execute at native C speeds without requiring any C++ compilation or `onnxruntime` installations on the host machine.

## Exhaustive Parity Checklist

### 1. Core FFI, CTypes, & Dynamic Library Orchestration (35+ items)

- [x] Implement generic `DynamicLibrary` loader for POSIX (`dlopen`, `dlsym`)
- [x] Implement generic `DynamicLibrary` loader for Windows (`LoadLibrary`, `GetProcAddress`)
- [x] Extract library paths dynamically using `ctypes.util.find_library`
- [x] Implement `RTLD_NOW` and `RTLD_GLOBAL` bindings for proper symbol resolution
- [x] Catch `OSError` cleanly when a hardware library is missing
- [x] Fallback gracefully between library versions (e.g. `libcudart.so.12` -> `libcudart.so.11`)
- [x] Define explicit C function signatures (`argtypes`, `restype`) for safety
- [x] Implement C-struct mappings natively in Python for hardware context handles
- [x] Prevent Python Garbage Collection on active C-pointers securely
- [x] Expose native pointers from Python integers (`ctypes.c_void_p`)
- [x] Abstract synchronous vs asynchronous hardware execution boundaries
- [x] Expose `dlerror()` strings natively to Python for debugging
- [x] Handle explicit Windows Calling Conventions (`__stdcall` vs `__cdecl`) securely
- [x] Verify host architecture dynamically (`x86_64`, `aarch64`, `arm64`)
- [x] Verify host OS dynamically (`Darwin`, `Linux`, `Windows`)
- [x] Wrap all FFI calls in minimal-overhead Python decorators
- [x] Optimize inner-loop FFI calls by caching `getattr` lookups
- [x] Provide thread-safe FFI invocations (releasing Python GIL if supported)
- [x] Support `cffi` ABI mode explicitly for lower overhead than `ctypes` where available
- [x] Manage Hardware Context handles (e.g. `cublasHandle_t`) safely across Threads
- [x] Destroy Hardware Context handles explicitly on `__del__` or Context Manager exit
- [x] Expose explicitly hardware-specific error codes mapped to Python `RuntimeError`
- [x] Verify hardware alignment constraints (e.g. 16-byte, 32-byte bounds) natively
- [x] Prevent illegal memory access (Segfaults) by bounds-checking Python array sizes before FFI dispatch
- [x] Map Python strings to `const char*` C-strings seamlessly
- [x] Map Python `bool` to C `int` natively
- [x] Emulate C++ `std::vector` contiguous data payloads across the C boundary
- [x] Provide an abstraction to inject user-provided shared libraries dynamically (`.so` / `.dll`)
- [x] Ensure 100% pure Python execution if NO hardware libraries are found (graceful fallback)
- [x] Provide exhaustive verbose logging of FFI load times
- [x] Isolate FFI imports to prevent crashing the main module on unsupported hardware
- [x] Profile `ctypes` call overhead to ensure it remains < 1 microsecond per op
- [x] Execute `sysctl` or `machdep` natively on macOS to query specific CPU features
- [x] Query `/proc/cpuinfo` on Linux natively to identify AVX/AVX2/AVX512/NEON extensions
- [x] Identify L1/L2/L3 Cache sizes natively to optimize BLAS tiling logic

### 2. Zero-Copy Memory & DLPack Integration (40+ items)

- [x] Implement strict `__dlpack__` protocol for `onnx9000.Tensor`
- [x] Implement strict `__dlpack_device__` protocol
- [x] Consume PyTorch `torch.Tensor` directly via DLPack (Zero-copy)
- [x] Consume JAX `jax.Array` directly via DLPack (Zero-copy)
- [x] Consume TensorFlow `tf.Tensor` directly via DLPack (Zero-copy)
- [x] Consume NumPy `np.ndarray` directly via `__array_interface__` (Zero-copy)
- [x] Ensure consumed DLPack tensors retain their original memory backend (CPU vs GPU)
- [x] Extract raw memory pointers (`data_ptr`) strictly natively
- [x] Extract memory strides strictly natively
- [x] Extract data types (`DLDataType` struct) strictly natively
- [x] Handle `kDLCPU` device natively
- [x] Handle `kDLCUDA` device natively
- [x] Handle `kDLCUDAManaged` (Pinned) device natively
- [x] Handle `kDLMetal` device natively
- [x] Handle `kDLROCM` device natively
- [x] Throw exceptions natively if memory is not `C_CONTIGUOUS` and operation requires it
- [x] Auto-generate contiguous copies if memory is `F_CONTIGUOUS` and operation requires C-layout
- [x] Support explicit CUDA Pinned Memory allocation (`cudaMallocHost`) natively in Python
- [x] Support explicit Unified Memory allocation (`cudaMallocManaged`) natively
- [x] Implement OS-level Page-Locked memory via `mlock` natively in Python
- [x] Allocate large memory arenas explicitly using anonymous `mmap` (`MAP_ANONYMOUS | MAP_PRIVATE`)
- [x] Ensure `madvise` (`MADV_HUGEPAGE`) is used for massive intermediate tensors on Linux
- [x] Guarantee 64-byte alignment explicitly during arena allocations for AVX512/GPU
- [x] Track memory views explicitly (slices of tensors do not reallocate memory)
- [x] Implement robust Reference Counting for memory arenas shared across multiple tensors
- [x] Support overlapping memory boundaries securely (In-place operations)
- [x] Prevent Race Conditions when multiple threads access the same Memory Arena
- [x] Provide `tensor.to(device)` mapping allocating memory directly on the target hardware
- [x] Transfer memory CPU -> GPU implicitly if Execution Provider mandates it
- [x] Transfer memory GPU -> CPU implicitly if Execution Provider mandates it
- [x] Track explicit bandwidth metrics (Bytes/sec) during CPU <-> GPU transfers
- [x] Support Asynchronous memory transfers (`cudaMemcpyAsync`) via explicit streams
- [x] Provide explicit memory barrier primitives (`cudaStreamSynchronize`)
- [x] Handle specific Apple Silicon Unified Memory architectures (Zero-copy CPU <-> GPU)
- [x] Extract memory pointers from WebAssembly (`Module.HEAPU8`) directly natively
- [x] Expose native OS memory metrics to the user (e.g. Current VRAM usage)
- [x] Trap Out-Of-Memory (OOM) explicitly and surface Python `MemoryError`
- [x] Guarantee safe cleanup of unmanaged C pointers via `weakref.finalize`
- [x] Handle Pyodide proxy objects converting to contiguous C pointers natively
- [x] Support explicit INT4 memory packing and pointer resolution natively

### 3. Apple Accelerate (vDSP & BNNS) Backend (40+ items)

- [x] Detect `Accelerate.framework` dynamically on macOS
- [x] Load `cblas` directly from Accelerate
- [x] Implement `MatMul` -> `cblas_sgemm` (Float32)
- [x] Implement `MatMul` -> `cblas_dgemm` (Float64)
- [x] Ensure `cblas` transposition arguments (`CblasNoTrans`, `CblasTrans`) correctly map ONNX layouts
- [x] Load `vDSP` directly from Accelerate
- [x] Implement `Add` -> `vDSP_vadd`
- [x] Implement `Sub` -> `vDSP_vsub`
- [x] Implement `Mul` -> `vDSP_vmul`
- [x] Implement `Div` -> `vDSP_vdiv`
- [x] Implement `Abs` -> `vDSP_vabs`
- [x] Implement `Neg` -> `vDSP_vneg`
- [x] Implement `Clip` -> `vDSP_vclip`
- [x] Implement `ReduceMax` -> `vDSP_maxv`
- [x] Implement `ReduceMin` -> `vDSP_minv`
- [x] Implement `ReduceSum` -> `vDSP_sve`
- [x] Implement `ReduceMean` -> `vDSP_meanv`
- [x] Implement Vector-Scalar Add -> `vDSP_vsadd`
- [x] Implement Vector-Scalar Mul -> `vDSP_vsmul`
- [x] Implement Vector-Scalar Div -> `vDSP_vsdiv`
- [x] Implement `Exp` -> `vforce_vexp`
- [x] Implement `Log` -> `vforce_vlog`
- [x] Implement `Sqrt` -> `vforce_vsqrt`
- [x] Implement `Sin` -> `vforce_vsin`
- [x] Implement `Cos` -> `vforce_vcos`
- [x] Implement `Tan` -> `vforce_vtan`
- [x] Implement `Tanh` -> `vforce_vtanh`
- [x] Load `BNNS` (Basic Neural Network Subroutines) dynamically from Accelerate
- [x] Implement `Conv` -> `BNNSFilterApply` (Convolution)
- [x] Implement `MaxPool` -> `BNNSFilterApply` (Pooling)
- [x] Implement `AveragePool` -> `BNNSFilterApply` (Pooling)
- [x] Implement `BatchNormalization` -> `BNNSFilterApply` (Normalization)
- [x] Implement `Gelu` -> `BNNSFilterApply` (Activation)
- [x] Implement `Softmax` -> `BNNSFilterApply` (Activation)
- [x] Manage BNNS filter creation (`BNNSFilterCreate...`) and caching dynamically
- [x] Release BNNS filters natively to prevent memory leaks (`BNNSFilterDestroy`)
- [x] Expose native AMX (Apple Matrix Coprocessor) instructions via Accelerate implicitly
- [x] Validate 32-byte alignment implicitly required by certain vDSP instructions
- [x] Provide pure Python loop fallback if vDSP length constraints are violated
- [x] Profile the latency of Accelerate framework calls natively

### 4. Apple Metal Performance Shaders (MPS) Backend (40+ items)

- [x] Detect `Metal.framework` dynamically
- [x] Detect `MetalPerformanceShaders.framework` (MPS) dynamically
- [x] Expose PyObjC bridging natively using pure `ctypes` (allocating `NSObject` securely)
- [x] Request `MTLCreateSystemDefaultDevice` natively
- [x] Create `MTLCommandQueue` natively
- [x] Create `MTLCommandBuffer` natively per execution step
- [x] Map `onnx9000.Tensor` to `MTLBuffer` using `newBufferWithBytesNoCopy` for zero-overhead
- [x] Map `MTLBuffer` to `onnx9000.Tensor` using `contents` securely
- [x] Sync `MTLCommandBuffer` using `waitUntilCompleted` natively
- [x] Initialize `MPSGraph` natively
- [x] Implement `Add` -> `[MPSGraph additionWithPrimaryTensor:...]`
- [x] Implement `Sub` -> `[MPSGraph subtractionWithPrimaryTensor:...]`
- [x] Implement `Mul` -> `[MPSGraph multiplicationWithPrimaryTensor:...]`
- [x] Implement `Div` -> `[MPSGraph divisionWithPrimaryTensor:...]`
- [x] Implement `MatMul` -> `[MPSGraph matrixMultiplicationWithPrimaryTensor:...]`
- [x] Implement `Conv` -> `[MPSGraph convolution2DWithSourceTensor:...]`
- [x] Map standard ONNX `pads`, `strides`, `dilations` to MPS `MPSGraphConvolution2DOpDescriptor`
- [x] Implement `MaxPool` -> `[MPSGraph maxPooling2DWithSourceTensor:...]`
- [x] Implement `AveragePool` -> `[MPSGraph avgPooling2DWithSourceTensor:...]`
- [x] Implement `Relu` -> `[MPSGraph reLUWithTensor:...]`
- [x] Implement `Gelu` -> `[MPSGraph geLUWithTensor:...]`
- [x] Implement `Softmax` -> `[MPSGraph softMaxWithTensor:...]`
- [x] Implement `ReduceSum` -> `[MPSGraph reductionSumWithTensor:...]`
- [x] Implement `ReduceMean` -> `[MPSGraph reductionMeanWithTensor:...]`
- [x] Implement `Reshape` -> `[MPSGraph reshapeTensor:...]`
- [x] Implement `Transpose` -> `[MPSGraph transposeTensor:...]`
- [x] Compile `MPSGraph` to `MPSGraphExecutable` statically to save runtime latency
- [x] Cache `MPSGraphExecutable` natively keyed by input shape signatures
- [x] Handle dynamic shapes inside `MPSGraph` gracefully
- [x] Provide explicit FP16 Metal execution (MPS runs natively in FP16 if requested)
- [x] Fallback unsupported operations from Metal back to Accelerate (CPU) seamlessly
- [x] Sync data explicitly between `MTLBuffer` and CPU Ram across Execution Provider boundaries
- [x] Extract GPU profiling latency natively using `MTLEvent` or `GPUEndTime` metrics
- [x] Manage `NSAutoreleasePool` natively in Python to prevent macOS memory leaks
- [x] Catch Metal validation layer errors gracefully and print them to Python stdout
- [x] Implement Custom Metal Shaders (MSL) compiled via `MTLLibrary` natively for unsupported Ops
- [x] Dispatch arbitrary MSL Compute Pipelines natively (`MTLComputeCommandEncoder`)
- [x] Set `threadgroupsPerGrid` and `threadsPerThreadgroup` perfectly for custom MSL
- [x] Inject `Cast` operations explicitly if MPS refuses to process specific data types (e.g. Int64)
- [x] Validate Apple Silicon (M1/M2/M3) unified memory bandwidth limits natively

### 5. NVIDIA CUDA Core & cuBLAS Backend (45+ items)

- [x] Detect `libcudart.so` / `cudart.dll` dynamically
- [x] Detect `libcublas.so` / `cublas.dll` dynamically
- [x] Detect `libcublasLt.so` / `cublasLt.dll` dynamically
- [x] Implement `cudaGetDeviceCount` natively
- [x] Implement `cudaSetDevice` natively
- [x] Implement `cudaMalloc` natively
- [x] Implement `cudaFree` natively
- [x] Implement `cudaMemcpy` (HostToDevice, DeviceToHost, DeviceToDevice) natively
- [x] Implement `cudaStreamCreate`, `cudaStreamDestroy`, `cudaStreamSynchronize` natively
- [x] Wrap all CUDA calls in strict error checking checking `cudaError_t` != 0
- [x] Create `cublasCreate_v2` handle securely
- [x] Destroy `cublasDestroy_v2` handle securely
- [x] Set cuBLAS streams via `cublasSetStream_v2`
- [x] Map `MatMul` (Float32) to `cublasSgemm`
- [x] Map `MatMul` (Float64) to `cublasDgemm`
- [x] Map `MatMul` (Float16) to `cublasHgemm`
- [x] Map Batched `MatMul` (Float32) to `cublasSgemmStridedBatched`
- [x] Map Batched `MatMul` (Float16) to `cublasHgemmStridedBatched`
- [x] Explicitly map ONNX Row-Major topologies to cuBLAS Column-Major APIs (using transposed flags `CUBLAS_OP_T`)
- [x] Evaluate explicitly required Workspace sizes for cuBLAS operations natively
- [x] Set cuBLAS Math Mode to `CUBLAS_TENSOR_OP_MATH` (enable Tensor Cores for FP16)
- [x] Set cuBLAS Math Mode to `CUBLAS_TF32_TENSOR_OP_MATH` (enable TF32 on Ampere+)
- [x] Implement scalar loading directly into GPU memory for cuBLAS `alpha` and `beta` values
- [x] Bind Elementwise Vector math via `cublasSscal`, `cublasSaxpy` natively
- [x] Fallback unsupported operations to CPU explicitly (triggering `cudaMemcpy` automatically)
- [x] Cache CUDA allocations via a Native Python memory arena (preventing `cudaMalloc` overhead)
- [x] Map specific `cublasLtMatmul` APIs for fused Epilogues (e.g. MatMul + Relu)
- [x] Provide explicit FP8 inference emulation via `cublasLt` if supported by Ampere/Hopper
- [x] Query GPU SM count (`cudaDeviceGetAttribute`) to optimize stream splitting
- [x] Expose native `torch.cuda.current_stream()` synchronization bridging
- [x] Ensure Multi-GPU topology execution (splitting nodes across Device 0 and Device 1 natively)
- [x] Trace latency across CUDA ops using `cudaEventRecord` natively
- [x] Load completely custom `.ptx` files natively using `cuModuleLoad`
- [x] Bind custom `.ptx` kernels to specific ONNX elements dynamically using `cuLaunchKernel`
- [x] Execute completely asynchronously, yielding to the Python event loop while the GPU computes
- [x] Resolve out-of-bounds pointer allocations safely without hard-crashing the Python process
- [x] Extract native VRAM metrics dynamically via `cudaMemGetInfo`
- [x] Emulate specific Activation functions via custom generated PTX kernels natively
- [x] Emulate specific Reductions via custom generated PTX kernels natively
- [x] Emulate `Gather` and `Scatter` using atomic operations in custom PTX
- [x] Inject explicit precision casts (`Float32` -> `Float16`) strictly on the GPU natively
- [x] Validate `Float16` numerical stability dynamically against CPU calculations (atol=1e-3)
- [x] Bind `cudaDeviceSynchronize` specifically at the end of the ONNX Graph Execution
- [x] Provide `CUDAExecutionProvider` Options natively mimicking standard ORT dictionaries
- [x] Support dynamic sizing (`-1` axes) implicitly by re-calling cuBLAS without reallocation if buffer fits

### 6. NVIDIA cuDNN & Tensor Core Backend (40+ items)

- [x] Detect `libcudnn.so` / `cudnn.dll` dynamically
- [x] Create `cudnnCreate` handle natively
- [x] Set cuDNN streams via `cudnnSetStream`
- [x] Map ONNX `Conv` explicitly to `cudnnConvolutionForward`
- [x] Create `cudnnTensorDescriptor_t` for inputs natively
- [x] Create `cudnnTensorDescriptor_t` for outputs natively
- [x] Create `cudnnFilterDescriptor_t` for weights natively
- [x] Create `cudnnConvolutionDescriptor_t` natively
- [x] Map ONNX Padding securely to cuDNN parameters
- [x] Map ONNX Strides securely to cuDNN parameters
- [x] Map ONNX Dilations securely to cuDNN parameters
- [x] Emulate Asymmetric padding natively (cuDNN only supports symmetric, must pre-pad inputs)
- [x] Map ONNX Groups (Depthwise) to cuDNN `cudnnSetConvolutionGroupCount`
- [x] Support NCHW tensor layout explicit mappings
- [x] Support NHWC tensor layout explicit mappings (Fastest on Tensor Cores)
- [x] Implement `cudnnGetConvolutionForwardAlgorithm_v7` to auto-tune the fastest convolution algorithm
- [x] Support `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM` natively
- [x] Support `CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED` natively
- [x] Set cuDNN Math Type to `CUDNN_TENSOR_OP_MATH` explicitly
- [x] Allocate specific Workspace buffers dynamically as requested by cuDNN heuristic search
- [x] Cache Algorithm selections dynamically based on Input Shapes (preventing tuning overhead)
- [x] Implement `cudnnPoolingForward` for `MaxPool`
- [x] Implement `cudnnPoolingForward` for `AveragePool`
- [x] Map ONNX Spatial parameters to `cudnnPoolingDescriptor_t`
- [x] Implement `cudnnActivationForward` for `Relu`
- [x] Implement `cudnnActivationForward` for `Sigmoid`
- [x] Implement `cudnnActivationForward` for `Tanh`
- [x] Implement `cudnnSoftmaxForward` for `Softmax`
- [x] Implement `cudnnSoftmaxForward` for `LogSoftmax`
- [x] Map `cudnnBatchNormalizationForwardInference` for `BatchNormalization`
- [x] Map `cudnnSpatialTfSamplerForward` for `GridSample` natively
- [x] Handle 1D Convolutions correctly by expanding to 2D internally for cuDNN
- [x] Handle 3D Convolutions (`cudnnConvolutionForward` ND representations)
- [x] Catch `CUDNN_STATUS_NOT_SUPPORTED` and fallback to generic PTX or CPU execution securely
- [x] Support FP16 convolutions optimally (Requires NHWC layout natively)
- [x] Clean up all cuDNN Descriptors natively upon graph garbage collection
- [x] Trace cuDNN specific kernel execution latencies via `cudaEvent` explicitly
- [x] Emulate TensorRT fusion graphs using purely cuDNN `backend` API (v8+) if available
- [x] Ensure strict error tracebacks printing the exact cuDNN failure string
- [x] Test numerical tolerance on Deep CNNs (ResNet/VGG) entirely on cuDNN

### 7. OpenBLAS & CPU Generic Execution Fallback (30+ items)

- [x] Detect `libopenblas.so` / `libopenblas.dll` dynamically
- [x] Implement fallback `MatMul` (Float32) using `cblas_sgemm`
- [x] Implement fallback `MatMul` (Float64) using `cblas_dgemm`
- [x] Set explicitly `openblas_set_num_threads` based on Session Options
- [x] Override `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` environment variables securely
- [x] Handle generic fallback CPU elementwise math natively in pure C loops (using `ctypes` loaded kernels)
- [x] Implement native C loop for `Add` (with broadcasting)
- [x] Implement native C loop for `Mul` (with broadcasting)
- [x] Implement native C loop for `Exp`, `Log`, `Softmax`
- [x] Compile the tiny C fallback kernels dynamically at runtime via standard `cc` / `gcc` if needed
- [x] Cache the tiny compiled `.so` fallback explicitly in `~/.onnx9000/cache`
- [x] Execute `ctypes` bindings on the dynamically compiled CPU Fallback kernels
- [x] Implement C-level parallelization natively (`#pragma omp parallel for`) in the fallback kernels
- [x] Support explicit mapping to Intel MKL (`libmkl_rt.so`) if OpenBLAS is not found
- [x] Call `cblas_sgemm` from Intel MKL identically
- [x] Call `vmlExp` / `vmlLn` from Intel MKL VML for vectorized transcendental math
- [x] Handle exact Memory alignment requirements for MKL AVX512 limits natively
- [x] Map explicit Threading overrides to MKL (`mkl_set_num_threads`)
- [x] Evaluate CPU execution latency exactly and compare against PyTorch CPU
- [x] Handle Float16 evaluation on CPU generically (Requires downcasting to Float32 dynamically, processing, and upcasting)
- [x] Expose `CPUExecutionProvider` natively
- [x] Enforce deterministic execution on the CPU fallback completely (fixing floating point associative issues)
- [x] Support strict N-dimensional Gather/Scatter loops purely in C arrays
- [x] Support exact slicing loops natively in C without memory copies (if strides allow)
- [x] Ensure purely synchronous execution boundaries for all CPU fallbacks
- [x] Process memory copies natively via highly optimized `memcpy`
- [x] Generate specific loop logic for CustomOps (Tokenizers/BPE) locally if purely Python is too slow
- [x] Verify execution is structurally thread-safe (calling `.run()` from multiple Python threads concurrently)
- [x] Validate multi-model multi-session execution without context collisions
- [x] Ensure standard C-libraries do not interfere with application Signal Handlers (e.g. SIGINT)

### 8. Native Hardware Profiling & Synchronization (25+ items)

- [x] Profile pure Python overhead time (Time spent in interpreter vs hardware execution)
- [x] Profile memory allocation time exactly
- [x] Profile memory transfer time (CPU -> GPU) exactly
- [x] Profile memory transfer time (GPU -> CPU) exactly
- [x] Profile Kernel dispatch time exactly
- [x] Dump metrics to standard Chrome Trace JSON formats (`chrome://tracing`)
- [x] Track peak VRAM usage natively across hardware
- [x] Track peak System RAM usage natively
- [x] Interleave Python `asyncio` loop tightly with hardware execution status polling
- [x] Support PyTorch CUDA streams natively (`torch.cuda.current_stream().cuda_stream`)
- [x] Provide explicit warm-up loops (executing the model 5 times before timing)
- [x] Pin GPU clocks natively (if running as root on Linux) to stabilize benchmark numbers
- [x] Execute exact MACs/FLOPs correlation with reported hardware latency
- [x] Identify Memory-Bound vs Compute-Bound layers directly from the trace
- [x] Highlight FFI binding overhead bottlenecks dynamically
- [x] Track WebGPU compute shader timestamps securely natively
- [x] Track WebAssembly SIMD execution latencies securely natively
- [x] Output dynamic string visualizations in the terminal showing Execution Provider assignments
- [x] Verify precise memory footprint matching the `onnx-tool` static simulations
- [x] Catch heavily unbalanced workloads and recommend specific CPU/GPU partitioning optimizations
- [x] Monitor CPU core utilization specifically during MKL / OpenBLAS execution
- [x] Catch explicitly un-fused layers causing massive Kernel dispatch overhead queues
- [x] Identify Subnormal (Denormal) floats dynamically if CPU execution grinds to a halt
- [x] Trace Garbage Collection (GC) pauses natively during the execution session
- [x] Profile execution natively within Cloudflare Worker limits (measuring wall-time cleanly)

### 9. Edge Cases, System Errors & Hardware Limitations (30+ items)

- [x] Provide explicit fallbacks if `libcudnn.so` fails to initialize properly
- [x] Manage `CUDA_ERROR_OUT_OF_MEMORY` gracefully without killing the Python process
- [x] Map PyTorch memory caching allocator states dynamically (integrating our arena into PyTorch's)
- [x] Catch `cblas` dimension limits (preventing `M * K` overflowing 32-bit `int`)
- [x] Catch Apple Accelerate `BNNS` limits natively and fallback
- [x] Fallback on specific MacOS versions gracefully if `MPSGraph` APIs are missing
- [x] Prevent Python `MemoryError` explicitly on multi-gigabyte unified memory buffers
- [x] Trace OS page faults during execution natively
- [x] Provide specific error logs indicating exactly which hardware library symbol is missing (e.g. `dlsym` failed for `cublasSgemmStridedBatched`)
- [x] Support generic CPU execution purely in Python loops if compiling C-kernels is strictly prohibited by security policies
- [x] Test the entire FFI dispatcher cleanly inside Pyodide (WASM) expecting graceful failure or mapping to WebAssembly C-imports
- [x] Validate multi-threading Python GIL releases across `ctypes` correctly function (e.g., executing two ONNX models on two threads concurrently over CPU)
- [x] Ensure execution determinism across identical hardware (no floating point non-determinism unless requested)
- [x] Unit Test: Verify `ResNet50` inference output between `onnxruntime` C++ and our native `cffi` CUDA backend exactly matches (atol=1e-5)
- [x] Unit Test: Verify `BERT` inference output between `onnxruntime` C++ and our native `cffi` Apple Accelerate backend exactly matches
- [x] Validate `Float16` (Half) execution matches exactly across CUDA TensorCores and Metal Performance Shaders
- [x] Protect against invalid ONNX topological connections fundamentally breaking the C-kernel boundaries
- [x] Output a rich hardware diagnostic log containing precise OS, Driver, and Hardware API versions on load
