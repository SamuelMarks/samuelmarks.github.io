# onnx-mlir Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `onnx-mlir` (Ahead-Of-Time Compilation) within the `onnx9000` ecosystem.
The standard `onnx-mlir` project relies on the massive LLVM compiler infrastructure and the MLIR (Multi-Level Intermediate Representation) dialect framework. Building and utilizing it requires a massive C++ toolchain.
Our `onnx9000` reimplementation completely bypasses LLVM/MLIR. Instead, it transpiles the pure-Python `core.ir` directly into highly optimized, strict C++23 source files using Jinja2 templates. This C++ can then be compiled natively via `g++`/`clang++` for zero-overhead server execution, or via `emcc` (Emscripten) into microscopic standalone WebAssembly (`.wasm`) payloads that require _zero external ML runtimes_ (like ONNX Runtime Web) to execute in the browser.

## Exhaustive Parity Checklist

### 1. Codegen Architecture & Static Memory Arena (40+ items)

- [x] Implement C++23 code generation engine using Jinja2 templates
- [x] Implement static shape resolution pass ahead of transpilation
- [x] Implement static dtype resolution pass ahead of transpilation
- [x] Implement global contiguous Memory Arena calculator for all intermediate tensors
- [x] Eliminate all dynamic `malloc`/`new` and `free`/`delete` calls during inference execution
- [x] Pre-calculate and hardcode exact byte offsets for every intermediate tensor in the Arena
- [x] Support generating isolated C++ execution functions (e.g., `execute_model()`)
- [x] Support generating C++ classes encapsulating the model state
- [x] Embed static `Constant` tensors directly into the C++ binary (arrays)
- [x] Support writing `Constant` tensors to an external `.bin` file loaded at runtime via `mmap`
- [x] Implement `std::expected` (C++23) for monadic error handling boundaries
- [x] Ensure all generated kernel executions are marked `noexcept`
- [x] Support loop unrolling natively via C++ `#pragma unroll` injections
- [x] Emit strict `#pragma omp parallel for` (OpenMP) directives for multi-threading target
- [x] Emit specific `#pragma clang loop vectorize(enable)` (LLVM/Clang) directives
- [x] Support compiling to standalone shared libraries (`.so` / `.dylib` / `.dll`)
- [x] Support compiling to standalone static libraries (`.a` / `.lib`)
- [x] Implement `pybind11` bridge generation for instantly loading the `.so` back into Python
- [x] Expose native C ABI (`extern "C"`) functions for generic FFI bindings (Rust, Go, etc.)
- [x] Emit zero-copy array pointers across the C API boundary
- [x] Validate generated C++ logic statically via `static_assert` statements
- [x] Emulate multidimensional arrays using contiguous flat `std::span` or `std::vector` abstractions
- [x] Implement broadcasting math macros statically in C++
- [x] Implement N-dimensional indexing macros statically (`index_4d(n, c, h, w)`)
- [x] Ensure strict adherence to `-Wall -Wextra -Werror` warnings
- [x] Optimize arithmetic precision (avoiding implicit `double` promotion in C++)
- [x] Support `__restrict__` pointers for alias-free compiler optimizations
- [x] Handle subgraphs (`If`, `Loop`) via recursive C++ function generation
- [x] Support generating static switch statements for categorical routing
- [x] Generate metadata accessors (`get_input_shape()`, `get_output_type()`)
- [x] Eliminate dead C++ variables (transpiler level DCE)
- [x] Optimize identical loop fusion directly in the C++ generator

### 2. WebAssembly (WASM) Backend Compilation (30+ items)

- [x] Detect Emscripten (`emcc`) installation automatically
- [x] Emit Emscripten JS glue code (`--bind` or WebIDL) automatically
- [x] Compile directly to `.wasm` payload
- [x] Compile to combined `.js` and `.wasm` standard module format
- [x] Support building strictly standalone WASM (no JS glue, using pure Wasm imports/exports)
- [x] Enable `-O3` Emscripten optimization flags by default
- [x] Enable `-Os` (size optimization) flag
- [x] Enable `-Oz` (extreme size optimization) flag
- [x] Inject `-msimd128` flags natively for WebAssembly SIMD
- [x] Emit explicit WASM SIMD intrinsics (`wasm_simd128.h`) for heavy math kernels
- [x] Ensure the WASM payload fits within standard browser limits (without breaking WebAssembly.instantiate limits)
- [x] Provide configurable WASM `INITIAL_MEMORY` parameters based on the calculated static Arena size
- [x] Provide configurable WASM `MAXIMUM_MEMORY` parameters
- [x] Enable `ALLOW_MEMORY_GROWTH=1` when shapes are perfectly static (performance boost)
- [x] Expose JS typed-array bridges for zero-copy input/output evaluation
- [x] Transpile JS `Float32Array` directly to the `execute` C++ pointers
- [x] Compile WebWorker wrappers automatically for off-main-thread browser execution
- [x] Support multithreading in WASM via `USE_PTHREADS=1` and `SharedArrayBuffer`
- [x] Compress large constants externally for HTTP chunking alongside the `.wasm`
- [x] Provide TypeScript definitions (`.d.ts`) for the generated WASM module natively
- [x] Verify execution exactly matches Python ONNX predictions (browser unit tests)
- [x] Support `Node.js` environment natively in the generated WASM wrappers
- [x] Support `Deno` environment natively in the generated WASM wrappers

### 3. CPU Core Operations (C++ Kernels) (40+ items)

- [x] Implement `Add` kernel (broadcasted and flat)
- [x] Implement `Sub` kernel
- [x] Implement `Mul` kernel
- [x] Implement `Div` kernel
- [x] Implement `MatMul` kernel (Naive 3-loop)
- [x] Implement `MatMul` kernel (Cache-blocked / Tiled)
- [x] Implement `Conv` kernel (Naive im2col + gemm)
- [x] Implement `Conv` kernel (Direct spatial convolution)
- [x] Implement `Conv` kernel (Depthwise specific optimization)
- [x] Implement `MaxPool` kernel
- [x] Implement `AveragePool` kernel
- [x] Implement `GlobalAveragePool` kernel
- [x] Implement `Relu` kernel (branchless `std::max`)
- [x] Implement `LeakyRelu` kernel
- [x] Implement `Sigmoid` kernel (using fast math approximations if enabled)
- [x] Implement `Tanh` kernel
- [x] Implement `Exp` kernel
- [x] Implement `Log` kernel
- [x] Implement `Softmax` kernel (numerically stable: subtract max)
- [x] Implement `ReduceSum` kernel
- [x] Implement `ReduceMean` kernel
- [x] Implement `ReduceMax` kernel
- [x] Implement `ReduceMin` kernel
- [x] Implement `Transpose` kernel
- [x] Implement `Reshape` (No-op in flat memory, pure logical remap)
- [x] Implement `Flatten` (No-op)
- [x] Implement `Squeeze` (No-op)
- [x] Implement `Unsqueeze` (No-op)
- [x] Implement `Concat` kernel
- [x] Implement `Split` kernel
- [x] Implement `Slice` kernel
- [x] Implement `Gather` kernel
- [x] Implement `ScatterElements` kernel
- [x] Implement `ScatterND` kernel
- [x] Implement `GatherND` kernel
- [x] Implement `Where` kernel
- [x] Implement `Cast` kernel
- [x] Implement `ConstantOfShape` kernel (memset)
- [x] Implement `Pad` kernel (constant padding)
- [x] Implement `NonMaxSuppression` kernel

### 4. Apple Accelerate Framework Integration (20+ items)

- [x] Detect `Accelerate` framework on macOS natively
- [x] Bind `MatMul` to `cblas_sgemm` (Float32)
- [x] Bind `MatMul` to `cblas_dgemm` (Float64)
- [x] Bind `MatMul` to `cblas_hgemm` (Float16 if supported)
- [x] Bind Elementwise `Add` to `vDSP_vadd`
- [x] Bind Elementwise `Mul` to `vDSP_vmul`
- [x] Bind Elementwise `Div` to `vDSP_vdiv`
- [x] Bind `Exp` to `vforce_vexp` / `vvexpf`
- [x] Bind `Log` to `vforce_vlog` / `vvlogf`
- [x] Bind `Sin` to `vforce_vsin` / `vvsinf`
- [x] Bind `Cos` to `vforce_vcos` / `vvcosf`
- [x] Bind `Tanh` to `vforce_vtanh` / `vvtanhf`
- [x] Bind `Sqrt` to `vforce_vsqrt` / `vvsqrtf`
- [x] Bind `ReduceSum` to `vDSP_sve`
- [x] Bind `ReduceMax` to `vDSP_maxv`
- [x] Bind `ReduceMin` to `vDSP_minv`
- [x] Bind `ReduceMean` to `vDSP_meanv`
- [x] Validate zero-copy passing of memory arena pointers to `cblas`
- [x] Dynamically link `-framework Accelerate` during `clang++` compilation
- [x] Fallback to native C++ loop if dimensions do not match BLAS requirements

### 5. OpenBLAS & MKL Fallback (15+ items)

- [x] Detect OpenBLAS on Linux/Windows natively
- [x] Bind `MatMul` to OpenBLAS `cblas_sgemm`
- [x] Detect Intel MKL on compatible hardware
- [x] Bind `MatMul` to Intel MKL `cblas_sgemm`
- [x] Support dynamic linking of `libopenblas.so` during compilation
- [x] Support static linking of OpenBLAS
- [x] Validate row-major vs col-major transpose flags (`CblasRowMajor`) for MKL
- [x] Inject `#include <cblas.h>` or `<mkl.h>` dynamically based on target flag
- [x] Fallback gracefully to cache-blocked C++ kernels if no BLAS is detected

### 6. Neural Architecture & Optimization Specifics (25+ items)

- [x] Implement `LayerNormalization` kernel natively in C++
- [x] Implement `BatchNormalization` kernel natively (Inference mode)
- [x] Implement `Gelu` kernel natively (Erf and Tanh approximations)
- [x] Implement `HardSwish` kernel
- [x] Implement `Mish` kernel
- [x] Compile `TreeEnsembleClassifier` natively into static C++ `if/else` bounds or loop structures
- [x] Compile `TreeEnsembleRegressor` natively
- [x] Support dynamic sequence execution (`RNN`, `LSTM`, `GRU`) via static unrolling if `seq_len` is constant
- [x] Support dynamic sequence execution via C++ `for` loops if `seq_len` is dynamic
- [x] Embed explicit lookup tables (LUTs) for complex math if requested (`--fast-math`)
- [x] Provide exact memory strides mathematically for N-Dimensional slicing without looping
- [x] Translate `ai.onnx.ml.Scaler` to a vectorized loop
- [x] Translate `ai.onnx.ml.OneHotEncoder` to explicit array indexing
- [x] Handle `Einsum` statically if equation is solvable at compile time
- [x] Generate standard C++ `<random>` library calls for `RandomUniform`
- [x] Generate standard C++ `<random>` library calls for `RandomNormal`
- [x] Generate standard C++ calls for `Multinomial`
- [x] Handle explicit memory `memset` for `ZerosLike`
- [x] Ensure `Softmax` operations are cache-friendly (processing rows continuously)
- [x] Strip out `Dropout` operations entirely during C++ generation (inference mode)
- [x] Strip out `Identity` operations natively during the transpiler phase
- [x] Validate `Cast` kernels correctly map `float` to `int` safely

### 7. Explicit Advanced C++ Transpiler Support (40+ items)

- [x] Support `float16` (`_Float16`) code generation natively in C++23
- [x] Support `bfloat16` (`__bf16`) code generation natively
- [x] Support `int8_t` memory alignment natively
- [x] Support `uint8_t` memory alignment natively
- [x] Support `int64_t` processing safely (preventing `int32` overflows in loops)
- [x] Generate pure `<complex>` headers for ONNX complex math operations
- [x] Implement `std::string` handling for ONNX `String` tensors in the C++ backend
- [x] Extract literal dimensions to explicit `constexpr` variables
- [x] Map Python strings to `constexpr std::string_view` mapping tables
- [x] Expose native C++ `execute(const float* input, float* output)` function signatures
- [x] Manage dynamically shaped inputs via pointer sizes `execute(float* in, size_t dim)`
- [x] Use `std::unique_ptr` for memory arena to prevent leaks if allocated dynamically
- [x] Handle `ConstantOfShape` with dynamic shapes via `std::vector` inside the Arena wrapper
- [x] Generate specific `Makefile` or `CMakeLists.txt` (optional) alongside the `.cpp` file
- [x] Execute `clang-format` automatically on generated C++ to maintain extreme readability
- [x] Auto-generate a `main.cpp` entrypoint for direct CLI testing/benchmarking of the compiled model
- [x] Generate internal C++ benchmarking macros (`#define PROFILE_LAYERS`) to time individual kernels
- [x] Ensure strict adherence to `clang-tidy` constraints
- [x] Replace `std::pow(x, 2)` with `x * x` explicitly for performance
- [x] Optimize division by powers of 2 into right-shifts (`>>`) for integers
- [x] Generate explicit branch-prediction hints (`[[likely]]`, `[[unlikely]]`) for `If` statements
- [x] Embed Model Name, Version, and Producer as `#define` strings
- [x] Extract and embed the original ONNX `doc_string` as a C++ multiline comment

### 8. Testing & Validation (Edge Cases) (30+ items)

- [x] Unit Test: Compile pure `Add` graph to C++ and execute via Pybind11
- [x] Unit Test: Compile `MatMul` (statically shaped) and execute natively
- [x] Unit Test: Compile `MatMul` (dynamic batch size) and execute natively
- [x] Unit Test: Compile `Conv` + `Relu` chain and validate output against ONNX Runtime (atol=1e-5)
- [x] Unit Test: Compile standard ResNet50 to C++ and evaluate ImageNet sample
- [x] Unit Test: Compile massive `TreeEnsemble` (Random Forest) to standalone WASM (<1MB payload)
- [x] Unit Test: Execute WASM binary strictly inside V8 (Node.js) and validate results
- [x] Unit Test: Execute `If` branching structures generated as C++ `if/else`
- [x] Unit Test: Validate loop unrolling limits gracefully (falling back to standard loops for large N)
- [x] Validate static Arena calculator correctly aliases memory perfectly across sequential layers
- [x] Catch dynamic shape violations at transpilation time
- [x] Unit Test: Transpile and link Apple Accelerate strictly on MacOS and benchmark
- [x] Validate OpenBLAS linkage on Ubuntu environments
- [x] Stress Test: Compile a 1000-layer generated graph (testing Jinja2 stack depth limits)
- [x] Ensure extreme model topologies do not cause C++ compiler OOM (Out Of Memory)
- [x] Test cross-compilation (e.g. compiling for `aarch64-linux-gnu` from x86_64) if LLVM/Clang is used natively
- [x] Validate WASM SIMD execution strictly in Chrome

### 9. Exhaustive C++ Operator Implementations (60+ items)

- [x] Implement `Abs` kernel (branchless `std::abs`)
- [x] Implement `Acos` kernel (`std::acos`)
- [x] Implement `Acosh` kernel (`std::acosh`)
- [x] Implement `Add` kernel (with explicit 1D, 2D, 3D, 4D broadcasting loops)
- [x] Implement `And` kernel
- [x] Implement `ArgMax` kernel
- [x] Implement `ArgMin` kernel
- [x] Implement `Asin` kernel (`std::asin`)
- [x] Implement `Asinh` kernel (`std::asinh`)
- [x] Implement `Atan` kernel (`std::atan`)
- [x] Implement `Atanh` kernel (`std::atanh`)
- [x] Implement `BitShift` kernel (`<<`, `>>`)
- [x] Implement `BitwiseAnd` kernel (`&`)
- [x] Implement `BitwiseNot` kernel (`~`)
- [x] Implement `BitwiseOr` kernel (`|`)
- [x] Implement `BitwiseXor` kernel (`^`)
- [x] Implement `Ceil` kernel (`std::ceil`)
- [x] Implement `Clip` kernel (`std::clamp`)
- [x] Implement `Compress` kernel
- [x] Implement `Constant` kernel (memcpy from ROM)
- [x] Implement `Cos` kernel (`std::cos`)
- [x] Implement `Cosh` kernel (`std::cosh`)
- [x] Implement `CumSum` kernel
- [x] Implement `DepthToSpace` kernel (memory permutation)
- [x] Implement `DequantizeLinear` kernel
- [x] Implement `Det` kernel
- [x] Implement `Dropout` kernel (inference mode - pass-through)
- [x] Implement `Einsum` kernel (naive nested loops based on generated string)
- [x] Implement `Elu` kernel
- [x] Implement `Equal` kernel (`==`)
- [x] Implement `Erf` kernel (`std::erf`)
- [x] Implement `Expand` kernel (logical broadcast remap)
- [x] Implement `EyeLike` kernel
- [x] Implement `Floor` kernel (`std::floor`)
- [x] Implement `GatherElements` kernel
- [x] Implement `GlobalLpPool` kernel
- [x] Implement `GlobalMaxPool` kernel
- [x] Implement `Greater` kernel (`>`)
- [x] Implement `GreaterOrEqual` kernel (`>=`)
- [x] Implement `Hardmax` kernel
- [x] Implement `HardSigmoid` kernel
- [x] Implement `Identity` kernel (if not stripped by DCE)
- [x] Implement `IsInf` kernel (`std::isinf`)
- [x] Implement `IsNaN` kernel (`std::isnan`)
- [x] Implement `LRN` kernel (Local Response Normalization)
- [x] Implement `Less` kernel (`<`)
- [x] Implement `LessOrEqual` kernel (`<=`)
- [x] Implement `LogSoftmax` kernel
- [x] Implement `LpNormalization` kernel
- [x] Implement `LpPool` kernel
- [x] Implement `Max` kernel (`std::max`)
- [x] Implement `MaxRoiPool` kernel
- [x] Implement `Mean` kernel (Elementwise mean)
- [x] Implement `Min` kernel (`std::min`)
- [x] Implement `Mod` kernel (`std::fmod` / `%`)
- [x] Implement `Multinomial` kernel (using `<random>`)
- [x] Implement `Neg` kernel (`-`)
- [x] Implement `NonZero` kernel (Dynamic memory allocation required)
- [x] Implement `Not` kernel (`!`)
- [x] Implement `OneHot` kernel
- [x] Implement `Or` kernel (`||`)

### 10. Memory Planning & Dynamic Allocations (30+ items)

- [x] Implement dynamic tensor memory reallocation gracefully (for `NonZero` and `Compress`)
- [x] Support falling back from Static Arena to `std::vector` if fully dynamic shapes are encountered
- [x] Provide explicit C++ `Context` struct tracking dynamic sizes at runtime
- [x] Provide explicit C++ `Allocator` interface for custom memory management integration
- [x] Optimize intermediate buffer reuse (e.g. `Buffer A` -> `Buffer B` -> `Buffer A`) via graph coloring
- [x] Validate memory graph coloring via static verification scripts
- [x] Support generating `alignas(64)` for strict cache-line alignment natively
- [x] Support generating `alignas(32)` for AVX instructions specifically
- [x] Guarantee `alignas(16)` for WebAssembly SIMD boundaries
- [x] Support `#pragma pack` for compacting structural data definitions
- [x] Export structural map of the arena layout natively to a JSON descriptor
- [x] Pre-calculate all broadcasting strides statically into `constexpr` tables
- [x] Eliminate dynamic stride multiplication in hot loops using pointer arithmetic directly
- [x] Utilize C++ `std::span` bounds checking if compiled with `-DDEBUG`
- [x] Disable all bounds checking implicitly under `-O3 -DNDEBUG`
- [x] Support generating explicit boundary checks for `Gather` (preventing segfaults)
- [x] Embed external weights (`.bin`) via cross-platform POSIX `mmap()` automatically
- [x] Embed external weights natively using Windows `CreateFileMapping` / `MapViewOfFile`
- [x] Optimize integer division using `libdivide` algorithms (statically generated) if divisor is constant
- [x] Calculate specific memory layout bytes mathematically given tensor shapes and dtypes natively

### 11. Pybind11 & C API Interop (20+ items)

- [x] Generate strict `<pybind11/pybind11.h>` headers and modules dynamically
- [x] Generate `<pybind11/numpy.h>` bridges automatically for `py::array_t<float>` handling
- [x] Guarantee zero-copy evaluation when Python arrays are strictly `C_CONTIGUOUS`
- [x] Safely copy arrays using C++ `memcpy` if Python inputs are fragmented or `F_CONTIGUOUS`
- [x] Extract pointer addresses dynamically from `py::buffer_info`
- [x] Release Python GIL (`py::gil_scoped_release`) natively before invoking the C++ Arena execution
- [x] Reacquire GIL safely before returning outputs to Python
- [x] Support compiling the generated `_model.cpp` dynamically inside Python using `subprocess`
- [x] Load the compiled `.so` using Python `ctypes` or `importlib` dynamically after JIT compilation
- [x] Generate standard C `extern` functions for C# / .NET P/Invoke integrations
- [x] Generate standard C `extern` functions for Go / CGO integrations
- [x] Generate standard C `extern` functions for Rust FFI bindings
- [x] Export model signature (`get_input_count`, `get_input_name`, `get_output_shape`) via C API
- [x] Catch native C++ exceptions `try/catch` and translate to Python `RuntimeError` securely

### 12. Advanced Emscripten & WASM Opts (20+ items)

- [x] Emit `--no-entry` flag dynamically if compiling a pure library (no main)
- [x] Emit `-s EXPORTED_FUNCTIONS=['_execute', '_malloc', '_free']` automatically
- [x] Emit `-s EXPORTED_RUNTIME_METHODS=['ccall', 'cwrap']` for dynamic invocation
- [x] Embed specific WASM `Module` initialization handlers dynamically into the emitted JS
- [x] Expose `HEAPF32` / `HEAPU8` array views seamlessly back to Javascript natively
- [x] Handle 64-bit integer variables in JS (using `BigInt` safely over the WASM boundary)
- [x] Test the execution latency difference of `-O3` vs `-Oz` specifically for CNN models
- [x] Test the file size difference of `-O3` vs `-Oz` specifically for deep tree ensembles
- [x] Evaluate `-s ALLOW_MEMORY_GROWTH=1` overhead impact
- [x] Optimize specific `Math.exp()` / `Math.log()` calls using fast-math if `-O3` is specified
- [x] Handle WebAssembly Out-Of-Bounds memory traps safely by surfacing Javascript Errors
- [x] Benchmark WASM `MatMul` speed scaling relative to input dimensions
- [x] Profile WASM execution without SIMD (Baseline tests)
- [x] Profile WASM execution with SIMD enabled (Performance tests)

### 13. Opset Compliance & Edge Cases (25+ items)

- [x] Implement `Pow` kernel (`std::pow`)
- [x] Implement `PRelu` kernel
- [x] Implement `QLinearConv` kernel (handling zero-points and scales)
- [x] Implement `QLinearMatMul` kernel (handling zero-points and scales)
- [x] Implement `QuantizeLinear` kernel
- [x] Implement `RNN` kernel (with explicit unrolling options)
- [x] Implement `RandomNormal` kernel
- [x] Implement `RandomNormalLike` kernel
- [x] Implement `RandomUniform` kernel
- [x] Implement `RandomUniformLike` kernel
- [x] Implement `Range` kernel
- [x] Implement `Reciprocal` kernel (`1.0 / x`)
- [x] Implement `ReduceL1` kernel
- [x] Implement `ReduceL2` kernel
- [x] Implement `ReduceLogSum` kernel
- [x] Implement `ReduceLogSumExp` kernel
- [x] Implement `ReduceProd` kernel
- [x] Implement `ReduceSumSquare` kernel
- [x] Handle unrolling depth limits natively in the C++ generator (preventing massive binary bloat)
- [x] Warn if `Loop` iterations are highly dynamic (generating `while` loops instead of `for`)
- [x] Verify execution exactly matches Python ONNX predictions (browser unit tests)
- [x] Compile and run `onnxruntime` standard compliance models (`test_add`, `test_matmul`) natively
- [x] Compile and run `onnxruntime` compliance models for CNNs (`test_resnet`) natively
- [x] Compile and run `onnxruntime` compliance models for NLP (`test_bert`) natively
- [x] Automatically fallback to executing Python IR if C++ compilation fails locally
