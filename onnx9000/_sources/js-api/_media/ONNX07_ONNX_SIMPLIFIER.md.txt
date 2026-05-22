# ONNX Simplifier Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `onnx-simplifier` within the `onnx9000` ecosystem.
The original `onnx-simplifier` infers shapes and folds constants by spinning up a heavy C++ `onnxruntime` session to evaluate subgraphs of constant nodes.
Our `onnx9000` implementation achieves identical or superior graph simplification entirely in pure Python. By embedding a lightweight, mathematically exact tensor execution engine, it evaluates constant nodes natively using NumPy or JS TypedArrays. Coupled with advanced algebraic rewriting (e.g., `A * 1 -> A`, `A + 0 -> A`), symbolic shape inference, and aggressive Dead Code Elimination (DCE), this zero-dependency simplifier can operate instantly inside Web Browsers (WASM), Serverless functions (AWS Lambda), and standard CI/CD pipelines without compiling external native libraries.

## Exhaustive Parity Checklist

### 1. Core Simplification Engine & Architecture (35+ items)

- [x] Implement pure-Python `simplify(model)` API parity
- [x] Implement zero-dependency `ModelProto` parsing and serialization
- [x] Maintain a strict Topological Sort during simplification passes
- [x] Implement multi-pass architecture (Iterate passes until graph converges/stops changing)
- [x] Detect and break infinite optimization loops automatically (max iterations limit)
- [x] Extract dynamically all subgraphs consisting entirely of `Constant` / `Initializer` nodes
- [x] Execute constant subgraphs natively in Python memory to produce new folded constants
- [x] Replace executed subgraphs with single `Constant` nodes
- [x] Bypass `onnxruntime` dependency completely for constant subgraph execution
- [x] Support dynamic custom dimensions (`dynamic_axes`) preservation
- [x] Parse explicitly requested input shape overrides (e.g., `--input-shape "x:1,3,224,224"`)
- [x] Parse explicitly requested tensor type overrides
- [x] Check explicit model size limits before and after simplification
- [x] Support large models (>2GB) utilizing external data `.bin` loading and saving natively
- [x] Emulate `onnx-simplifier` strict adherence to ONNX Opset standard topologies
- [x] Update standard `ValueInfoProto` automatically with inferred shapes/types
- [x] Provide explicit skips for specific optimization rules via configuration dicts
- [x] Expose `skip_constant_folding` flag
- [x] Expose `skip_shape_inference` flag
- [x] Expose `skip_fuse_bn` flag
- [x] Output a rich JSON summary of nodes removed, folded, and simplified
- [x] Preserve the model's original `doc_string` and `producer_name` (or append simplifier signature safely)
- [x] Ensure Graph inputs (`ModelProto.graph.input`) maintain their exact order and names
- [x] Ensure Graph outputs maintain exact order and names
- [x] Automatically upgrade older opsets (e.g. Opset 9) to a target opset (e.g. Opset 13) before simplification if requested
- [x] Automatically downgrade to target opsets if structurally compatible
- [x] Preserve all un-recognized `CustomOp` nodes gracefully without crashing
- [x] Skip constant folding on `CustomOp` nodes to prevent mathematical divergence
- [x] Allow users to register custom pure-Python execution kernels for `CustomOp` constant folding
- [x] Run seamlessly inside Pyodide / WASM without C++ extensions
- [x] Run seamlessly inside Node.js without C++ native addons
- [x] Ensure peak RAM usage during simplification doesn't exceed 2x the model size (preventing OOMs)
- [x] Profile execution time of each simplification pass individually
- [x] Support handling models with multiple disjoint/disconnected subgraphs properly
- [x] Extract and embed the `onnx9000` execution engine footprint securely within Cloudflare limits

### 2. Algebraic Rewriting & Simplification (45+ items)

- [x] Rewrite `Add(X, 0)` -> `X` (Identity Addition)
- [x] Rewrite `Add(0, X)` -> `X`
- [x] Rewrite `Sub(X, 0)` -> `X` (Identity Subtraction)
- [x] Rewrite `Sub(X, X)` -> `Constant(0)`
- [x] Rewrite `Mul(X, 1)` -> `X` (Identity Multiplication)
- [x] Rewrite `Mul(1, X)` -> `X`
- [x] Rewrite `Mul(X, 0)` -> `Constant(0)`
- [x] Rewrite `Mul(0, X)` -> `Constant(0)`
- [x] Rewrite `Div(X, 1)` -> `X` (Identity Division)
- [x] Rewrite `Div(X, X)` -> `Constant(1)` (Assuming X != 0)
- [x] Rewrite `Pow(X, 1)` -> `X`
- [x] Rewrite `Pow(X, 0)` -> `Constant(1)`
- [x] Rewrite `Neg(Neg(X))` -> `X` (Double negation)
- [x] Rewrite `Abs(Abs(X))` -> `Abs(X)`
- [x] Rewrite `Exp(Log(X))` -> `X`
- [x] Rewrite `Log(Exp(X))` -> `X`
- [x] Rewrite `Sqrt(Pow(X, 2))` -> `Abs(X)`
- [x] Rewrite `Cast(Cast(X, typeA), typeA)` -> `Cast(X, typeA)`
- [x] Rewrite `Cast(X, typeX)` -> `X` (Identity Cast)
- [x] Rewrite `ReduceSum(X)` (when X is scalar) -> `X`
- [x] Rewrite `ReduceMean(X)` (when X is scalar) -> `X`
- [x] Rewrite `Concat([X])` -> `X` (Concat of a single tensor)
- [x] Rewrite `Max(X, X)` -> `X`
- [x] Rewrite `Min(X, X)` -> `X`
- [x] Rewrite `And(X, X)` -> `X`
- [x] Rewrite `Or(X, X)` -> `X`
- [x] Rewrite `Not(Not(X))` -> `X`
- [x] Rewrite `Where(True, X, Y)` -> `X`
- [x] Rewrite `Where(False, X, Y)` -> `Y`
- [x] Rewrite `Where(Cond, X, X)` -> `X`
- [x] Rewrite `Slice(X, 0, MAX_INT)` -> `X` (Full slice)
- [x] Rewrite `Pad(X, 0)` -> `X` (Zero padding)
- [x] Rewrite `Reshape(X, Shape(X))` -> `X` (Redundant reshape)
- [x] Rewrite `Reshape(Reshape(X, shape1), shape2)` -> `Reshape(X, shape2)`
- [x] Rewrite `Transpose(Transpose(X, perm1), perm2)` -> `Transpose(X, perm_combined)`
- [x] Rewrite `Transpose(X, [0, 1, 2, 3])` -> `X` (Identity Permutation)
- [x] Rewrite `Expand(X, Shape(X))` -> `X`
- [x] Rewrite `Tile(X, [1, 1, 1])` -> `X`
- [x] Rewrite `Squeeze(Unsqueeze(X, axis), axis)` -> `X`
- [x] Rewrite `Unsqueeze(Squeeze(X, axis), axis)` -> `X` (If rank is known)
- [x] Fold sequential `Add` -> `Add` into a single `Add` with folded constants
- [x] Fold sequential `Mul` -> `Mul` into a single `Mul` with folded constants
- [x] Implement algebraic distribution `Mul(Add(X, C1), C2)` -> `Add(Mul(X, C2), C1*C2)` (if profitable)
- [x] Eliminate `Dropout` entirely in eval models (Ratio=0 or Train=False)
- [x] Eliminate `Identity` nodes natively and rewire connections

### 3. Structural & Constant Folding (Math) (45+ items)

- [x] Fold `Add` (Constants only) natively
- [x] Fold `Sub` (Constants only) natively
- [x] Fold `Mul` (Constants only) natively
- [x] Fold `Div` (Constants only) natively
- [x] Fold `Pow` natively
- [x] Fold `Sqrt` natively
- [x] Fold `Exp` natively
- [x] Fold `Log` natively
- [x] Fold `Sin`, `Cos`, `Tan` natively
- [x] Fold `Asin`, `Acos`, `Atan` natively
- [x] Fold `Sinh`, `Cosh`, `Tanh` natively
- [x] Fold `Abs` natively
- [x] Fold `Neg` natively
- [x] Fold `Sign` natively
- [x] Fold `Ceil` natively
- [x] Fold `Floor` natively
- [x] Fold `Round` natively
- [x] Fold `Mod` natively
- [x] Fold `Max`, `Min` natively
- [x] Fold `Clip` natively
- [x] Fold `And`, `Or`, `Not`, `Xor` natively
- [x] Fold `Equal`, `Greater`, `Less` natively
- [x] Fold `GreaterOrEqual`, `LessOrEqual` natively
- [x] Fold `BitShift` natively
- [x] Fold `BitwiseAnd`, `BitwiseOr`, `BitwiseNot`, `BitwiseXor` natively
- [x] Fold `IsInf`, `IsNaN` natively
- [x] Fold `Erf` natively
- [x] Fold `Relu` natively
- [x] Fold `Sigmoid` natively
- [x] Fold `ReduceSum` natively
- [x] Fold `ReduceMean` natively
- [x] Fold `ReduceMax`, `ReduceMin` natively
- [x] Fold `ReduceProd` natively
- [x] Fold `ReduceL1`, `ReduceL2` natively
- [x] Fold `ReduceLogSum`, `ReduceLogSumExp` natively
- [x] Fold `Cast` natively (safely applying precision bounds)
- [x] Fold `CastLike` natively
- [x] Handle explicit FP16 constant folding without overflowing in pure Python
- [x] Handle explicit BF16 constant folding natively
- [x] Handle explicit INT8 constant folding natively
- [x] Handle explicit INT64 constant folding seamlessly
- [x] Prevent division by zero during `Div` constant folding (bypass fold)
- [x] Prevent `NaN` propagation dynamically (if node generates `NaN`, skip folding)
- [x] Validate mathematically that folded constant arrays exactly match C++ ORT output (atol=1e-5)
- [x] Extract scalar constants explicitly (converting `1D` array of size 1 to `0D` scalar if topological rules allow)

### 4. Structural & Constant Folding (Tensors & NN) (40+ items)

- [x] Fold `Reshape` (Constants only) natively
- [x] Fold `Transpose` natively
- [x] Fold `Flatten` natively
- [x] Fold `Squeeze` natively
- [x] Fold `Unsqueeze` natively
- [x] Fold `Concat` natively
- [x] Fold `Split` natively
- [x] Fold `Slice` natively
- [x] Fold `Gather` natively
- [x] Fold `GatherElements` natively
- [x] Fold `GatherND` natively
- [x] Fold `Scatter` / `ScatterElements` natively
- [x] Fold `ScatterND` natively
- [x] Fold `Tile` natively
- [x] Fold `Expand` natively
- [x] Fold `Pad` natively
- [x] Fold `ConstantOfShape` natively
- [x] Fold `Where` natively
- [x] Fold `NonZero` natively
- [x] Fold `TopK` natively
- [x] Fold `Unique` natively
- [x] Fold `CumSum` natively
- [x] Fold `ReverseSequence` natively
- [x] Fold `Compress` natively
- [x] Fold `Trilu` natively
- [x] Fold `Shape` natively (crucial for dynamic to static resolution)
- [x] Fold `Size` natively
- [x] Fold `Conv` natively (If inputs AND weights are constants, e.g., in feature extractors)
- [x] Fold `MatMul` natively (If inputs AND weights are constants)
- [x] Fold `Gemm` natively (If inputs AND weights are constants)
- [x] Fold `BatchNormalization` natively (If all inputs are constants)
- [x] Fold `LayerNormalization` natively
- [x] Fold `InstanceNormalization` natively
- [x] Fold `MaxPool` natively
- [x] Fold `AveragePool` natively
- [x] Fuse `BatchNormalization` into preceding `Conv` weights
- [x] Fuse `BatchNormalization` into preceding `Gemm` weights
- [x] Strip purely constant subgraphs feeding into unused dynamic nodes
- [x] Evaluate `SequenceConstruct` natively
- [x] Evaluate `SequenceAt` natively

### 5. Control Flow & Advanced Dead Code Elimination (30+ items)

- [x] Implement Graph-wide Dead Code Elimination (DCE)
- [x] Prune Nodes with zero consumer edges
- [x] Prune Initializers with zero consumer edges
- [x] Prune Graph Inputs dynamically if explicitly specified
- [x] Retain Graph Outputs strictly (even if fed by constants)
- [x] Fold `If` node natively if the `cond` input is a `Constant` (injecting the True/False subgraph directly into parent)
- [x] Prune the un-executed branch of a folded `If` node
- [x] Connect internal subgraph outputs smoothly to the `If` node consumers
- [x] Fold `Loop` node if `max_trip_count` is `Constant(0)` (Prune loop body, route initial states to outputs)
- [x] Fold `Loop` node if `max_trip_count` is `Constant(1)` (Inject loop body directly into parent graph)
- [x] Unroll `Loop` natively if `max_trip_count` is a small constant (e.g. < 10)
- [x] Resolve dynamic sequence lengths inside unrolled `Loop` instances
- [x] Ensure sub-graph `Initializer` scopes are promoted to the parent graph seamlessly when injecting
- [x] Re-map internal topological names to prevent collisions during subgraph injection
- [x] Eliminate `SequenceEmpty` if unused
- [x] Evaluate `If` conditions mathematically if preceded by constant folding (e.g., `Greater(5, 2) -> True -> Fold If`)
- [x] Check `Scan` nodes for dead-code pruning dynamically
- [x] Sweep disconnected components (entire islands of nodes that don't reach outputs)
- [x] Output total number of parameters eliminated during DCE in JSON logs
- [x] Warn explicitly if an Output node becomes entirely disconnected from all Inputs (Constant Output)
- [x] Provide an API to preserve intermediate nodes explicitly (preventing DCE on specific debug probes)
- [x] Sweep unused `ValueInfo` properties dynamically
- [x] Remove `CustomOp` nodes safely if they have no consumers
- [x] Detect implicit cyclical dependencies (invalid ONNX) and break cleanly
- [x] Support DCE inside nested sub-graphs recursively (`If` inside `Loop`)
- [x] Fold shape dimensions explicitly inside Subgraphs (propagating known shapes down)
- [x] Flatten nested `Sequence` operations if statically bounded
- [x] Avoid unrolling loops if it explodes the `.onnx` graph size past a user-defined threshold
- [x] Expose `--no-large-tensor` flag to prevent evaluating constants larger than a threshold (e.g. 10MB)
- [x] Track memory of in-flight constant evaluations explicitly to avoid RAM spikes

### 6. Advanced Shape & Type Inference (30+ items)

- [x] Implement fully native `onnx9000.shape_inference` passes (bypassing ONNX C++ `infer_shapes`)
- [x] Resolve unknown dimensions symbolically (e.g., `batch_size`, `seq_len`)
- [x] Apply mathematical constraints to symbols (`seq_len * 2`)
- [x] Validate `MatMul` tensor dimension alignment exactly
- [x] Validate `Conv` padding and stride calculations exactly
- [x] Propagate shape constraints through `Reshape` when `0` or `-1` are used
- [x] Propagate shape constraints through `Slice` with dynamic axes
- [x] Propagate types (`Float32`, `Int64`) strictly through elementwise ops
- [x] Catch explicitly invalid typings (e.g., `Float32` fed into `And` operator)
- [x] Infer output types for `Cast` nodes
- [x] Infer types for implicit promotions (if required by older opsets)
- [x] Resolve shapes for `Gather` dynamically based on indices length
- [x] Resolve shapes for `Split` natively
- [x] Resolve shapes for `Concat` cleanly
- [x] Infer explicit `ValueInfo` for `Loop` body outputs
- [x] Infer explicit `ValueInfo` for `If` branch outputs
- [x] Merge `If` branch output shapes natively (promoting to dynamic if shapes differ)
- [x] Expose an API to manually inject shape hints to aid inference (`--input-shape "images:1,3,512,512"`)
- [x] Expose an API to lock dynamic axes to static dimensions (e.g., locking `batch_size=1` to allow massive folding)
- [x] Propagate exact `BFloat16` and `Float16` typings
- [x] Update `ModelProto.graph.value_info` array dynamically
- [x] Remove old `ValueInfo` entries for nodes deleted during DCE
- [x] Prevent shape inference loops on recursive models natively
- [x] Ensure shape inference matches `onnx.checker` structural rules perfectly
- [x] Strip out explicitly conflicting shape metadata
- [x] Re-calculate `onnx.checker.check_model` using pure Python alternative natively
- [x] Analyze dynamic sizes (e.g. `NonZero`) by propagating `Unknown` shape tokens
- [x] Expose bounding values for `Unknown` shapes (e.g., `max_size=1000`)
- [x] Fallback gracefully when encountering un-inferable `CustomOp` domains
- [x] Log every successful and failed shape inference derivation for debugging

### 7. Environment, CLI & Web Integrations (25+ items)

- [x] Expose pure Python CLI: `onnx9000 simplify model.onnx model_sim.onnx`
- [x] Expose JS/TypeScript API: `const simplifiedModel = await onnx9000.simplify(modelBuffer)`
- [x] Provide WebWorker wrapper natively in JS to prevent UI freezing during large simplifies
- [x] Integrate seamlessly with `Netron` reimplementation (Button: "Simplify Graph")
- [x] Run seamlessly in Node.js pipelines
- [x] Provide strict JSON logging for automated CI/CD integration metrics
- [x] Output Markdown table summarization of simplification (Before/After Node Counts, Sizes)
- [x] Allow streaming of >2GB models via Python `mmap` parsing
- [x] Do not require `pip install onnx` natively (Zero-dependency)
- [x] Do not require `pip install onnxruntime` natively
- [x] Emulate official `onnxsim` CLI arguments perfectly (`--input-shape`, `--skip-fuse-bn`, etc.)
- [x] Expose `--custom-ops` definitions dynamically
- [x] Prevent file overwrites implicitly (unless `--overwrite` is specified)
- [x] Validate standard ONNX file magic bytes safely
- [x] Expose API to simplify an already loaded `onnx9000.Graph` object directly in memory
- [x] Implement auto-fallback: If an optimization throws a math exception, rollback the graph and continue safely
- [x] Output human-readable error traces when encountering fundamentally broken ONNX topologies
- [x] Support reading models from AWS S3 / HTTP endpoints dynamically
- [x] Support generating optimized `safetensors` mappings immediately post-simplification
- [x] Compress the JS transpiled simplifier to < 1MB (minified)
- [x] Validate Cloudflare Worker integration natively
- [x] Validate AWS Lambda integration natively
- [x] Run perfectly within Pyodide strict WASM memory constraints
- [x] Inject WebGPU compute bounds validation explicitly if requested
- [x] Verify standard Python 3.9 through 3.12 compatibility

### 8. Testing, Compliance & Edge Cases (30+ items)

- [x] Unit Test: Simplify purely arithmetic graph (`Add(2, Mul(3, 4))` -> `Constant(14)`)
- [x] Unit Test: Simplify redundant `Reshape` nodes on standard ResNet
- [x] Unit Test: Fold `BatchNormalization` natively on standard MobileNet
- [x] Unit Test: Lock batch size on BERT and evaluate `Shape` constant folding cascading effects
- [x] Unit Test: Prune unused multi-head attention outputs natively
- [x] Unit Test: Evaluate `Slice` folding on text tokenizer bounds
- [x] Unit Test: Execute `onnx-simplifier` canonical tests and achieve identical structural node counts
- [x] Unit Test: Compare resulting mathematical output against original unsimplified graph using PyTorch (atol=1e-5)
- [x] Test folding of massive matrices (100MB+ constants) natively without OOM
- [x] Handle models with exactly zero parameters (math-only graphs) natively
- [x] Handle models with exactly zero graph inputs (pure constant generators) natively
- [x] Catch explicitly invalid `input_shape` argument strings cleanly
- [x] Support evaluating `Einsum` into pure constants if inputs are constants
- [x] Catch numerical underflow when folding `Exp` with massive negative inputs
- [x] Catch numerical overflow when folding `Exp` with massive positive inputs (replace with `Inf`)
- [x] Maintain `NaN` outputs logically if folded math results in undefined values
- [x] Verify `BitShift` edge cases (shifting beyond 32/64 bits) match Python/C++ exactly
- [x] Fold boolean logic (`And`, `Or`) cleanly matching Python's short-circuit logic
- [x] Handle models containing `Float64` precision without silent downcasting to `Float32`
- [x] Verify `Round` constant folding matches "Round half to even" (Banker's rounding) natively
- [x] Optimize the traversal sorting sequence (DFS/BFS) to process cascades in O(N) time
- [x] Test the simplifier on highly complex generative LLMs (GPT-J, Llama 2)
- [x] Track simplifier latency on 1000+ node graphs (target < 100ms)
- [x] Ensure Graph Outputs mapped directly from Constants are serialized correctly
- [x] Verify all `ValueInfo` shapes use `-1` or named string symbols natively for dynamic boundaries
- [x] Catch explicitly duplicated topological names during `If`/`Loop` extraction
- [x] Validate `onnx9000.shape_inference` handles `ai.onnx.ml` domain topologies seamlessly
- [x] Prevent topological loops fundamentally inside the rewrite engine
- [x] Emulate `onnxruntime` strict Type constraints (e.g. `Where` condition must be Bool)
- [x] Emit detailed warning if simplifier encounters unsupported custom ops breaking the cascade

### 9. Advanced Edge Cases, Operator Specifics, & Tooling (30+ items)

- [x] Evaluate `NonMaxSuppression` if all inputs (boxes, scores, max_output) are constants
- [x] Handle `ConstantOfShape` where shape is an empty tensor `[]` correctly (generating a scalar)
- [x] Handle `ConstantOfShape` where shape contains `0` correctly (generating an empty tensor)
- [x] Prevent folding `RandomUniform` and `RandomNormal` (must remain dynamic)
- [x] Prevent folding `RandomUniformLike` and `RandomNormalLike`
- [x] Prevent folding `Multinomial` (must remain dynamic)
- [x] Fold `Range` natively if `start`, `limit`, and `delta` are constants
- [x] Evaluate `OneHot` natively if `indices`, `depth`, and `values` are constants
- [x] Handle `Softmax` on massive constants natively (subtracting max before exp)
- [x] Support handling specific `bfloat16` packing and unpacking natively in pure Python if `torch` is absent
- [x] Prevent massive memory spikes by implementing a `--size-limit` flag for constant folding (e.g., skip folding if result > 200MB)
- [x] Inject `Identity` node temporarily if an output is directly wired to a constant, then resolve according to ONNX spec rules
- [x] Validate `ai.onnx.ml.ArrayFeatureExtractor` shapes recursively
- [x] Validate `ai.onnx.ml.LinearClassifier` shapes recursively
- [x] Validate `ai.onnx.ml.TreeEnsembleClassifier` shapes recursively
- [x] Map PyTorch exported `aten::arange` -> `Range` and fold if statically known
- [x] Highlight un-foldable branches specifically in the console output (e.g., "Branch blocked by dynamic Shape")
- [x] Provide an explicit `--check-n=3` flag (like `onnxsim`) to run the simplified model N times and verify output consistency
- [x] Evaluate `Einsum` into explicit pure constants natively (if equation is valid)
- [x] Fold `CumSum` natively in Python/JS if inputs are constants
- [x] Fold `Trilu` (upper/lower triangle extraction) natively
- [x] Prevent `SplitToSequence` from folding into massive arrays if length is unbound
- [x] Optimize sequential `Expand` -> `Expand` operations cleanly
- [x] Clean up redundant `Cast` -> `CastLike` combinations
- [x] Explicitly test Simplifier stability on models with `dim_param` nested inside subgraph values
- [x] Ensure output `ValueInfoProto` arrays are strictly sorted alphabetically if requested
- [x] Strip ONNX `ModelProto.metadata_props` if explicitly requested (to save space)
- [x] Maintain `ModelProto.opset_import` strictly, deleting domains that are fully eliminated during DCE
- [x] Fallback gracefully when `Float16` constant folding overflows, leaving the node unfolded rather than corrupting the graph
- [x] Output a visual DAG difference (`A.onnx` vs `B.onnx`) JSON file for external visualizers to diff before/after topological changes
