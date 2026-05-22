# ONNXScript / Spox Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `ONNXScript` and `Spox` within the `onnx9000` ecosystem.
The original `ONNXScript` and `Spox` projects provide fantastic Pythonic abstractions for authoring ONNX graphs, but they rely heavily on the native C++ `protobuf` library and the official `onnx` Python package to validate, build, and serialize models.
Our `onnx9000` reimplementation provides a **zero-dependency, fluent, PyTorch-like Python API**. By utilizing our pure-Python internal Intermediate Representation (IR) and execution engine, developers can author, trace, shape-infer, and execute complex ONNX graphs node-by-node dynamically. This architecture enables building and exporting ONNX models entirely inside a web browser via WASM/Pyodide or in strict serverless environments without ever installing C++ Protobuf or the massive `onnx` package.

## Exhaustive Parity Checklist

### 1. Core Authoring API & Tracing Architecture (40+ items)

- [x] Implement `onnx9000.script()` decorator for tracing Python functions
- [x] Implement `onnx9000.Tensor` base abstraction for eager/symbolic execution
- [x] Implement `onnx9000.Parameter` abstraction for trainable weights
- [x] Implement `onnx9000.Constant` abstraction for frozen tensors
- [x] Implement PyTorch-like fluent API (e.g., `tensor.sum()`, `tensor.reshape()`)
- [x] Implement explicit `onnx9000.op.*` namespace mapping perfectly to ONNX domains
- [x] Support generating `ai.onnx` operators cleanly
- [x] Support generating `ai.onnx.ml` operators cleanly
- [x] Support custom operator domains (`custom_opset="ai.onnx.contrib"`)
- [x] Extract Python function arguments dynamically as Graph Inputs
- [x] Extract Python function return values dynamically as Graph Outputs
- [x] Support explicit type annotations (e.g., `x: Float[10, 20]`) mapping to `ValueInfo`
- [x] Support symbolic dimension annotations (e.g., `x: Float["batch", "seq_len"]`)
- [x] Implement dynamic variable tracing (e.g., `z = x + y` records an `Add` node)
- [x] Record Topological order exactly as Python executes the function
- [x] Implement `GraphBuilder` context manager for imperative manual construction
- [x] Catch dynamic Python control flow (`if x > 0:`) natively and warn if un-traceable
- [x] Map static Python `if` statements to traced paths cleanly
- [x] Provide explicit `onnx9000.If(cond, true_fn, false_fn)` API for dynamic branching
- [x] Provide explicit `onnx9000.Loop(max_trip, cond, body_fn)` API for dynamic looping
- [x] Extract Python nested functions as sub-graphs cleanly
- [x] Support multi-output operators natively (unpacking tuples)
- [x] Track explicitly named intermediate tensors (`x.name = "hidden_state"`)
- [x] Inject `doc_string` metadata explicitly into the generated nodes
- [x] Inject `ModelProto` global metadata (Producer, Version, Description)
- [x] Build pure-Python `ModelProto` completely bypassing C++ `protobuf`
- [x] Support auto-casting Python scalars (e.g., `x + 1.0` -> `Add(x, Constant(1.0))`)
- [x] Support auto-casting Python lists (e.g., `x.reshape([1, 2])`)
- [x] Manage Graph Inputs and Initializers separately
- [x] Expose an API to lock dynamic axes to static numbers (`batch_size=1`)
- [x] Validate topological connectivity before serialization
- [x] Reject cycles (cyclic dependencies) natively
- [x] Allow nesting `@onnx9000.script` functions (Sub-function tracing)
- [x] Prevent Python Garbage Collector from sweeping intermediate tensors during tracing
- [x] Support tracing inside Pyodide/Browser environments seamlessly
- [x] Export purely to raw `.onnx` bytes dynamically (`to_bytes()`)
- [x] Save purely to disk (`to_file('model.onnx')`)
- [x] Expose `to_json()` for human-readable debugging
- [x] Parse `from_json()` to reconstruct the builder dynamically
- [x] Emulate `Spox` `Var` and `Node` abstractions implicitly

### 2. Fluent Tensor Math Operators (45+ items)

- [x] Implement `tensor + other` (`__add__`)
- [x] Implement `tensor - other` (`__sub__`)
- [x] Implement `tensor * other` (`__mul__`)
- [x] Implement `tensor / other` (`__truediv__`)
- [x] Implement `tensor // other` (`__floordiv__`) mapping to Div+Floor
- [x] Implement `tensor % other` (`__mod__`)
- [x] Implement `tensor ** other` (`__pow__`)
- [x] Implement `-tensor` (`__neg__`)
- [x] Implement `abs(tensor)` (`__abs__`)
- [x] Implement `tensor == other` (`__eq__`)
- [x] Implement `tensor != other` (`__ne__`)
- [x] Implement `tensor < other` (`__lt__`)
- [x] Implement `tensor <= other` (`__le__`)
- [x] Implement `tensor > other` (`__gt__`)
- [x] Implement `tensor >= other` (`__ge__`)
- [x] Implement `tensor & other` (`__and__`)
- [x] Implement `tensor | other` (`__or__`)
- [x] Implement `tensor ^ other` (`__xor__`)
- [x] Implement `~tensor` (`__invert__`)
- [x] Implement `tensor.add(other)`
- [x] Implement `tensor.sub(other)`
- [x] Implement `tensor.mul(other)`
- [x] Implement `tensor.div(other)`
- [x] Implement `tensor.pow(other)`
- [x] Implement `tensor.exp()`
- [x] Implement `tensor.log()`
- [x] Implement `tensor.sqrt()`
- [x] Implement `tensor.sin()`
- [x] Implement `tensor.cos()`
- [x] Implement `tensor.tan()`
- [x] Implement `tensor.asin()`
- [x] Implement `tensor.acos()`
- [x] Implement `tensor.atan()`
- [x] Implement `tensor.sinh()`
- [x] Implement `tensor.cosh()`
- [x] Implement `tensor.tanh()`
- [x] Implement `tensor.asinh()`
- [x] Implement `tensor.acosh()`
- [x] Implement `tensor.atanh()`
- [x] Implement `tensor.ceil()`
- [x] Implement `tensor.floor()`
- [x] Implement `tensor.round()`
- [x] Implement `tensor.sign()`
- [x] Implement `tensor.erf()`
- [x] Implement `tensor.isnan()`
- [x] Implement `tensor.isinf()`

### 3. Fluent Tensor Shape & Manipulation (40+ items)

- [x] Implement `tensor.reshape(shape)`
- [x] Implement `tensor.view(shape)` (Alias for reshape)
- [x] Implement `tensor.transpose(axes)`
- [x] Implement `tensor.permute(axes)` (Alias for transpose)
- [x] Implement `tensor.T` property (2D transpose)
- [x] Implement `tensor.flatten(axis)`
- [x] Implement `tensor.squeeze(axes)`
- [x] Implement `tensor.unsqueeze(axes)`
- [x] Implement `tensor.shape` property (returning `Shape` node if dynamic)
- [x] Implement `tensor.size()` (returning `Size` node if dynamic)
- [x] Implement `tensor.dim()` (returning rank)
- [x] Implement `onnx9000.concat(tensors, axis)`
- [x] Implement `onnx9000.stack(tensors, axis)`
- [x] Implement `tensor.split(split_size_or_sections, axis)`
- [x] Implement `tensor.chunk(chunks, axis)`
- [x] Implement `tensor.slice(starts, ends, axes, steps)`
- [x] Implement Python `__getitem__` (e.g., `tensor[0:5, :, 1]`) -> `Slice`/`Gather`
- [x] Implement `tensor.gather(indices, axis)`
- [x] Implement `tensor.scatter(indices, updates, axis)`
- [x] Implement `tensor.scatter_nd(indices, updates)`
- [x] Implement `tensor.gather_nd(indices)`
- [x] Implement `tensor.tile(repeats)`
- [x] Implement `tensor.expand(shape)`
- [x] Implement `tensor.pad(pads, mode, value)`
- [x] Implement `tensor.cast(dtype)`
- [x] Implement `tensor.to(dtype)`
- [x] Implement `tensor.astype(dtype)`
- [x] Implement `tensor.where(condition, other)`
- [x] Implement `onnx9000.where(condition, x, y)`
- [x] Implement `tensor.nonzero()`
- [x] Implement `tensor.topk(k, axis, largest, sorted)`
- [x] Implement `tensor.unique(sorted, return_inverse, return_counts)`
- [x] Implement `tensor.cumsum(axis)`
- [x] Implement `tensor.reverse(axis)`
- [x] Implement `tensor.compress(condition, axis)`
- [x] Implement `tensor.trilu(k, upper)`
- [x] Map Python boolean masks `tensor[tensor > 0]` -> `Where`/`NonZero`/`GatherND`
- [x] Map Python `None` cleanly to empty Optional Inputs
- [x] Extract constant dimensions `x.reshape([batch, -1])` cleanly
- [x] Support dynamic shapes cleanly inside list arguments (e.g., `Concat`)

### 4. Fluent Reductions & Neural Network Layers (40+ items)

- [x] Implement `tensor.sum(axis, keepdims)`
- [x] Implement `tensor.mean(axis, keepdims)`
- [x] Implement `tensor.max(axis, keepdims)`
- [x] Implement `tensor.min(axis, keepdims)`
- [x] Implement `tensor.prod(axis, keepdims)`
- [x] Implement `tensor.std(axis, keepdims)` (Subgraph)
- [x] Implement `tensor.var(axis, keepdims)` (Subgraph)
- [x] Implement `tensor.norm(p, axis, keepdims)` (Subgraph)
- [x] Implement `tensor.argmax(axis, keepdims)`
- [x] Implement `tensor.argmin(axis, keepdims)`
- [x] Implement `tensor.argsort(axis, descending)`
- [x] Implement `tensor.any(axis, keepdims)`
- [x] Implement `tensor.all(axis, keepdims)`
- [x] Implement `tensor.matmul(other)`
- [x] Implement `tensor @ other` (`__matmul__`)
- [x] Implement `onnx9000.nn.Linear(in_features, out_features)`
- [x] Implement `onnx9000.nn.Conv1d`
- [x] Implement `onnx9000.nn.Conv2d`
- [x] Implement `onnx9000.nn.Conv3d`
- [x] Implement `onnx9000.nn.ConvTranspose2d`
- [x] Implement `onnx9000.nn.MaxPool2d`
- [x] Implement `onnx9000.nn.AvgPool2d`
- [x] Implement `onnx9000.nn.AdaptiveAvgPool2d`
- [x] Implement `onnx9000.nn.BatchNorm2d`
- [x] Implement `onnx9000.nn.LayerNorm`
- [x] Implement `onnx9000.nn.GroupNorm`
- [x] Implement `onnx9000.nn.InstanceNorm2d`
- [x] Implement `onnx9000.nn.Embedding`
- [x] Implement `onnx9000.nn.RNN`
- [x] Implement `onnx9000.nn.LSTM`
- [x] Implement `onnx9000.nn.GRU`
- [x] Implement `onnx9000.nn.Dropout`
- [x] Implement `tensor.relu()`
- [x] Implement `tensor.leaky_relu(alpha)`
- [x] Implement `tensor.sigmoid()`
- [x] Implement `tensor.tanh()`
- [x] Implement `tensor.softmax(axis)`
- [x] Implement `tensor.log_softmax(axis)`
- [x] Implement `tensor.gelu()`
- [x] Implement `tensor.silu()` / `swish()`

### 5. Advanced ONNX Operators & Attributes Mapping (30+ items)

- [x] Expose `onnx9000.op.Einsum` with string equations
- [x] Expose `onnx9000.op.GridSample`
- [x] Expose `onnx9000.op.RoiAlign`
- [x] Expose `onnx9000.op.MaxRoiPool`
- [x] Expose `onnx9000.op.NonMaxSuppression`
- [x] Expose `onnx9000.op.Resize`
- [x] Expose `onnx9000.op.SpaceToDepth`
- [x] Expose `onnx9000.op.DepthToSpace`
- [x] Expose `onnx9000.op.BitShift`
- [x] Expose `onnx9000.op.Compress`
- [x] Ensure Operator Attributes (`axis`, `keepdims`, `epsilon`) are explicitly strictly typed
- [x] Type-check Attributes natively before ONNX serialization
- [x] Map Python `int` to ONNX `INT`
- [x] Map Python `float` to ONNX `FLOAT`
- [x] Map Python `str` to ONNX `STRING`
- [x] Map Python `list[int]` to ONNX `INTS`
- [x] Map Python `list[float]` to ONNX `FLOATS`
- [x] Map Python `list[str]` to ONNX `STRINGS`
- [x] Map `onnx9000.Tensor` implicitly to `TENSOR` attributes
- [x] Map Python functions to `GRAPH` attributes (for `If`/`Loop`)
- [x] Throw Python exceptions gracefully when attributes miss ONNX spec constraints
- [x] Extract explicit Opsets (`opset=15`) globally across the script decorator
- [x] Check operator availability explicitly against the requested Opset version
- [x] Throw `UnsupportedOperatorError` if the requested op doesn't exist in the target opset
- [x] Expose standard type promotion rules (Int32 + Float32 -> Float32) explicitly
- [x] Handle explicit broadcasting rules across binary ops dynamically
- [x] Provide explicit API `onnx9000.op.CastLike(tensor, target_tensor)`
- [x] Expose custom operator builder `onnx9000.custom_op("domain", "OpName", inputs, attributes)`
- [x] Map Python `complex` numbers to ONNX representations if standard supports it
- [x] Emulate `Spox` strict type/shape invariant checking during node construction

### 6. Shape Inference & Symbolic Trace Validation (25+ items)

- [x] Execute `onnx9000.shape_inference` dynamically during tracing
- [x] Assign explicit output shapes to generated Tensors natively
- [x] Resolve symbolic math variables natively (e.g. `batch * seq`)
- [x] Throw `ShapeError` natively during python tracing if dimensions mismatch (e.g. MatMul conflicts)
- [x] Track tensor `dtype` natively and throw `TypeError` on conflicts
- [x] Generate explicit `ValueInfoProto` for all intermediate tensors natively
- [x] Strip out explicitly undefined dimensions cleanly
- [x] Propagate shape derivations through `Slice` and `Reshape` cleanly
- [x] Propagate type derivations through `Cast` cleanly
- [x] Propagate shapes through `If` branch bodies (checking equality)
- [x] Propagate shapes through `Loop` bodies
- [x] Allow explicit User hints: `tensor.set_shape([1, 224, 224])` to fix inference
- [x] Allow explicit User hints: `tensor.set_type(onnx9000.float32)`
- [x] Check outputs match defined return annotations cleanly
- [x] Check inputs match defined input annotations cleanly
- [x] Validate `ai.onnx.ml` domain topological constraints strictly
- [x] Provide pure-Python alternative to `onnx.checker.check_model` running automatically on export
- [x] Catch explicitly invalid ONNX graph constructs (disconnected nodes) before saving
- [x] Expose an API to print a rich ASCII DAG summary of the traced function
- [x] Execute a dry-run inference pass completely inside Python using dummy constants
- [x] Output structural diagnostics matching `Spox` verbosity
- [x] Test the inference stability with dynamic batch sizes (e.g. `N`)
- [x] Evaluate tensor indexing (`tensor[0:N]`) bounds correctly
- [x] Handle sequence structures explicitly (`SequenceConstruct` shape tracking)
- [x] Validate `SplitToSequence` output shapes

### 7. Zero-Dependency Browser & Serverless Execution (30+ items)

- [x] Export `onnx9000.script` decorator specifically for Web/Pyodide usage
- [x] Ensure 0 bytes of C++ dependencies (No `onnx`, no `protobuf` PIP packages)
- [x] Prevent usage of native Python `ctypes` bindings during the tracing process
- [x] Validate multi-GB Constant extraction without Python MemoryError
- [x] Map Numpy array Constants directly to `TensorProto` bytes explicitly in memory
- [x] Map native Python lists directly to `TensorProto` bytes explicitly
- [x] Serialize entirely via pure-Python `struct` and standard encoding methods
- [x] Serialize standard `.safetensors` payloads alongside the `.onnx` model explicitly
- [x] Expose an interactive Python terminal (Jupyter/Pyodide) compatible tracing object
- [x] Support importing pre-existing `.onnx` models, appending nodes, and re-exporting
- [x] Expose `GraphBuilder.append(other_graph)` seamlessly
- [x] Support generating isolated `FunctionProto` sub-graphs (reusable functions)
- [x] Map Python `@onnx9000.function` explicitly to ONNX `FunctionProto` abstractions
- [x] Reuse `FunctionProto` references inside the main Graph sequentially
- [x] Embed the exact versions of `onnx9000` into `ModelProto.producer_version`
- [x] Output purely binary payloads for HTTP streaming
- [x] Output purely Base64 strings for direct Javascript `<script>` injection
- [x] Ensure execution latency of the trace is < 50ms for a ResNet topology
- [x] Output raw structural JSON dynamically to debug traces visually (Netron integration)
- [x] Prevent memory leaks by explicitly cleaning up `GraphBuilder` state on exception
- [x] Support AWS Lambda instantaneous cold-start model generation
- [x] Support Cloudflare Worker compatible JS API mapping (if transpiled)
- [x] Allow exporting directly to a Python bytearray (`io.BytesIO`)
- [x] Map Python integers securely to 64-bit bounds without overflow
- [x] Encode Unicode strings accurately for ONNX String tensors
- [x] Check maximum Protobuf hard-limits (2GB) and strictly enforce `.safetensors` external data usage
- [x] Implement chunked writing to disk to prevent OOM on massive model saves
- [x] Check explicitly for file permissions and OS limits during local exports
- [x] Provide explicit WASM browser testing validation checks
- [x] Validate backwards compatibility (loading the generated model using original `onnxruntime` C++)

### 8. Extensive Testing & Edge Cases (30+ items)

- [x] Unit Test: Trace `Add(A, B)` and evaluate exact binary equivalence with `onnx` package
- [x] Unit Test: Trace `MatMul` with implicit broadcasting
- [x] Unit Test: Trace `Reshape` utilizing `Shape` symbolic extraction
- [x] Unit Test: Trace a complex `If` condition (e.g. `If ReduceSum(x) > 0`)
- [x] Unit Test: Trace a custom `Loop` iterating exactly 10 times
- [x] Unit Test: Trace a standard Multi-Layer Perceptron (MLP)
- [x] Unit Test: Trace a standard Convolutional Neural Network (CNN)
- [x] Unit Test: Trace an auto-regressive text generator (GPT-like loop)
- [x] Unit Test: Extract parameters correctly from nested `onnx9000.nn.Module` classes
- [x] Unit Test: Validate explicit Operator Fallbacks (Opset 13 vs Opset 15 specific nodes)
- [x] Test 0D Scalar Python inputs mapped to ONNX Scalar dimensions `[]`
- [x] Test 1D Python list inputs mapped to `[N]` dimensions
- [x] Prevent recursive depth limits (e.g. Maximum Recursion Depth Exceeded) on 1000+ node chains
- [x] Catch explicitly nested tuples `((A, B), C)` and unpack them cleanly
- [x] Support tracing `dict` inputs safely `def forward(inputs: dict[str, Tensor])`
- [x] Catch dynamic Python loops `for i in range(10): x = x + 1` (Unrolling explicitly)
- [x] Warn when a Python loop is excessively unrolled (>1000 nodes generated)
- [x] Test Python list comprehensions `[x + 1 for x in tensors]`
- [x] Prevent `RuntimeError` on disconnected `Constant` nodes
- [x] Handle PyTorch `__dlpack__` / `__array__` interface compatibility for seamless interoperability
- [x] Validate PyTorch to ONNX9000 conversion script accuracy
- [x] Evaluate `np.nan` and `np.inf` Constant encoding exactly
- [x] Test numerical limits of Float16 Constant packing explicitly in pure Python
- [x] Ensure Graph Outputs mapped directly from Inputs (Identity mapping) are valid
- [x] Expose an API to run `onnx9000.simplify()` implicitly during the `script` export
- [x] Validate the topological sort explicitly places `Constant` nodes before consumers
- [x] Check `If` branch sub-graphs do not contain illegal overlapping names
- [x] Prevent `Loop` body sub-graphs from overriding parent graph constants illegally
- [x] Validate execution against all `ai.onnx` standard test cases dynamically
- [x] Provide comprehensive `TypeError` messages containing explicit file/line numbers (Stack Traces)

### 9. Spox Specific Emulation & Extended Validations (25+ items)

- [x] Implement `Spox`-style strong typing (`Var` validation during instantiation)
- [x] Support `Spox` strict `type` annotations (rejecting construction if `Float32` is fed to `Int64` op)
- [x] Support `Spox` strict `shape` annotations (rejecting construction if `[1, 3]` is fed to `[10]`)
- [x] Map Python `None` cleanly to Spox `Optional` input signatures
- [x] Evaluate Spox-style nested subgraphs natively within `@onnx9000.script` bounds
- [x] Extract Spox `inline` capabilities natively (embedding another `@script` directly into the current one)
- [x] Support explicit node renaming at the time of authoring (`node = x.add(y, name="MyAdd")`)
- [x] Provide explicit Spox `build` methods simulating `onnx9000.export` seamlessly
- [x] Catch dynamic Python dictionaries (`{"a": x, "b": y}`) as valid ONNX Sequence/Map constructs if supported by target opset
- [x] Map Python `set` to ONNX `Sequence` or `Map` selectively
- [x] Support PyTorch-like `nn.ParameterList` equivalent abstraction in `onnx9000`
- [x] Support PyTorch-like `nn.ModuleDict` equivalent abstraction
- [x] Ensure all `onnx9000` functions support `*args` and `**kwargs` unpacking correctly
- [x] Support mapping Python variadic arguments (`*args`) directly to ONNX variadic inputs (e.g. `Concat(*args)`)
- [x] Raise explicit `ValueError` if keyword arguments do not match expected ONNX attributes
- [x] Trace recursive Python functions securely (with a hard depth limit, e.g. 50, to prevent infinite graphs)
- [x] Convert `dict` returns (`return {"out": x}`) strictly to named ONNX graph outputs
- [x] Provide a purely procedural, non-decorator API `graph.add_node("Add", inputs=[A, B], outputs=[C])`
- [x] Prevent graph mutation during active iteration/serialization phases
- [x] Validate `dlpack` ingestion from PyTorch directly traces as a `Constant` natively
- [x] Validate `dlpack` ingestion from JAX directly traces as a `Constant` natively
- [x] Support Python explicit `list` -> ONNX `Sequence` translation explicitly if typed as such
- [x] Extract variable names dynamically using `inspect` frame analysis (e.g. `my_var = x+1` names the node `my_var`)
- [x] Guarantee serialization of 100+ layer Transformer graphs in under 200ms natively
- [x] Emit detailed stack traces of original Python files into ONNX `doc_string` explicitly for debugging
