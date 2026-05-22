# Torch & TF Exporters Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of the PyTorch exporter (`torch.onnx`) within the `onnx9000` ecosystem.
Standard `torch.onnx.export` relies on massive C++ dependencies (`libtorch`) to trace operations and translate ATen (PyTorch's internal C++ tensor library) to ONNX.
Our `onnx9000` reimplementation completely bypasses C++ tracing. It provides a pure-Python, PyTorch-like API and leverages pure-Python `torch.fx` (Dynamo) graph extraction to map PyTorch operations directly to ONNX. This lightweight architecture allows model conversion to execute seamlessly in the browser via Pyodide, or instantly on edge devices, completely removing the gigabytes of C++ overhead typically required to export a model.

## Exhaustive Parity Checklist

### 1. Core Export API & Architecture (30+ items)

- [xx] Implement pure-Python `onnx9000.export(model, args, f)`
- [xx] Implement pure-Python `onnx9000.dynamo_export(model, args)`
- [xx] Support `export_params=True` (embedding weights as initializers)
- [xx] Support `export_params=False` (weights as graph inputs)
- [xx] Support `verbose=True` for topological logging
- [xx] Support `training=TrainingMode.EVAL` strictly
- [xx] Support `training=TrainingMode.TRAINING`
- [xx] Support `input_names` lists for explicit input naming
- [xx] Support `output_names` lists for explicit output naming
- [xx] Support `dynamic_axes` dictionaries for symbolic dimension overrides
- [xx] Support `keep_initializers_as_inputs=True/False`
- [xx] Support explicit ONNX `opset_version` targeting (Opsets 9 through 21)
- [xx] Extract PyTorch `nn.Module` state dictionaries entirely in Python memory
- [xx] Extract nested `nn.Module` sub-modules tracking namespaces (e.g., `layer.0.conv`)
- [xx] Map `torch.Tensor` data seamlessly to `onnx9000.Tensor` (via Pyodide or DLPack)
- [xx] Wrap PyTorch `__torch_function__` natively to trace ops without `libtorch`
- [xx] Translate PyTorch `torch.dtype` (e.g., `torch.float32`) to ONNX `TensorProto.DataType`
- [xx] Translate `torch.bfloat16` to ONNX `BFLOAT16`
- [xx] Translate `torch.complex64` gracefully (or throw explicit unsupported errors)
- [xx] Handle PyTorch arbitrary nested tuples/lists/dicts as model inputs
- [xx] Handle PyTorch arbitrary nested tuples/lists/dicts as model outputs
- [xx] Implement PyTorch `FakeTensor` tracing compatibility natively
- [xx] Support `custom_opsets` dictionary for domain-specific mappings
- [xx] Catch un-traced or pure Python branching (`if x.sum() > 0`) dynamically and warn
- [xx] Intercept `torch.autograd.Function` custom backward definitions (skip in eval mode)
- [xx] Emit ONNX `ModelProto` purely in Python
- [xx] Export multi-GB models safely using external data format (`.bin`) without OOM
- [xx] Provide drop-in replacement API `onnx9000.nn.Module` for fully native non-torch execution
- [xx] Emulate `torch.no_grad()` scoping implicitly during export
- [xx] Ensure strict adherence to PyTorch `fx.GraphModule` topological traversal

### 2. ATen Math & Elementwise Operator Mappings (50+ items)

- [xx] Map `aten::add` -> `Add`
- [xx] Map `aten::add_` (inplace) -> `Add`
- [xx] Map `aten::sub` -> `Sub`
- [xx] Map `aten::sub_` -> `Sub`
- [xx] Map `aten::mul` -> `Mul`
- [xx] Map `aten::mul_` -> `Mul`
- [xx] Map `aten::div` -> `Div`
- [xx] Map `aten::div_` -> `Div`
- [xx] Map `aten::floor_divide` -> `Div` + `Floor`
- [xx] Map `aten::fmod` -> `Mod`
- [xx] Map `aten::remainder` -> `Mod`
- [xx] Map `aten::abs` -> `Abs`
- [xx] Map `aten::abs_` -> `Abs`
- [xx] Map `aten::neg` -> `Neg`
- [xx] Map `aten::neg_` -> `Neg`
- [xx] Map `aten::exp` -> `Exp`
- [xx] Map `aten::expm1` -> `Exp` + `Sub(1)`
- [xx] Map `aten::log` -> `Log`
- [xx] Map `aten::log10` -> `Log` / `Log(10)`
- [xx] Map `aten::log1p` -> `Add(1)` + `Log`
- [xx] Map `aten::log2` -> `Log` / `Log(2)`
- [xx] Map `aten::sin` -> `Sin`
- [xx] Map `aten::cos` -> `Cos`
- [xx] Map `aten::tan` -> `Tan`
- [xx] Map `aten::asin` -> `Asin`
- [xx] Map `aten::acos` -> `Acos`
- [xx] Map `aten::atan` -> `Atan`
- [xx] Map `aten::atan2` -> Custom Subgraph / Math
- [xx] Map `aten::sinh` -> `Sinh`
- [xx] Map `aten::cosh` -> `Cosh`
- [xx] Map `aten::tanh` -> `Tanh`
- [xx] Map `aten::asinh` -> `Asinh`
- [xx] Map `aten::acosh` -> `Acosh`
- [xx] Map `aten::atanh` -> `Atanh`
- [xx] Map `aten::pow` -> `Pow`
- [xx] Map `aten::sqrt` -> `Sqrt`
- [xx] Map `aten::rsqrt` -> `Sqrt` + `Reciprocal`
- [xx] Map `aten::ceil` -> `Ceil`
- [xx] Map `aten::floor` -> `Floor`
- [xx] Map `aten::round` -> `Round`
- [xx] Map `aten::trunc` -> `Cast` (int)
- [xx] Map `aten::sign` -> `Sign`
- [xx] Map `aten::erf` -> `Erf`
- [xx] Map `aten::erfc` -> `Sub(1, Erf)`
- [xx] Map `aten::reciprocal` -> `Reciprocal`
- [xx] Map `aten::bitwise_not` -> `BitwiseNot`
- [xx] Map `aten::bitwise_and` -> `BitwiseAnd`
- [xx] Map `aten::bitwise_or` -> `BitwiseOr`
- [xx] Map `aten::bitwise_xor` -> `BitwiseXor`
- [xx] Map `aten::bitwise_left_shift` -> `BitShift`
- [xx] Map `aten::bitwise_right_shift` -> `BitShift`
- [xx] Resolve implicit `alpha` scaling in `aten::add(a, b, alpha=2)` -> `Add(a, Mul(b, 2))`

### 3. ATen Neural Network Layer Mappings (40+ items)

- [xx] Map `aten::linear` -> `MatMul` + `Add` or `Gemm`
- [xx] Map `aten::conv1d` -> `Conv`
- [xx] Map `aten::conv2d` -> `Conv`
- [xx] Map `aten::conv3d` -> `Conv`
- [xx] Map `aten::conv_transpose1d` -> `ConvTranspose`
- [xx] Map `aten::conv_transpose2d` -> `ConvTranspose`
- [xx] Map `aten::conv_transpose3d` -> `ConvTranspose`
- [xx] Map `aten::_convolution` -> `Conv`
- [xx] Map `aten::batch_norm` -> `BatchNormalization`
- [xx] Map `aten::native_batch_norm` -> `BatchNormalization`
- [xx] Map `aten::layer_norm` -> `LayerNormalization`
- [xx] Map `aten::native_layer_norm` -> `LayerNormalization`
- [xx] Map `aten::group_norm` -> `Reshape` + `InstanceNormalization` + `Reshape`
- [xx] Map `aten::instance_norm` -> `InstanceNormalization`
- [xx] Map `aten::max_pool1d` -> `MaxPool`
- [xx] Map `aten::max_pool2d` -> `MaxPool`
- [xx] Map `aten::max_pool2d_with_indices` -> `MaxPool` (2 outputs)
- [xx] Map `aten::max_pool3d` -> `MaxPool`
- [xx] Map `aten::avg_pool1d` -> `AveragePool`
- [xx] Map `aten::avg_pool2d` -> `AveragePool`
- [xx] Map `aten::avg_pool3d` -> `AveragePool`
- [xx] Map `aten::adaptive_avg_pool1d` -> `GlobalAveragePool` / Custom
- [xx] Map `aten::adaptive_avg_pool2d` -> `GlobalAveragePool`
- [xx] Map `aten::adaptive_max_pool1d` -> `GlobalMaxPool` / Custom
- [xx] Map `aten::adaptive_max_pool2d` -> `GlobalMaxPool`
- [xx] Map `aten::reflection_pad1d` -> `Pad` (reflect)
- [xx] Map `aten::reflection_pad2d` -> `Pad` (reflect)
- [xx] Map `aten::replication_pad1d` -> `Pad` (edge)
- [xx] Map `aten::replication_pad2d` -> `Pad` (edge)
- [xx] Map `aten::constant_pad_nd` -> `Pad` (constant)
- [xx] Map `aten::dropout` -> `Dropout` or `Identity`
- [xx] Map `aten::dropout_` -> `Dropout` or `Identity`
- [xx] Map `aten::feature_dropout` -> `Dropout` or `Identity`
- [xx] Map `aten::alpha_dropout` -> `Dropout` or `Identity`
- [xx] Map `aten::rnn_tanh` -> `RNN`
- [xx] Map `aten::rnn_relu` -> `RNN`
- [xx] Map `aten::lstm` -> `LSTM`
- [xx] Map `aten::gru` -> `GRU`
- [xx] Map `aten::embedding` -> `Gather`
- [xx] Map `aten::embedding_bag` -> `Gather` + `ReduceSum`

### 4. ATen Activation Functions Mappings (25+ items)

- [xx] Map `aten::relu` -> `Relu`
- [xx] Map `aten::relu_` -> `Relu`
- [xx] Map `aten::relu6` -> `Clip(0, 6)`
- [xx] Map `aten::leaky_relu` -> `LeakyRelu`
- [xx] Map `aten::leaky_relu_` -> `LeakyRelu`
- [xx] Map `aten::elu` -> `Elu`
- [xx] Map `aten::elu_` -> `Elu`
- [xx] Map `aten::selu` -> `Selu`
- [xx] Map `aten::selu_` -> `Selu`
- [xx] Map `aten::celu` -> `Celu`
- [xx] Map `aten::gelu` -> `Gelu` or `Erf` subgraph
- [xx] Map `aten::silu` -> `Mul(X, Sigmoid(X))` (Swish)
- [xx] Map `aten::silu_` -> `Mul(X, Sigmoid(X))`
- [xx] Map `aten::mish` -> `Mish`
- [xx] Map `aten::sigmoid` -> `Sigmoid`
- [xx] Map `aten::sigmoid_` -> `Sigmoid`
- [xx] Map `aten::hardsigmoid` -> `HardSigmoid`
- [xx] Map `aten::hardswish` -> `HardSwish`
- [xx] Map `aten::hardswish_` -> `HardSwish`
- [xx] Map `aten::tanh` -> `Tanh`
- [xx] Map `aten::tanh_` -> `Tanh`
- [xx] Map `aten::softplus` -> `Softplus`
- [xx] Map `aten::softsign` -> `Softsign`
- [xx] Map `aten::softmax` -> `Softmax`
- [xx] Map `aten::log_softmax` -> `LogSoftmax`

### 5. ATen Tensor Manipulation & Shape Mappings (45+ items)

- [xx] Map `aten::view` -> `Reshape`
- [xx] Map `aten::reshape` -> `Reshape`
- [xx] Map `aten::transpose` -> `Transpose`
- [xx] Map `aten::t` -> `Transpose`
- [xx] Map `aten::permute` -> `Transpose`
- [xx] Map `aten::squeeze` -> `Squeeze`
- [xx] Map `aten::squeeze_` -> `Squeeze`
- [xx] Map `aten::unsqueeze` -> `Unsqueeze`
- [xx] Map `aten::unsqueeze_` -> `Unsqueeze`
- [xx] Map `aten::flatten` -> `Flatten`
- [xx] Map `aten::unflatten` -> `Reshape`
- [xx] Map `aten::cat` -> `Concat`
- [xx] Map `aten::stack` -> `Unsqueeze` + `Concat`
- [xx] Map `aten::split` -> `Split`
- [xx] Map `aten::split_with_sizes` -> `Split`
- [xx] Map `aten::chunk` -> `Split`
- [xx] Map `aten::slice` -> `Slice`
- [xx] Map `aten::gather` -> `GatherElements`
- [xx] Map `aten::scatter` -> `ScatterElements`
- [xx] Map `aten::scatter_add` -> `ScatterElements` (with Add reduction)
- [xx] Map `aten::scatter_` -> `ScatterElements`
- [xx] Map `aten::index_select` -> `Gather`
- [xx] Map `aten::index_put` -> `ScatterND`
- [xx] Map `aten::masked_select` -> `NonZero` + `GatherND`
- [xx] Map `aten::masked_fill` -> `Where`
- [xx] Map `aten::masked_fill_` -> `Where`
- [xx] Map `aten::where` -> `Where`
- [xx] Map `aten::expand` -> `Expand`
- [xx] Map `aten::expand_as` -> `Shape` + `Expand`
- [xx] Map `aten::repeat` -> `Tile`
- [xx] Map `aten::repeat_interleave` -> Subgraph
- [xx] Map `aten::broadcast_to` -> `Expand`
- [xx] Map `aten::broadcast_tensors` -> Multiple `Expand`
- [xx] Map `aten::contiguous` -> Identity (No-op in ONNX)
- [xx] Map `aten::clone` -> Identity
- [xx] Map `aten::to` -> `Cast`
- [xx] Map `aten::type_as` -> `CastLike`
- [xx] Map `aten::size` -> `Shape`
- [xx] Map `aten::numel` -> `Size`
- [xx] Map `aten::dim` -> `Shape` + `Size`
- [xx] Map `aten::zeros` -> `ConstantOfShape`
- [xx] Map `aten::zeros_like` -> `Shape` + `ConstantOfShape`
- [xx] Map `aten::ones` -> `ConstantOfShape`
- [xx] Map `aten::ones_like` -> `Shape` + `ConstantOfShape`
- [xx] Map `aten::full` -> `ConstantOfShape`
- [xx] Map `aten::full_like` -> `Shape` + `ConstantOfShape`

### 6. ATen Reductions, Logical & Linear Algebra (40+ items)

- [xx] Map `aten::sum` -> `ReduceSum`
- [xx] Map `aten::mean` -> `ReduceMean`
- [xx] Map `aten::max` -> `ReduceMax` (reduction) or `Max` (elementwise)
- [xx] Map `aten::min` -> `ReduceMin` (reduction) or `Min` (elementwise)
- [xx] Map `aten::prod` -> `ReduceProd`
- [xx] Map `aten::std` -> Subgraph
- [xx] Map `aten::var` -> Subgraph
- [xx] Map `aten::norm` -> `ReduceL2` or `ReduceL1`
- [xx] Map `aten::any` -> `ReduceMax` (bool)
- [xx] Map `aten::all` -> `ReduceMin` (bool)
- [xx] Map `aten::argmax` -> `ArgMax`
- [xx] Map `aten::argmin` -> `ArgMin`
- [xx] Map `aten::topk` -> `TopK`
- [xx] Map `aten::sort` -> `TopK` (K=all)
- [xx] Map `aten::argsort` -> `TopK` (indices)
- [xx] Map `aten::kthvalue` -> `TopK` (K=k)
- [xx] Map `aten::unique` -> `Unique`
- [xx] Map `aten::eq` -> `Equal`
- [xx] Map `aten::ne` -> `Equal` + `Not`
- [xx] Map `aten::lt` -> `Less`
- [xx] Map `aten::le` -> `LessOrEqual`
- [xx] Map `aten::gt` -> `Greater`
- [xx] Map `aten::ge` -> `GreaterOrEqual`
- [xx] Map `aten::logical_and` -> `And`
- [xx] Map `aten::logical_or` -> `Or`
- [xx] Map `aten::logical_not` -> `Not`
- [xx] Map `aten::logical_xor` -> `Xor`
- [xx] Map `aten::isnan` -> `IsNaN`
- [xx] Map `aten::isinf` -> `IsInf`
- [xx] Map `aten::bmm` -> `MatMul`
- [xx] Map `aten::matmul` -> `MatMul`
- [xx] Map `aten::mm` -> `MatMul`
- [xx] Map `aten::addmm` -> `Gemm`
- [xx] Map `aten::addbmm` -> `MatMul` + `Add`
- [xx] Map `aten::baddbmm` -> `MatMul` + `Add`
- [xx] Map `aten::einsum` -> `Einsum`
- [xx] Map `aten::dot` -> `Mul` + `ReduceSum`
- [xx] Map `aten::tensordot` -> `MatMul` / Subgraph
- [xx] Map `aten::cross` -> Subgraph
- [xx] Map `aten::diag` -> `EyeLike` + `Mul`
- [xx] Map `aten::tril` -> `Trilu`
- [xx] Map `aten::triu` -> `Trilu`

### 7. TorchScript, Dynamo & Control Flow (25+ items)

- [xx] Implement `prim::If` -> `If`
- [xx] Implement `prim::Loop` -> `Loop`
- [xx] Implement `prim::ListConstruct` -> `SequenceConstruct`
- [xx] Implement `prim::TupleConstruct` -> Flattened multi-outputs
- [xx] Implement `prim::TupleUnpack` -> Flattened multi-inputs
- [xx] Implement `prim::ListUnpack` -> `SplitToSequence` / Subgraph
- [xx] Implement `prim::DictConstruct` -> Dictionary serialization map
- [xx] Handle `torch.jit.script` graphs seamlessly via ATen IR conversion
- [xx] Handle `torch.fx.GraphModule` dynamically via Dynamo export pathway
- [xx] Traverse PyTorch `torch.fx.Node` topologies translating exact ATen ops
- [xx] Support handling `operator.add` natively inside FX graphs
- [xx] Support handling `operator.mul` natively inside FX graphs
- [xx] Support handling `operator.getitem` natively inside FX graphs
- [xx] Translate `torch.ops.aten.add.Tensor` strictly to ONNX `Add`
- [xx] Resolve Python `math.pi` and `math.e` natively to `Constant`
- [xx] Inline standard pure Python functions seamlessly during Dynamo tracing
- [xx] Strip `.backward()` hooks entirely during trace
- [xx] Strip `torch.profiler` hooks entirely during trace
- [xx] Flatten `torch.nn.ModuleList` iterations into sequential topologies natively
- [xx] Flatten `torch.nn.Sequential` iterations natively
- [xx] Validate `torch.nn.Parameter` gradients are frozen (`requires_grad=False`) for export
- [xx] Map `torch.Tensor.item()` calls natively (may require tracing breaks or constants)
- [xx] Map `torch.Tensor.tolist()` calls natively
- [xx] Emit specific Warnings when Python control flow forces an unrolled graph
- [xx] Support exporting models with custom C++ ops via explicit `custom_opsets` definitions

### 8. Web/Pyodide & Zero-Dependency Validations (20+ items)

- [xx] CLI fully operational without installing the pip `torch` package (requires providing `onnx9000.nn` equivalent models)
- [xx] Execute `onnx9000.export` seamlessly inside the browser (WASM/Pyodide)
- [xx] Load and trace PyTorch `.pt` / `.pth` state dictionaries purely via Python `pickle` parsers natively without `torch`
- [xx] Map `OrderedDict` state dicts directly to ONNX `Initializer` tensors
- [xx] Parse `__torch_function__` dispatch hooks natively inside Pyodide without C++ crashes
- [xx] Ensure the Python tracing module fits within 2MB for Cloudflare Worker deployments
- [xx] Validate multi-gigabyte models can be exported sequentially without OOM
- [xx] Test exporting a ResNet50 model directly inside Chrome
- [xx] Test exporting a Transformer model (BERT) directly inside Safari
- [xx] Provide pure-Python polyfills for PyTorch's `Tensor` shape/stride classes
- [xx] Provide pure-Python implementations of DLPack for zero-copy numpy conversions
- [xx] Optimize dictionary lookup bottlenecks during `fx.Graph` topological sorting
- [xx] Validate serialization of PyTorch `bfloat16` byte structures cleanly in JS/Python
- [xx] Parse PyTorch `qint8` and `quint8` quantized tensor definitions seamlessly
- [xx] Resolve memory fragmentation by actively garbage collecting ATen variables during the trace
- [xx] Execute completely synchronously if requested (no async barriers during export)
- [xx] Test `onnx9000` PyTorch exporter against official `torch.onnx.export` (graph isomorphism checks)
- [xx] Guarantee mathematical equivalence (atol=1e-5) between PyTorch CPU execution and exported ONNX CPU execution
- [xx] Implement `onnx9000.dynamo_export` compatibility with PyTorch 2.0+ `torch.compile` internals
- [xx] Export `model.safetensors` mappings natively alongside the ONNX topology implicitly

### 9. Vision, Audio & Advanced Operations (25+ items)

- [xx] Map `torchvision.ops.nms` -> `NonMaxSuppression`
- [xx] Map `torchvision.ops.roi_align` -> `RoiAlign`
- [xx] Map `torchvision.ops.roi_pool` -> `MaxRoiPool`
- [xx] Map `torchvision.ops.deform_conv2d` -> Custom Subgraph
- [xx] Map `torch.nn.functional.interpolate` (nearest) -> `Resize`
- [xx] Map `torch.nn.functional.interpolate` (bilinear) -> `Resize`
- [xx] Map `torch.nn.functional.interpolate` (bicubic) -> `Resize`
- [xx] Map `torch.nn.functional.grid_sample` -> `GridSample` / Subgraph
- [xx] Map `torch.nn.functional.affine_grid` -> Subgraph
- [xx] Map `torch.nn.functional.pixel_shuffle` -> `DepthToSpace`
- [xx] Map `torch.nn.functional.pixel_unshuffle` -> `SpaceToDepth`
- [xx] Map `torch.fft.rfft` -> Custom Subgraph or Extension
- [xx] Map `torch.fft.irfft` -> Custom Subgraph or Extension
- [xx] Map `torchaudio.functional.spectrogram` -> Subgraph
- [xx] Map `torch.nn.functional.scaled_dot_product_attention` -> Math Subgraph or `Attention`
- [xx] Map `torch.nn.functional.pad` (constant) -> `Pad`
- [xx] Map `torch.nn.functional.pad` (reflect) -> `Pad`
- [xx] Map `torch.nn.functional.pad` (replicate) -> `Pad`
- [xx] Map `torch.nn.functional.fold` -> `Col2Im`
- [xx] Map `torch.nn.functional.unfold` -> Im2Col Subgraph
- [xx] Map `torch.nn.functional.one_hot` -> `OneHot`
- [xx] Map `torch.nn.functional.embedding` -> `Gather`
- [xx] Map `torch.nn.functional.cosine_similarity` -> Subgraph
- [xx] Map `torch.nn.functional.pdist` -> Subgraph (Pairwise distance)
- [xx] Map `torch.distributions.uniform.Uniform` -> `RandomUniform`
- [xx] Map `torch.distributions.normal.Normal` -> `RandomNormal`

### 10. Graph Optimizations & Export Post-Processing (20+ items)

- [xx] Fold static PyTorch shapes (`x.shape[0]`) into `Constant` nodes automatically
- [xx] Remove `aten::contiguous` operations completely (No-op in ONNX)
- [xx] Remove `aten::to` operations where `dtype` is identical
- [xx] Remove `aten::clone` operations completely
- [xx] Fuse `aten::batch_norm` into `aten::conv2d` natively during export if requested
- [xx] Optimize sequential `aten::transpose` calls statically during tracing
- [xx] Clean up redundant `aten::view` nodes immediately after generation
- [xx] Identify isolated non-differentiable subgraphs and bake them into constants
- [xx] Track tensor symbolic relationships (`batch_size`, `seq_len`) perfectly from `dynamic_axes` definitions
- [xx] Generate standard ONNX ValueInfo structures automatically based on PyTorch intermediate shapes
- [xx] Validate the generated ONNX Graph using `onnx9000` internal GraphSurgeon tools natively
- [xx] Apply `onnx9000` shape inference pass implicitly before saving the exported file
- [xx] Replace PyTorch explicit `float64` casts with `float32` selectively if `--fp32-export` is passed
- [xx] Support pruning un-used submodules from the traced graph seamlessly
- [xx] Remove explicitly `aten::dropout` ops natively if tracing in eval mode
- [xx] Apply Constant Folding to all scalar math generated by Python indexing logic natively
- [xx] Provide interactive traceback logs mapping exact PyTorch lines of code to generated ONNX nodes
- [xx] Format PyTorch stack traces securely inside `ModelProto.doc_string`
- [xx] Export directly to `.onnx` byte buffer for instant HTTP transmission or WebSocket delivery
- [xx] Guarantee the generated `.onnx` requires ZERO external pre-processing to run in `onnxruntime-web`
