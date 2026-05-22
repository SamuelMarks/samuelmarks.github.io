# paddle2onnx Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `paddle2onnx` within the `onnx9000` ecosystem.
Unlike the original project, which binds tightly to the native C++ `paddlepaddle` framework, this implementation uses a zero-dependency, pure-Python parser to read Paddle's internal protobuf (`.pdmodel`) and binary parameter (`.pdiparams`) formats.
By decoupling the converter from the massive native Paddle engine, it can execute entirely inside a web browser (via WASM/WebGPU) or run in high-concurrency server environments instantly without bloated container images.

## Exhaustive Parity Checklist

### 1. Pure-Python Parsers & Loaders (Zero-Dependency)

- [x] Implement pure-Python parser for Paddle Inference Model (`.pdmodel`) Protobuf format
- [x] Implement pure-Python parser for Paddle Inference Parameters (`.pdiparams`) binary format
- [x] Implement pure-Python parser for Legacy Paddle Inference (`__model__`) Protobuf format
- [x] Implement pure-Python parser for Legacy Paddle Inference (`__params__`) binary format
- [x] Support loading Paddle's combined model format
- [x] Support loading Paddle's non-combined (separated files) model format
- [x] Implement zero-copy weight extraction mapping binary blobs directly to ONNX `TensorProto`
- [x] Extract variables, constants, and tensors from Paddle `BlockDesc`
- [x] Extract operations from Paddle `OpDesc`
- [x] Extract inputs/outputs (feeds/fetches) recursively from `ProgramDesc`
- [x] Support overriding feed (input) shapes dynamically
- [x] Support overriding fetch (output) targets dynamically
- [x] Handle multi-block architectures natively
- [x] Reconstruct sub-graphs from `BlockDesc` indices
- [x] Parse LoD (Level of Detail) Tensor metadata
- [x] Extract and map string parameters
- [x] Identify and parse quantized model schemas
- [x] Implement Paddle variable type resolution (e.g., `LOD_TENSOR`, `LOD_TENSOR_ARRAY`)
- [x] Handle missing explicit data types using inferred shapes/types from downstream ops
- [x] Decrypt / Handle Paddle encrypted model formats (if keys provided)

### 2. Graph Pre-Processing & Topology Adjustments

- [x] Eliminate dead code and unconnected ops
- [x] Prune backward pass / training specific ops (e.g., `sgd`, `adam`)
- [x] Resolve variable renaming conflicts between blocks
- [x] Promote block-local variables to global graph inputs where necessary
- [x] Infer ONNX symbolic dimensions for Paddle `-1` dynamic shapes
- [x] Implement continuous shape inference to patch missing output shapes
- [x] Resolve scalar vs 1D tensor mismatches (Paddle treats some scalars as 1D)
- [x] Standardize Paddle's implicit broadcasting rules to ONNX explicit broadcasing
- [x] Handle Paddle's in-place operations by creating pure SSA (Static Single Assignment) mappings
- [x] Strip out dropout operations during inference mode translation
- [x] Map Paddle data types (`VAR_TYPE.FP32`, `VAR_TYPE.INT64`, etc.) to ONNX `TensorProto.DataType`
- [x] Handle `VAR_TYPE.FP16` and `VAR_TYPE.BF16` conversion
- [x] Extract and bake-in `scale` and `bias` tensors that Paddle leaves floating

### 3. Control Flow & LoDTensor Mapping

- [x] Map Paddle `while` op -> ONNX `Loop`
- [x] Map Paddle `conditional_block` -> ONNX `If`
- [x] Map Paddle `select_input` -> ONNX `If` or `Where`
- [x] Map Paddle `select_output` -> ONNX `If` or `Where`
- [x] Map Paddle `is_empty` -> ONNX `Size` + `Equal(0)`
- [x] Map Paddle `increment` -> ONNX `Add`
- [x] Map Paddle `assign` -> ONNX `Identity`
- [x] Map Paddle `assign_value` -> ONNX `Constant`
- [x] Map Paddle `tensor_array_to_tensor` -> ONNX `SequenceConstruct` + `Concat`
- [x] Map Paddle `write_to_array` -> ONNX `SequenceInsert`
- [x] Map Paddle `read_from_array` -> ONNX `SequenceAt`
- [x] Map Paddle `lod_reset` -> ONNX Custom or Sequence mapping
- [x] Map Paddle `lod_append` -> ONNX Custom or Sequence mapping
- [x] Map Paddle `sequence_pool` -> ONNX Sequence operators + Reduce
- [x] Map Paddle `sequence_conv` -> ONNX Sequence operators + Conv
- [x] Map Paddle `sequence_softmax` -> ONNX Sequence operators + Softmax
- [x] Map Paddle `sequence_concat` -> ONNX `ConcatFromSequence`
- [x] Map Paddle `sequence_expand` -> ONNX `Sequence` expansion
- [x] Map Paddle `sequence_expand_as` -> ONNX `Sequence` expansion
- [x] Map Paddle `sequence_pad` -> ONNX `Pad`
- [x] Map Paddle `sequence_unpad` -> ONNX `Slice`

### 4. Mathematical Element-wise Operators (50+ items)

- [x] Map `elementwise_add` -> `Add`
- [x] Map `elementwise_sub` -> `Sub`
- [x] Map `elementwise_mul` -> `Mul`
- [x] Map `elementwise_div` -> `Div`
- [x] Map `elementwise_mod` -> `Mod`
- [x] Map `elementwise_floordiv` -> `Div` + `Floor`
- [x] Map `elementwise_pow` -> `Pow`
- [x] Map `elementwise_max` -> `Max`
- [x] Map `elementwise_min` -> `Min`
- [x] Map `bmm` -> `MatMul`
- [x] Map `matmul` -> `MatMul`
- [x] Map `matmul_v2` -> `MatMul`
- [x] Map `mul` -> `MatMul` (Flattening 2D mapping)
- [x] Map `dot` -> `Mul` + `ReduceSum`
- [x] Map `cross` -> ONNX Custom Subgraph
- [x] Map `exp` -> `Exp`
- [x] Map `log` -> `Log`
- [x] Map `log1p` -> `Log` + `Add(1)`
- [x] Map `log2` -> `Log` / `Log(2)`
- [x] Map `log10` -> `Log` / `Log(10)`
- [x] Map `sqrt` -> `Sqrt`
- [x] Map `rsqrt` -> `Sqrt` + `Reciprocal`
- [x] Map `square` -> `Pow(2)`
- [x] Map `sin` -> `Sin`
- [x] Map `cos` -> `Cos`
- [x] Map `tan` -> `Tan`
- [x] Map `asin` -> `Asin`
- [x] Map `acos` -> `Acos`
- [x] Map `atan` -> `Atan`
- [x] Map `sinh` -> `Sinh`
- [x] Map `cosh` -> `Cosh`
- [x] Map `tanh` -> `Tanh`
- [x] Map `asinh` -> `Asinh`
- [x] Map `acosh` -> `Acosh`
- [x] Map `atanh` -> `Atanh`
- [x] Map `abs` -> `Abs`
- [x] Map `ceil` -> `Ceil`
- [x] Map `floor` -> `Floor`
- [x] Map `round` -> `Round`
- [x] Map `reciprocal` -> `Reciprocal`
- [x] Map `erf` -> `Erf`
- [x] Map `sign` -> `Sign`
- [x] Map `scale` -> `Mul` + `Add`
- [x] Map `clip` -> `Clip`

### 5. Reduction & Logical Operators (30+ items)

- [x] Map `reduce_sum` -> `ReduceSum`
- [x] Map `reduce_mean` -> `ReduceMean`
- [x] Map `reduce_max` -> `ReduceMax`
- [x] Map `reduce_min` -> `ReduceMin`
- [x] Map `reduce_prod` -> `ReduceProd`
- [x] Map `reduce_all` -> `ReduceMin` (boolean)
- [x] Map `reduce_any` -> `ReduceMax` (boolean)
- [x] Map `cumsum` -> `CumSum`
- [x] Map `cumprod` -> ONNX Custom Subgraph
- [x] Map `arg_max` -> `ArgMax`
- [x] Map `arg_min` -> `ArgMin`
- [x] Map `top_k` -> `TopK`
- [x] Map `top_k_v2` -> `TopK`
- [x] Map `argsort` -> `TopK` (with sort)
- [x] Map `unique` -> `Unique`
- [x] Map `logical_and` -> `And`
- [x] Map `logical_or` -> `Or`
- [x] Map `logical_not` -> `Not`
- [x] Map `logical_xor` -> `Xor`
- [x] Map `equal` -> `Equal`
- [x] Map `not_equal` -> `Equal` + `Not`
- [x] Map `less_than` -> `Less`
- [x] Map `less_equal` -> `LessOrEqual`
- [x] Map `greater_than` -> `Greater`
- [x] Map `greater_equal` -> `GreaterOrEqual`
- [x] Map `where` -> `Where`
- [x] Map `where_index` -> `NonZero` + `Transpose`
- [x] Map `nonzero` -> `NonZero`

### 6. Activations (20+ items)

- [x] Map `relu` -> `Relu`
- [x] Map `relu6` -> `Clip(0, 6)`
- [x] Map `leaky_relu` -> `LeakyRelu`
- [x] Map `gelu` -> `Erf` subgraph
- [x] Map `swish` -> `Mul` + `Sigmoid`
- [x] Map `hard_swish` -> `HardSwish`
- [x] Map `hard_shrink` -> `HardShrink`
- [x] Map `soft_shrink` -> `SoftShrink`
- [x] Map `sigmoid` -> `Sigmoid`
- [x] Map `hard_sigmoid` -> `HardSigmoid`
- [x] Map `logsigmoid` -> Subgraph `Log(Sigmoid)`
- [x] Map `elu` -> `Elu`
- [x] Map `selu` -> `Selu`
- [x] Map `softplus` -> `Softplus`
- [x] Map `softsign` -> `Softsign`
- [x] Map `mish` -> `Mish`
- [x] Map `prelu` -> `PRelu`
- [x] Map `softmax` -> `Softmax`
- [x] Map `log_softmax` -> `LogSoftmax`
- [x] Map `bipolar_sigmoid` -> Subgraph

### 7. Neural Network & Vision Operators (50+ items)

- [x] Map `conv2d` -> `Conv`
- [x] Map `depthwise_conv2d` -> `Conv` (with groups)
- [x] Map `conv2d_transpose` -> `ConvTranspose`
- [x] Map `conv3d` -> `Conv`
- [x] Map `conv3d_transpose` -> `ConvTranspose`
- [x] Map `pool2d` (max) -> `MaxPool`
- [x] Map `pool2d` (avg) -> `AveragePool`
- [x] Map `pool3d` -> `MaxPool` / `AveragePool`
- [x] Map `max_pool2d_with_index` -> `MaxPool` (2 outputs)
- [x] Map `adaptive_pool2d` (avg) -> `GlobalAveragePool` (if 1x1) or Custom
- [x] Map `batch_norm` -> `BatchNormalization`
- [x] Map `sync_batch_norm` -> `BatchNormalization`
- [x] Map `layer_norm` -> `LayerNormalization`
- [x] Map `instance_norm` -> `InstanceNormalization`
- [x] Map `group_norm` -> ONNX Reshape+InstanceNorm Subgraph
- [x] Map `pad` -> `Pad`
- [x] Map `pad2d` -> `Pad`
- [x] Map `pad3d` -> `Pad`
- [x] Map `bilinear_interp` -> `Resize`
- [x] Map `bilinear_interp_v2` -> `Resize`
- [x] Map `nearest_interp` -> `Resize`
- [x] Map `nearest_interp_v2` -> `Resize`
- [x] Map `bicubic_interp` -> `Resize`
- [x] Map `bicubic_interp_v2` -> `Resize`
- [x] Map `trilinear_interp` -> `Resize`
- [x] Map `roi_align` -> `RoiAlign`
- [x] Map `roi_pool` -> `MaxRoiPool`
- [x] Map `yolo_box` -> ONNX YOLO Custom Subgraph
- [x] Map `prior_box` -> ONNX Custom Subgraph
- [x] Map `multiclass_nms` -> `NonMaxSuppression` + loop subgraph
- [x] Map `multiclass_nms3` -> `NonMaxSuppression` + loop subgraph
- [x] Map `matrix_nms` -> ONNX Custom Subgraph
- [x] Map `generate_proposals` -> ONNX Custom Subgraph
- [x] Map `generate_proposals_v2` -> ONNX Custom Subgraph
- [x] Map `distribute_fpn_proposals` -> ONNX Custom Subgraph
- [x] Map `box_coder` -> ONNX Custom Subgraph
- [x] Map `bipartite_match` -> ONNX Custom Subgraph
- [x] Map `affine_channel` -> ONNX Custom Subgraph
- [x] Map `anchor_generator` -> ONNX Custom Subgraph
- [x] Map `collect_fpn_proposals` -> ONNX Custom Subgraph

### 8. Tensor Manipulation & Creation Operators (50+ items)

- [x] Map `reshape` -> `Reshape`
- [x] Map `reshape2` -> `Reshape`
- [x] Map `flatten` -> `Flatten`
- [x] Map `flatten_contiguous_range` -> `Flatten`
- [x] Map `squeeze` -> `Squeeze`
- [x] Map `squeeze2` -> `Squeeze`
- [x] Map `unsqueeze` -> `Unsqueeze`
- [x] Map `unsqueeze2` -> `Unsqueeze`
- [x] Map `transpose` -> `Transpose`
- [x] Map `transpose2` -> `Transpose`
- [x] Map `concat` -> `Concat`
- [x] Map `stack` -> `Unsqueeze` + `Concat`
- [x] Map `unstack` -> `Split` + `Squeeze`
- [x] Map `split` -> `Split`
- [x] Map `slice` -> `Slice`
- [x] Map `strided_slice` -> `Slice`
- [x] Map `gather` -> `Gather`
- [x] Map `gather_nd` -> `GatherND`
- [x] Map `scatter` -> `ScatterElements`
- [x] Map `scatter_nd` -> `ScatterND`
- [x] Map `scatter_nd_add` -> `ScatterND` + `Add`
- [x] Map `index_select` -> `Gather`
- [x] Map `tile` -> `Tile`
- [x] Map `expand` -> `Expand`
- [x] Map `expand_v2` -> `Expand`
- [x] Map `expand_as_v2` -> `Expand`
- [x] Map `cast` -> `Cast`
- [x] Map `shape` -> `Shape`
- [x] Map `size` -> `Size`
- [x] Map `fill_constant` -> `ConstantOfShape`
- [x] Map `fill_constant_batch_size_like` -> `Shape` + `ConstantOfShape`
- [x] Map `fill_any_like` -> `ConstantOfShape`
- [x] Map `zeros_like` -> `ConstantOfShape` (0)
- [x] Map `ones_like` -> `ConstantOfShape` (1)
- [x] Map `arange` -> `Range`
- [x] Map `linspace` -> ONNX Subgraph
- [x] Map `eye` -> ONNX Subgraph (or ConstantOfShape + EyeLike)
- [x] Map `uniform_random` -> `RandomUniform`
- [x] Map `gaussian_random` -> `RandomNormal`
- [x] Map `randint` -> `RandomUniform` (int)
- [x] Map `randperm` -> ONNX Custom
- [x] Map `dropout` -> `Dropout` or `Identity` (inference)
- [x] Map `roll` -> ONNX Subgraph
- [x] Map `flip` -> `ReverseSequence`
- [x] Map `unbind` -> `Split`
- [x] Map `meshgrid` -> ONNX Subgraph

### 9. NLP & Sequence Operators (20+ items)

- [x] Map `embedding` -> `Gather`
- [x] Map `lookup_table` -> `Gather`
- [x] Map `lookup_table_v2` -> `Gather`
- [x] Map `rnn` -> `RNN`
- [x] Map `gru` -> `GRU`
- [x] Map `lstm` -> `LSTM`
- [x] Map `gru_unit` -> `GRU` step subgraph
- [x] Map `lstm_unit` -> `LSTM` step subgraph
- [x] Map `beam_search` -> ONNX BeamSearch subgraph/operator
- [x] Map `beam_search_decode` -> ONNX Subgraph
- [x] Map `layer_norm` (NLP variant) -> `LayerNormalization`
- [x] Map `crf_decoding` -> ONNX Custom
- [x] Map `viterbi_decode` -> ONNX Custom
- [x] Map `fused_attention` -> ONNX Custom or Math ops
- [x] Map `fused_feedforward` -> ONNX Custom or Math ops

### 10. Quantization & Mixed Precision (20+ items)

- [x] Map `quantize_linear` -> `QuantizeLinear`
- [x] Map `dequantize_linear` -> `DequantizeLinear`
- [x] Map `fake_quantize_abs_max` -> Quantize Subgraph
- [x] Map `fake_quantize_range_abs_max` -> Quantize Subgraph
- [x] Map `fake_quantize_moving_average_abs_max` -> Quantize Subgraph
- [x] Map `fake_channel_wise_quantize_abs_max` -> Quantize Subgraph
- [x] Map `fake_dequantize_max_abs` -> Dequantize Subgraph
- [x] Map `fake_channel_wise_dequantize_max_abs` -> Dequantize Subgraph
- [x] Handle INT8 Conv weight extraction from quantized models
- [x] Handle INT8 MatMul weight extraction from quantized models
- [x] Map Paddle dynamic quantization parameters to ONNX dynamic quantize
- [x] Strip out FakeQuantize ops for FP32 targets natively

### 11. Graph Optimizations (Paddle Specific)

- [x] Constant Folding pass
- [x] Redundant `cast` Elimination
- [x] `scale` operation folding into Conv/MatMul weights
- [x] `fill_constant` resolution at translation time
- [x] Fuse `conv2d` + `elementwise_add` + `relu` -> `Conv(activation='Relu')`
- [x] Fuse `matmul` + `elementwise_add` -> `Gemm`
- [x] Eliminate `dropout` identity pass-through
- [x] Subgraph simplification for Paddle's complex `multiclass_nms` structures
- [x] Fuse `batch_norm` into `conv2d`

### 12. Opset Compliance

- [x] Target ONNX Opset 7
- [x] Target ONNX Opset 8
- [x] Target ONNX Opset 9
- [x] Target ONNX Opset 10
- [x] Target ONNX Opset 11
- [x] Target ONNX Opset 12
- [x] Target ONNX Opset 13
- [x] Target ONNX Opset 14
- [x] Target ONNX Opset 15
- [x] Target ONNX Opset 16
- [x] Target ONNX Opset 17
- [x] Target ONNX Opset 18
- [x] Target ONNX Opset 19
- [x] Target ONNX Opset 20
- [x] Target ONNX Opset 21

### 13. Zero-Dependency & Lightweight Runtime Features (20+ items)

- [x] CLI fully operational without `paddlepaddle` installed
- [x] Convert `.pdmodel` files purely via python `protobuf`
- [x] Memory-efficient streamed reading of `.pdiparams` binary blobs
- [x] Deployable natively to Browser / WASM environments
- [x] WebWorker parallel execution compatibility
- [x] Run conversion securely in AWS Lambda / Cloudflare Workers
- [x] Distributed / Ray cluster compatible for mass-conversion jobs
- [x] Generates strongly-typed `ai.onnx` graphs verified internally
- [x] Expose native JS/TypeScript bindings for drag-and-drop web UI
- [x] Dynamic WebGPU Shader preparation metrics generated during conversion

### 14. Additional Paddle Operators & Special Cases (30+ items)

- [x] Map `grid_sampler` -> `GridSample`
- [x] Map `affine_grid` -> ONNX Custom Subgraph
- [x] Map `deformable_conv` -> ONNX Custom Subgraph
- [x] Map `deformable_conv_v1` -> ONNX Custom Subgraph
- [x] Map `gru_unit` (dynamic) -> ONNX Loop + GRU
- [x] Map `lstm_unit` (dynamic) -> ONNX Loop + LSTM
- [x] Map `layer_norm` (fused bias/scale) -> `LayerNormalization`
- [x] Map `multihead_attention` -> ONNX Custom Subgraph
- [x] Map `mish` (custom variant) -> `Mish`
- [x] Map `swish` (custom variant) -> `Mul` + `Sigmoid`
- [x] Map `bce_loss` -> ONNX Custom
- [x] Map `cross_entropy` -> ONNX Custom
- [x] Map `huber_loss` -> ONNX Custom
- [x] Map `l1_loss` -> ONNX Custom
- [x] Map `mse_loss` -> ONNX Custom
- [x] Map `nll_loss` -> `NegativeLogLikelihoodLoss`
- [x] Map `smooth_l1_loss` -> ONNX Custom
- [x] Map `sigmoid_cross_entropy_with_logits` -> ONNX Custom Subgraph
- [x] Map `softmax_with_cross_entropy` -> `SoftmaxCrossEntropyLoss`
- [x] Map `hard_shrink` -> `HardShrink`
- [x] Map `soft_shrink` -> `SoftShrink`
- [x] Map `bincount` -> `Bincount`
- [x] Map `histogram` -> `Histogram` (experimental)
- [x] Map `kthvalue` -> `TopK`
- [x] Map `mode` -> `TopK` (K=1)
- [x] Map `sort` -> `TopK` (all elements)
- [x] Map `diag` -> `EyeLike` + `Mul`
- [x] Map `diag_embed` -> `EyeLike` + `Mul`
- [x] Map `trace` -> `EyeLike` + `Mul` + `ReduceSum`
- [x] Map `triu` -> `Trilu`
- [x] Map `tril` -> `Trilu`
