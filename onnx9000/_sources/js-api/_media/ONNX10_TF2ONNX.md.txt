# tf2onnx Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `tf2onnx` within the `onnx9000` ecosystem.
Unlike the original project which relies heavily on a massive native `tensorflow` installation to parse and trace graphs, this implementation uses a pure-Python, zero-dependency `protobuf` / `flatbuffers` parser.
This allows the converter to run entirely in the browser via WASM/WebGPU or incredibly efficiently in serverless/distributed environments without pulling in gigabytes of TF dependencies.

## Exhaustive Parity Checklist

### 1. Pure-Python Parsers & Loaders (Zero-Dependency)

- [x] Implement pure-Python parser for TensorFlow `GraphDef` (`.pb`) formats
- [x] Implement pure-Python parser for TensorFlow `SavedModel` v1 formats
- [x] Implement pure-Python parser for TensorFlow `SavedModel` v2 formats
- [x] Implement pure-Python loader for TensorFlow Checkpoints (`.ckpt`)
- [x] Implement pure-Python `flatbuffers` parser for TFLite (`.tflite`) formats
- [x] Implement Keras `.h5` model parser (via `h5py` compatible pure-Python reader)
- [x] Implement Keras v3 `.keras` archive parser
- [x] Support parsing multiple `MetaGraphDef` tags (e.g., `serve`, `train`)
- [x] Extract signature definitions (inputs/outputs) from `SavedModel` automatically
- [x] Support overriding default input shapes during parsing
- [x] Support overriding default output nodes during parsing
- [x] Extract variables and constants directly from Checkpoint indices
- [x] Handle TF1 style `Session.run` graph extraction
- [x] Handle TF2 `tf.function` concrete function extraction
- [x] Support loading graphs with unresolved external variables (providing mock shapes)
- [x] Implement `Asset` parsing for SavedModels
- [x] Extract vocabulary tables for Text models
- [x] Load and decode `LookupTableFind` and `HashTableV2`
- [x] Handle TensorFlow `Resource` handles natively

### 2. Graph Pre-Processing & Topology Adjustments

- [x] Implement dead code elimination for unconnected TF nodes
- [x] Remove training-specific ops (e.g., `ApplyAdam`, `AssignVariableOp`)
- [x] Remove assertion ops (`Assert`, `CheckNumerics`)
- [x] Resolve and constant-fold `Shape` ops based on input signatures
- [x] Strip `StopGradient` and `PreventGradient` nodes
- [x] Implement `NHWC` to `NCHW` layout conversion pass for CNNs
- [x] Track spatial dimensions across Reshape/Transpose to maintain layout consistency
- [x] Resolve dynamic shapes (TensorFlow `None` -> ONNX `-1` or dynamic symbolic dims)
- [x] Implement Subgraph extraction (pruning graph to specified inputs/outputs)
- [x] Rename invalid TF node names to ONNX-compliant identifiers
- [x] Map TensorFlow data types (`DT_FLOAT`, `DT_INT32`, etc.) to ONNX `TensorProto.DataType`
- [x] Support `DT_BFLOAT16` to ONNX `BFLOAT16` conversion
- [x] Support `DT_HALF` to ONNX `FLOAT16` conversion
- [x] Support `DT_STRING` to ONNX `STRING` conversion
- [x] Expand macros/fused ops in `GraphDef` (e.g., `LSTMBlockCell`)
- [x] Implement `TensorArray` to ONNX `Sequence` transformation
- [x] Rewrite variable reads (`ReadVariableOp`) to direct inputs
- [x] Rewrite resource initializations
- [x] Prune identity chains created during Graph building

### 3. TensorFlow Control Flow v1 & v2 Mapping

- [x] Map TF1 `Switch` -> ONNX `If` branch condition
- [x] Map TF1 `Merge` -> ONNX `If` branch output
- [x] Map TF1 `Enter` -> ONNX `Loop` initial context
- [x] Map TF1 `Exit` -> ONNX `Loop` final output
- [x] Map TF1 `NextIteration` -> ONNX `Loop` state update
- [x] Map TF1 `LoopCond` -> ONNX `Loop` condition
- [x] Map TF2 `StatelessIf` -> ONNX `If`
- [x] Map TF2 `StatelessWhile` -> ONNX `Loop`
- [x] Map TF2 `While` -> ONNX `Loop`
- [x] Map TF2 `Cond` -> ONNX `If`
- [x] Map TF2 `Case` -> Chained ONNX `If`
- [x] Map `TensorListReserve` -> ONNX `SequenceEmpty`
- [x] Map `TensorListPushBack` -> ONNX `SequenceInsert`
- [x] Map `TensorListPopBack` -> ONNX `SequenceErase`
- [x] Map `TensorListStack` -> ONNX `SequenceConstruct` + `Concat`
- [x] Map `TensorListFromTensor` -> ONNX `SplitToSequence`
- [x] Map `TensorListSetItem` -> ONNX `SequenceUpdate` (via `Loop`)
- [x] Map `TensorListGetItem` -> ONNX `SequenceAt`
- [x] Map `TensorListLength` -> ONNX `SequenceLength`
- [x] Map `TensorArrayV3` -> ONNX `Sequence`
- [x] Map `TensorArrayReadV3` -> ONNX `SequenceAt`
- [x] Map `TensorArrayWriteV3` -> ONNX `SequenceUpdate` (via `Loop`)
- [x] Map `TensorArrayGatherV3` -> ONNX `SequenceConstruct` + `Concat`
- [x] Map `TensorArrayScatterV3` -> ONNX `SplitToSequence`
- [x] Map `TensorArraySizeV3` -> ONNX `SequenceLength`
- [x] Map `TensorArrayCloseV3` -> No-op equivalent

### 4. Mathematical Operator Mapping (70+ items)

- [x] Map `Add` -> `Add`
- [x] Map `AddV2` -> `Add`
- [x] Map `Sub` -> `Sub`
- [x] Map `Mul` -> `Mul`
- [x] Map `Div` -> `Div`
- [x] Map `RealDiv` -> `Div`
- [x] Map `FloorDiv` -> `Div` + `Floor`
- [x] Map `TruncateDiv` -> `Div` + `Cast`
- [x] Map `Mod` -> `Mod`
- [x] Map `FloorMod` -> `Mod`
- [x] Map `Abs` -> `Abs`
- [x] Map `Neg` -> `Neg`
- [x] Map `Sign` -> `Sign`
- [x] Map `Ceil` -> `Ceil`
- [x] Map `Floor` -> `Floor`
- [x] Map `Round` -> `Round`
- [x] Map `Exp` -> `Exp`
- [x] Map `Expm1` -> `Exp` + `Sub(1)`
- [x] Map `Log` -> `Log`
- [x] Map `Log1p` -> `Add(1)` + `Log`
- [x] Map `Sin` -> `Sin`
- [x] Map `Cos` -> `Cos`
- [x] Map `Tan` -> `Tan`
- [x] Map `Asin` -> `Asin`
- [x] Map `Acos` -> `Acos`
- [x] Map `Atan` -> `Atan`
- [x] Map `Atan2` -> ONNX custom subgraph
- [x] Map `Sinh` -> `Sinh`
- [x] Map `Cosh` -> `Cosh`
- [x] Map `Tanh` -> `Tanh`
- [x] Map `Asinh` -> `Asinh`
- [x] Map `Acosh` -> `Acosh`
- [x] Map `Atanh` -> `Atanh`
- [x] Map `Sqrt` -> `Sqrt`
- [x] Map `Rsqrt` -> `Sqrt` + `Reciprocal`
- [x] Map `Square` -> `Pow(2)`
- [x] Map `Pow` -> `Pow`
- [x] Map `Maximum` -> `Max`
- [x] Map `Minimum` -> `Min`
- [x] Map `ComplexAbs` -> `Abs`
- [x] Map `AddN` -> Chained `Add` or `Sum`
- [x] Map `Angle` -> Subgraph (Atan2(img, real))
- [x] Map `Conj` -> Cast/Neg Subgraph
- [x] Map `Real` -> `Slice` or Cast
- [x] Map `Imag` -> `Slice` or Cast
- [x] Map `Xdivy` -> `Where(x==0, 0, x/y)`
- [x] Map `Xlogy` -> `Where(x==0, 0, x*log(y))`
- [x] Map `Zeta` -> ONNX Custom
- [x] Map `Polygamma` -> ONNX Custom
- [x] Map `BesselI0e` -> ONNX Custom
- [x] Map `BesselI1e` -> ONNX Custom
- [x] Map `LogMatrixDeterminant` -> ONNX Custom
- [x] Map `MatrixInverse` -> ONNX Custom
- [x] Map `Cholesky` -> ONNX Custom
- [x] Map `Svd` -> ONNX Custom
- [x] Map `Qr` -> ONNX Custom
- [x] Map `Einsum` -> `Einsum`
- [x] Map `BatchMatMul` -> `MatMul`
- [x] Map `BatchMatMulV2` -> `MatMul`
- [x] Map `BiasAdd` -> `Add`

### 5. Logical & Reduction Operator Mapping (30+ items)

- [x] Map `LogicalAnd` -> `And`
- [x] Map `LogicalOr` -> `Or`
- [x] Map `LogicalNot` -> `Not`
- [x] Map `Equal` -> `Equal`
- [x] Map `NotEqual` -> `Equal` + `Not`
- [x] Map `Greater` -> `Greater`
- [x] Map `GreaterEqual` -> `GreaterOrEqual`
- [x] Map `Less` -> `Less`
- [x] Map `LessEqual` -> `LessOrEqual`
- [x] Map `Select` -> `Where`
- [x] Map `SelectV2` -> `Where`
- [x] Map `Where3` -> `Where`
- [x] Map `Where` (1 input) -> `NonZero` + `Transpose`
- [x] Map `ReduceSum` -> `ReduceSum`
- [x] Map `ReduceMean` -> `ReduceMean`
- [x] Map `ReduceMax` -> `ReduceMax`
- [x] Map `ReduceMin` -> `ReduceMin`
- [x] Map `ReduceProd` -> `ReduceProd`
- [x] Map `ReduceAll` -> `ReduceMin` (boolean)
- [x] Map `ReduceAny` -> `ReduceMax` (boolean)
- [x] Map `EuclideanNorm` -> `ReduceL2`
- [x] Map `ArgMax` -> `ArgMax`
- [x] Map `ArgMin` -> `ArgMin`
- [x] Map `TopK` -> `TopK`
- [x] Map `TopKV2` -> `TopK`
- [x] Map `Unique` -> `Unique`
- [x] Map `UniqueWithCounts` -> `Unique` (with custom count logic)
- [x] Map `InvertPermutation` -> ONNX Subgraph

### 6. Neural Network Operator Mapping (40+ items)

- [x] Map `MatMul` -> `MatMul` (handling adj_a / adj_b)
- [x] Map `Conv2D` -> `Conv` (NHWC to NCHW mapping)
- [x] Map `DepthwiseConv2dNative` -> `Conv` (with `group` attribute)
- [x] Map `Conv3D` -> `Conv` (NDHWC to NCDHW)
- [x] Map `Conv2DBackpropInput` -> `ConvTranspose`
- [x] Map `Conv2DBackpropFilter` -> ONNX Subgraph or Unfold/MatMul
- [x] Map `Conv3DBackpropInputV2` -> `ConvTranspose`
- [x] Map `MaxPool` -> `MaxPool`
- [x] Map `MaxPoolV2` -> `MaxPool`
- [x] Map `MaxPoolWithArgmax` -> `MaxPool` (2 outputs)
- [x] Map `MaxPool3D` -> `MaxPool`
- [x] Map `AvgPool` -> `AveragePool`
- [x] Map `AvgPool3D` -> `AveragePool`
- [x] Map `FusedBatchNorm` -> `BatchNormalization`
- [x] Map `FusedBatchNormV2` -> `BatchNormalization`
- [x] Map `FusedBatchNormV3` -> `BatchNormalization`
- [x] Map `Relu` -> `Relu`
- [x] Map `Relu6` -> `Clip` (0, 6)
- [x] Map `LeakyRelu` -> `LeakyRelu`
- [x] Map `Elu` -> `Elu`
- [x] Map `Selu` -> `Selu`
- [x] Map `Softmax` -> `Softmax`
- [x] Map `LogSoftmax` -> `LogSoftmax`
- [x] Map `Softplus` -> `Softplus`
- [x] Map `Softsign` -> `Softsign`
- [x] Map `Gelu` -> `Erf` subgraph
- [x] Map `Sigmoid` -> `Sigmoid`
- [x] Map `HardSigmoid` -> `HardSigmoid`
- [x] Map `Swish` -> `HardSwish`
- [x] Map `LRN` -> `LRN`
- [x] Map `Pad` -> `Pad` (handling constant)
- [x] Map `PadV2` -> `Pad`
- [x] Map `MirrorPad` -> `Pad` (handling reflect, symmetric)
- [x] Map `Dilation2D` -> ONNX Custom or MaxPool with dilation
- [x] Map `SpaceToDepth` -> `SpaceToDepth`
- [x] Map `DepthToSpace` -> `DepthToSpace`
- [x] Map `SpaceToBatchND` -> Reshape/Transpose Subgraph
- [x] Map `BatchToSpaceND` -> Reshape/Transpose Subgraph
- [x] Map `RNN` -> `RNN`
- [x] Map `GRUBlockCell` -> `GRU`
- [x] Map `LSTMBlockCell` -> `LSTM`

### 7. Array & Tensor Manipulation Mapping (40+ items)

- [x] Map `Reshape` -> `Reshape`
- [x] Map `Transpose` -> `Transpose`
- [x] Map `Concat` -> `Concat`
- [x] Map `ConcatV2` -> `Concat`
- [x] Map `Pack` -> `Unsqueeze` + `Concat`
- [x] Map `Unpack` -> `Split` + `Squeeze`
- [x] Map `Split` -> `Split`
- [x] Map `SplitV` -> `Split`
- [x] Map `Slice` -> `Slice`
- [x] Map `StridedSlice` -> `Slice` (handling begin/end/strides/bitmasks)
- [x] Map `Gather` -> `Gather`
- [x] Map `GatherV2` -> `Gather`
- [x] Map `GatherNd` -> `GatherND`
- [x] Map `ScatterNd` -> `ScatterND`
- [x] Map `TensorScatterUpdate` -> `ScatterND`
- [x] Map `TensorScatterAdd` -> `ScatterND` + `Add`
- [x] Map `ExpandDims` -> `Unsqueeze`
- [x] Map `Squeeze` -> `Squeeze`
- [x] Map `Cast` -> `Cast`
- [x] Map `Shape` -> `Shape`
- [x] Map `ShapeN` -> Multiple `Shape` ops
- [x] Map `Size` -> `Size`
- [x] Map `Rank` -> `Shape` + `Size`
- [x] Map `Tile` -> `Tile`
- [x] Map `Reverse` -> `ReverseSequence`
- [x] Map `ReverseV2` -> `ReverseSequence`
- [x] Map `Fill` -> `ConstantOfShape`
- [x] Map `ZerosLike` -> `ConstantOfShape`
- [x] Map `OnesLike` -> `ConstantOfShape`
- [x] Map `BroadcastTo` -> `Expand`
- [x] Map `MatrixDiag` -> Diagonal Mask Subgraph
- [x] Map `MatrixDiagV2` -> Diagonal Mask Subgraph
- [x] Map `MatrixDiagV3` -> Diagonal Mask Subgraph
- [x] Map `MatrixSetDiag` -> Subgraph
- [x] Map `MatrixSetDiagV2` -> Subgraph
- [x] Map `MatrixSetDiagV3` -> Subgraph
- [x] Map `MatrixBandPart` -> `Trilu`
- [x] Map `OneHot` -> `OneHot`
- [x] Map `Cumsum` -> `CumSum`
- [x] Map `Cumprod` -> ONNX Custom Subgraph
- [x] Map `Bitcast` -> `BitShift` or custom

### 8. Image, Resize & Audio Mapping (20+ items)

- [x] Map `ResizeBilinear` -> `Resize` (mode: linear)
- [x] Map `ResizeNearestNeighbor` -> `Resize` (mode: nearest)
- [x] Map `ResizeBicubic` -> `Resize` (mode: cubic)
- [x] Map `CropAndResize` -> `RoiAlign`
- [x] Map `NonMaxSuppressionV2` -> `NonMaxSuppression`
- [x] Map `NonMaxSuppressionV3` -> `NonMaxSuppression`
- [x] Map `NonMaxSuppressionV4` -> `NonMaxSuppression`
- [x] Map `NonMaxSuppressionV5` -> `NonMaxSuppression`
- [x] Map `ExtractImagePatches` -> im2col subgraph
- [x] Map `RGBToHSV` -> ONNX Subgraph
- [x] Map `HSVToRGB` -> ONNX Subgraph
- [x] Map `AdjustContrastv2` -> ONNX Subgraph
- [x] Map `AdjustSaturation` -> ONNX Subgraph
- [x] Map `AdjustHue` -> ONNX Subgraph
- [x] Map `AudioSpectrogram` -> ONNX STFT custom or Subgraph
- [x] Map `Mfcc` -> MelWeight Subgraph
- [x] Map `DecodeJpeg` -> ONNX DecodeImage
- [x] Map `DecodePng` -> ONNX DecodeImage
- [x] Map `EncodeJpeg` -> ONNX EncodeImage
- [x] Map `EncodePng` -> ONNX EncodeImage

### 9. String, Text & Random Operators (20+ items)

- [x] Map `StringJoin` -> `StringConcat`
- [x] Map `StringLower` -> `StringNormalizer`
- [x] Map `StringUpper` -> `StringNormalizer`
- [x] Map `RegexReplace` -> Regex Extension Node
- [x] Map `StringSplit` -> ONNX Custom
- [x] Map `StringStrip` -> ONNX Custom
- [x] Map `Substr` -> ONNX Custom
- [x] Map `RandomUniform` -> `RandomUniform`
- [x] Map `RandomUniformInt` -> `RandomUniform`
- [x] Map `RandomStandardNormal` -> `RandomNormal`
- [x] Map `TruncatedNormal` -> `RandomNormal` + `Clip`
- [x] Map `Multinomial` -> `Multinomial`
- [x] Map `RandomGamma` -> ONNX Custom
- [x] Map `RandomPoisson` -> ONNX Custom
- [x] Map `StatelessRandomUniform` -> `RandomUniform`
- [x] Map `StatelessRandomNormal` -> `RandomNormal`

### 10. TFLite-Specific Operator Mapping (30+ items)

- [x] Map TFLite `CONV_2D` -> `Conv`
- [x] Map TFLite `DEPTHWISE_CONV_2D` -> `Conv`
- [x] Map TFLite `FULLY_CONNECTED` -> `MatMul` + `Add`
- [x] Map TFLite `MEAN` -> `ReduceMean`
- [x] Map TFLite `AVERAGE_POOL_2D` -> `AveragePool`
- [x] Map TFLite `MAX_POOL_2D` -> `MaxPool`
- [x] Map TFLite `L2_NORMALIZATION` -> `LpNormalization`
- [x] Map TFLite `CONCATENATION` -> `Concat`
- [x] Map TFLite `RESHAPE` -> `Reshape`
- [x] Map TFLite `LOGISTIC` -> `Sigmoid`
- [x] Map TFLite `TANH` -> `Tanh`
- [x] Map TFLite `ADD` -> `Add`
- [x] Map TFLite `MUL` -> `Mul`
- [x] Map TFLite `SUB` -> `Sub`
- [x] Map TFLite `DIV` -> `Div`
- [x] Map TFLite `RSQRT` -> `Sqrt` + `Reciprocal`
- [x] Map TFLite `SQUARED_DIFFERENCE` -> `Sub` + `Pow(2)`
- [x] Map TFLite `STRIDED_SLICE` -> `Slice`
- [x] Map TFLite `PACK` -> `Unsqueeze` + `Concat`
- [x] Map TFLite `UNPACK` -> `Split` + `Squeeze`
- [x] Map TFLite `SPLIT` -> `Split`
- [x] Map TFLite `SPLIT_V` -> `Split`
- [x] Map TFLite `TRANSPOSE_CONV` -> `ConvTranspose`
- [x] Map TFLite `QUANTIZE` -> `QuantizeLinear`
- [x] Map TFLite `DEQUANTIZE` -> `DequantizeLinear`
- [x] Map TFLite `CAST` -> `Cast`
- [x] Map TFLite `EXPAND_DIMS` -> `Unsqueeze`
- [x] Map TFLite `SQUEEZE` -> `Squeeze`
- [x] Map TFLite `FILL` -> `ConstantOfShape`

### 11. Graph Optimizations (tf2onnx specifics)

- [x] Implement Constant Folding pass
- [x] Implement Redundant Transpose Elimination (canceling `Transpose` -> `Transpose`)
- [x] Implement Reshape Fusion (merging adjacent `Reshape` ops)
- [x] Implement Identity Elimination (removing `Identity`, `IdentityN`)
- [x] Implement BatchNorm Folding (fusing `BatchNormalization` into `Conv` weights)
- [x] Implement Cast Elimination (removing unnecessary type casts)
- [x] Implement Squeeze/Unsqueeze Fusion
- [x] Optimizing scalar additions and multiplications (fusing into biases)
- [x] Implement TFLite dequantize-quantize fusion
- [x] Optimize sequential `Slice` ops into a single `Slice`
- [x] Eliminate `Pad` + `Slice` exact inverse sequences
- [x] Optimize `Split` + `Concat` inverse sequences
- [x] Fuse `Conv` + `Add` + `Relu` where possible
- [x] Drop `Shape` ops tied to fixed known inputs

### 12. Opset Compliance

- [x] Support generating ONNX Opset 7
- [x] Support generating ONNX Opset 8
- [x] Support generating ONNX Opset 9
- [x] Support generating ONNX Opset 10
- [x] Support generating ONNX Opset 11
- [x] Support generating ONNX Opset 12
- [x] Support generating ONNX Opset 13
- [x] Support generating ONNX Opset 14
- [x] Support generating ONNX Opset 15
- [x] Support generating ONNX Opset 16
- [x] Support generating ONNX Opset 17
- [x] Support generating ONNX Opset 18
- [x] Support generating ONNX Opset 19
- [x] Support generating ONNX Opset 20
- [x] Support generating ONNX Opset 21
- [x] Domain standard compliance tests for `ai.onnx`
- [x] Domain standard compliance tests for `ai.onnx.ml`

### 13. Zero-Dependency & Lightweight Runtime Features

- [x] CLI fully operational without `tensorflow` or `tflite` PIP packages
- [x] Convert models entirely in a browser using Emscripten compiled WASM
- [x] Support WebWorker parallel parsing of massive `.pb` files
- [x] Zero-copy translation of weights from TF Protobuf to ONNX TensorProto in memory
- [x] Support HTTP streaming parsing of TF GraphDefs (chunked conversion)
- [x] Serverless deployment compatibility (AWS Lambda, Cloudflare Workers)
- [x] Distributed conversion for multi-GB SavedModels via Ray/Celery
- [x] Dynamic WebGPU shader allocation verification during conversion
- [x] Expose TypeScript bindings for in-browser drag-and-drop conversion
- [x] Memory-mapped model rewriting for models > 2GB (preventing OOM)
