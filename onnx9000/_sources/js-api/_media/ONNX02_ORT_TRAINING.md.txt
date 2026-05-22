# ONNX Runtime Training Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `ONNX Runtime Training` (`orttraining`) within the `onnx9000` ecosystem.
The original `orttraining` requires a massive C++ dependency to dynamically calculate gradients (Automatic Differentiation) and update weights at runtime.
Our `onnx9000` reimplementation uses a completely different architecture: **Ahead-of-Time (AOT) Symbolic Autograd**. We parse a standard ONNX inference graph and mathematically compile the exact Vector-Jacobian Products (VJPs) into pure, standard `ai.onnx` forward math operators (`MatMul`, `Add`, `Transpose`).
This means we output a static "Training Graph" `.onnx` file that inherently contains the forward pass, loss calculation, backward pass, and optimizer steps. This file can then be executed on _any_ standard inference engine (like our browser WASM/WebGPU backend), allowing for true zero-dependency on-device training in the browser.

## Exhaustive Parity Checklist

### 1. Autograd Core Architecture & Gradient Tracing (40+ items)

- [xx] Implement AOT Autograd Compiler in pure Python
- [xx] Implement reverse-mode Automatic Differentiation algorithm
- [xx] Extract trainable `Parameter` nodes explicitly
- [xx] Exclude `Constant` / Frozen nodes from gradient tracking (requires_grad=False)
- [xx] Implement Topological Sort to establish forward execution order
- [xx] Implement Forward pass activation caching (saving intermediate tensors required for backward)
- [xx] Map Forward nodes to their exact Vector-Jacobian Product (VJP) generator functions
- [xx] Statically allocate explicit output tensors for all computed gradients (`dY/dX`)
- [xx] Accumulate gradients (`Add`) explicitly when a tensor fans out to multiple consumers
- [xx] Construct the entire backward pass strictly using standard `ai.onnx` operators
- [xx] Ensure backward pass contains no specialized `ai.onnx.training` operators (maximizing portability)
- [xx] Eliminate mathematically redundant gradient calculations (`0 * dY -> 0`) natively
- [xx] Optimize sequential gradient additions into `Sum` operators
- [xx] Validate shape dimensionality mathematically during VJP injection
- [xx] Ensure VJP broadcasting rules exactly invert Forward broadcasting rules (e.g. `ReduceSum` on broadcasted axes)
- [xx] Map non-differentiable operations to `0` gradients (e.g., `ArgMax`)
- [xx] Allow users to explicitly stop gradients (`StopGradient` emulation)
- [xx] Detect and break gradient tracking loops intelligently
- [xx] Support higher-order derivatives (Hessian-Vector Products) experimentally if needed
- [xx] Output a single monolithic ONNX graph: Inputs (`X`, `Y_true`) -> Outputs (`Loss`, `Grads`)
- [xx] Expose API to yield intermediate gradients for debugging (`RetainGrad`)
- [xx] Provide analytical shape inference across the entire generated backward graph
- [xx] Ensure `Float32` precision is strictly maintained across gradient accumulation
- [xx] Support mixed-precision scaling (loss scaling for `Float16` backprop)
- [xx] Provide explicit sub-graph partitioning (Separating Forward.onnx and Backward.onnx)
- [xx] Support generating isolated Loss computation graphs
- [xx] Support generating isolated Optimizer step graphs
- [xx] Support dynamic shapes (`-1`) during gradient generation natively
- [xx] Validate symbolic gradient outputs match PyTorch `autograd` perfectly (atol=1e-5)
- [xx] Generate standard ONNX `GradientProto` if specifically requested by user flags
- [xx] Prevent Out-Of-Memory by actively freeing cached forward activations early during the backward pass (Simulated Memory Arena optimization)
- [xx] Track tensor dependencies explicitly to optimize memory reuse statically
- [xx] Output human-readable DAG summary of the generated backward topology

### 2. Elementwise & Math VJP (Vector-Jacobian Products) (35+ items)

- [xx] Implement VJP for `Add` (dY -> dX1=dY, dX2=dY with un-broadcasting)
- [xx] Implement VJP for `Sub` (dY -> dX1=dY, dX2=-dY with un-broadcasting)
- [xx] Implement VJP for `Mul` (dY -> dX1=dY*X2, dX2=dY*X1 with un-broadcasting)
- [xx] Implement VJP for `Div` (dY -> dX1=dY/X2, dX2=-dY\*X1/(X2^2) with un-broadcasting)
- [xx] Implement VJP for `Abs` (dY -> dY \* Sign(X))
- [xx] Implement VJP for `Neg` (dY -> -dY)
- [xx] Implement VJP for `Exp` (dY -> dY \* Exp(X))
- [xx] Implement VJP for `Log` (dY -> dY / X)
- [xx] Implement VJP for `Sin` (dY -> dY \* Cos(X))
- [xx] Implement VJP for `Cos` (dY -> dY \* -Sin(X))
- [xx] Implement VJP for `Tan` (dY -> dY / Cos(X)^2)
- [xx] Implement VJP for `Sinh` (dY -> dY \* Cosh(X))
- [xx] Implement VJP for `Cosh` (dY -> dY \* Sinh(X))
- [xx] Implement VJP for `Tanh` (dY -> dY \* (1 - Tanh(X)^2))
- [xx] Implement VJP for `Pow` (Constant exponent, e.g., X^2 -> dY \* 2X)
- [xx] Implement VJP for `Pow` (Variable exponent X^Y)
- [xx] Implement VJP for `Sqrt` (dY -> dY / (2 \* Sqrt(X)))
- [xx] Implement VJP for `Reciprocal` (dY -> -dY / X^2)
- [xx] Implement VJP for `Clip` (dY -> dY \* Where(min < X < max, 1, 0))
- [xx] Implement VJP for `Relu` (dY -> dY \* Where(X > 0, 1, 0))
- [xx] Implement VJP for `LeakyRelu` (dY -> dY \* Where(X > 0, 1, alpha))
- [xx] Implement VJP for `PRelu` (dY -> dY _ Where(X > 0, 1, alpha), dAlpha -> dY _ Where(X < 0, X, 0))
- [xx] Implement VJP for `Sigmoid` (dY -> dY _ Sigmoid(X) _ (1 - Sigmoid(X)))
- [xx] Implement VJP for `Erf` (dY -> dY _ 2/sqrt(pi) _ exp(-X^2))
- [xx] Implement VJP for `Gelu` (Exact derivative based on Erf)
- [xx] Implement VJP for `HardSwish`
- [xx] Implement VJP for `Softplus` (dY -> dY \* Sigmoid(X))
- [xx] Implement VJP for `Softsign` (dY -> dY / (1 + |X|)^2)
- [xx] Implement explicit Un-broadcasting logic (ReduceSum over broadcasted axes) internally for all binary math
- [xx] Verify shape extraction cleanly supports dynamic un-broadcasting logic
- [xx] Implement non-differentiable markers for `Round`, `Floor`, `Ceil` (0 gradient)
- [xx] Implement non-differentiable markers for logical ops (`Equal`, `Less`)
- [xx] Implement VJP for `Where` (Condition routing)

### 3. Neural Network Layers VJP Parity (40+ items)

- [xx] Implement VJP for `MatMul` (dY -> dX1=dY @ X2^T, dX2=X1^T @ dY)
- [xx] Implement VJP for `Gemm` (handling alpha, beta, transA, transB cleanly)
- [xx] Implement VJP for `Conv` (dX -> ConvTranspose(dY, W), dW -> Conv2D(X, dY), dB -> ReduceSum(dY))
- [xx] Handle Conv strides natively in VJP
- [xx] Handle Conv padding natively in VJP
- [xx] Handle Conv dilations natively in VJP
- [xx] Handle Conv groups (Depthwise) natively in VJP
- [xx] Implement VJP for `ConvTranspose` (dX -> Conv(dY, W))
- [xx] Implement VJP for `MaxPool` (dX -> ScatterND routing gradients to ArgMax indices)
- [xx] Implement VJP for `AveragePool` (dX -> Distribute dY / pool_size uniformly)
- [xx] Implement VJP for `GlobalAveragePool` (dX -> Expand dY / (H\*W))
- [xx] Implement VJP for `GlobalMaxPool` (dX -> ScatterND routing)
- [xx] Implement VJP for `Softmax` (dY -> dY _ Softmax(X) - ReduceSum(dY _ Softmax(X)) \* Softmax(X))
- [xx] Implement VJP for `LogSoftmax`
- [xx] Implement VJP for `BatchNormalization` (Training mode: Tracking mean/var + gradient derivation)
- [xx] Implement VJP for `LayerNormalization` (Gradient with respect to X, Scale, Bias)
- [xx] Implement VJP for `InstanceNormalization`
- [xx] Implement VJP for `Dropout` (dX -> dY \* Mask / (1 - ratio))
- [xx] Implement VJP for `Pad` (dX -> Slice(dY, removing padded regions))
- [xx] Implement VJP for `Resize` (Nearest -> Pooling/Gathering logic)
- [xx] Implement VJP for `Resize` (Linear/Bilinear -> Sparse distribution logic)
- [xx] Implement VJP for `Flatten` (dX -> Reshape(dY, original_shape))
- [xx] Implement VJP for `Reshape` (dX -> Reshape(dY, original_shape))
- [xx] Implement VJP for `Transpose` (dX -> Transpose(dY, inverse_permutation))
- [xx] Implement VJP for `Squeeze` (dX -> Unsqueeze(dY))
- [xx] Implement VJP for `Unsqueeze` (dX -> Squeeze(dY))
- [xx] Implement VJP for `Concat` (dX_i -> Slice(dY, along axis i))
- [xx] Implement VJP for `Split` (dX -> Concat(dY_i))
- [xx] Implement VJP for `Slice` (dX -> Pad(dY, filling zero in sliced out regions))
- [xx] Implement VJP for `Gather` (dX -> ScatterND/ScatterElements(dY) accumulating on indices)
- [xx] Implement VJP for `ScatterElements` (dX -> Gather(dY))
- [xx] Implement VJP for `Tile` (dX -> ReduceSum(dY, along tiled axes))
- [xx] Implement VJP for `Expand` (dX -> ReduceSum(dY, along expanded axes))
- [xx] Implement VJP for `SequenceConstruct`
- [xx] Implement VJP for `SplitToSequence`
- [xx] Handle dynamic unrolling for recurrent layers (`RNN`, `LSTM`, `GRU`) if traced natively

### 4. Loss Functions & Objective Compilation (20+ items)

- [xx] Compile `MeanSquaredError` natively (Forward + VJP generator)
- [xx] Compile `MeanAbsoluteError` natively
- [xx] Compile `HuberLoss` natively
- [xx] Compile `BinaryCrossEntropy` natively
- [xx] Compile `BinaryCrossEntropyWithLogits` natively (Fused for numerical stability)
- [xx] Compile `CategoricalCrossEntropy` natively
- [xx] Compile `SoftmaxCrossEntropyLoss` natively (ONNX operator representation)
- [xx] Compile `NegativeLogLikelihoodLoss` natively (ONNX operator representation)
- [xx] Compile `KullbackLeiblerDivergence` natively
- [xx] Compile `CosineEmbeddingLoss` natively
- [xx] Compile `MarginRankingLoss` natively
- [xx] Compile `TripletMarginLoss` natively
- [xx] Support custom user-defined loss graphs (trace any math sub-graph dynamically)
- [xx] Allow explicit masking of specific targets (e.g., `-100` ignore index in NLP)
- [xx] Allow class-weighted loss application
- [xx] Provide reduction semantics: `mean`, `sum`, `none` natively
- [xx] Compute loss scaling explicitly for mixed precision workflows (Float16)
- [xx] Auto-inject loss node directly into the monolithic training graph

### 5. Optimizers & Weight Update Integration (25+ items)

- [xx] AOT compile `SGD` optimizer directly into the training graph
- [xx] AOT compile `SGD + Momentum` natively
- [xx] AOT compile `Adam` directly into the training graph
- [xx] AOT compile `AdamW` (Adam with weight decay) natively
- [xx] AOT compile `RMSProp` natively
- [xx] AOT compile `Adagrad` natively
- [xx] AOT compile `Adadelta` natively
- [xx] Expose global `learning_rate` as a dynamic ONNX Graph Input
- [xx] Expose global `step` as a dynamic ONNX Graph Input (for Adam bias correction)
- [xx] Expose `beta1`, `beta2`, `epsilon` as dynamic Inputs/Constants
- [xx] Allocate state tensors natively (e.g. `Momentum_1`, `Momentum_2`) as Graph Inputs and Outputs (Loop carried states)
- [xx] Replace `Parameter` nodes with explicitly updated nodes `W_new = W_old - lr * grad`
- [xx] Embed L2 Regularization (Weight Decay) math explicitly into the gradient before optimizer step
- [xx] Implement Gradient Clipping (Global Norm) natively across all gradients before optimizer step
- [xx] Implement Gradient Clipping (Value based) natively
- [xx] Output a fully encapsulated `TrainingStep` graph: `[Inputs, Targets, LR, State] -> [Loss, New_States, New_Weights]`
- [xx] Output a simplified `AccumulateGradients` graph
- [xx] Output a simplified `ApplyOptimizer` graph
- [xx] Ensure optimizer mathematical equivalence with standard PyTorch optimizers (atol=1e-5)
- [xx] Auto-generate initialization values for optimizer states (Zero tensors) dynamically
- [xx] Support generating training graphs directly targeting WebGPU execution limits

### 6. Zero-Dependency & Web/Server Interop (20+ items)

- [xx] No `onnxruntime-training` wheel required natively in Python
- [xx] No CMake, Ninja, or LLVM bindings required to trace gradients
- [xx] Execute compiled training graphs instantly inside standard Web Browsers (WASM/WebGPU)
- [xx] Execute compiled training graphs instantly on AWS Lambda / edge devices
- [xx] Provide TypeScript/JS API to feed batches into the WebGPU training graph dynamically
- [xx] Ensure training graphs do not contain proprietary `com.microsoft` opsets
- [xx] Provide explicit checkpoint saving (exporting updated weights back to `.safetensors`)
- [xx] Provide explicit checkpoint loading in WASM
- [xx] Track VRAM usage of the training graph natively using `onnx-tool` reimplementation
- [xx] Estimate batch size limits statically before OOM occurs in the browser
- [xx] Stream training data incrementally from IndexedDB/Fetch directly into the graph
- [xx] Support Federated Learning explicitly (exporting delta gradients instead of applying them natively)
- [xx] Calculate gradient communication bounds (MB/s) for federated updates
- [xx] Expose Python API `onnx9000.training.compile_training_graph(model, loss, optimizer)`

### 7. Explicit Unit Tests & Edge Cases (30+ items)

- [xx] Unit Test: Train standard Linear Regression completely in ONNX (Loss decreasing over 10 steps)
- [xx] Unit Test: Train Logistic Regression natively
- [xx] Unit Test: Train MLP (Multi-Layer Perceptron) 2-layers (Matching PyTorch loss curve exactly)
- [xx] Unit Test: Train standard CNN (Conv+Relu+MaxPool+Linear) (Matching PyTorch gradients exactly)
- [xx] Unit Test: Fine-tune a pre-trained ResNet bottleneck layer
- [xx] Unit Test: Validate Transformer Attention VJP correctness (QKV gradient routing)
- [xx] Unit Test: Confirm LayerNorm gradients match PyTorch natively
- [xx] Unit Test: Confirm BatchNorm tracking variables update correctly during training mode
- [xx] Ensure evaluation mode (inference) accurately freezes BatchNorm statistics
- [xx] Ensure evaluation mode accurately disables Dropout logic
- [xx] Ensure gradients correctly accumulate when a weight tensor is used multiple times (Shared parameters)
- [xx] Test VJP shape preservation across `Reshape` boundaries exactly
- [xx] Test VJP un-broadcasting safely handles implicit right-alignment natively
- [xx] Test VJP of `MatMul` with implicit batch expansion
- [xx] Catch un-differentiable operations cleanly and raise precise `RuntimeError` warnings
- [xx] Validate `Float16` numeric stability across Adam bias correction loops

### 8. Exhaustive Gradient Operators & Subgraph Tracing (40+ items)

- [xx] Implement VJP for `TopK` (dX -> ScatterND routing gradients to extracted indices)
- [xx] Implement VJP for `SpaceToDepth` (dX -> DepthToSpace(dY))
- [xx] Implement VJP for `DepthToSpace` (dX -> SpaceToDepth(dY))
- [xx] Implement VJP for `Acos` (dY -> -dY / Sqrt(1 - X^2))
- [xx] Implement VJP for `Acosh` (dY -> dY / Sqrt(X^2 - 1))
- [xx] Implement VJP for `Asin` (dY -> dY / Sqrt(1 - X^2))
- [xx] Implement VJP for `Asinh` (dY -> dY / Sqrt(X^2 + 1))
- [xx] Implement VJP for `Atan` (dY -> dY / (1 + X^2))
- [xx] Implement VJP for `Atanh` (dY -> dY / (1 - X^2))
- [xx] Implement VJP for `Celul` (dY -> dY _ Where(X > 0, 1, alpha _ Exp(X)))
- [xx] Implement VJP for `Elu` (dY -> dY _ Where(X > 0, 1, alpha _ Exp(X)))
- [xx] Implement VJP for `Selu` (dY -> dY _ scale _ Where(X > 0, 1, alpha \* Exp(X)))
- [xx] Implement VJP for `Mish` (dY -> dY _ (Softplus(X) + X _ Sigmoid(X) \* (1 - Tanh(Softplus(X))^2)))
- [xx] Implement VJP for `Shrink`
- [xx] Implement VJP for `CumSum` (dX -> CumSum(dY, reverse=True))
- [xx] Implement VJP for `ReverseSequence` (dX -> ReverseSequence(dY))
- [xx] Implement VJP for `Compress` (dX -> Scatter Elements on condition)
- [xx] Implement VJP for `Trilu` (dX -> Trilu(dY))
- [xx] Implement VJP for `GatherElements` (dX -> ScatterElements(dY))
- [xx] Implement VJP for `GatherND` (dX -> ScatterND(dY))
- [xx] Implement VJP for `ScatterND` (dX -> GatherND(dY), dUpdates -> GatherND(dY))
- [xx] Implement VJP for `ScatterElements` (dX -> GatherElements(dY), dUpdates -> GatherElements(dY))
- [xx] Implement VJP for `ReduceMax` (dX -> ScatterND(dY) distributing to ArgMax indices)
- [xx] Implement VJP for `ReduceMin` (dX -> ScatterND(dY) distributing to ArgMin indices)
- [xx] Implement VJP for `ReduceMean` (dX -> Expand(dY / N))
- [xx] Implement VJP for `ReduceProd` (dX -> dY \* Prod(X) / X)
- [xx] Implement VJP for `ReduceL1` (dX -> dY \* Sign(X))
- [xx] Implement VJP for `ReduceL2` (dX -> dY \* X / L2_Norm(X))
- [xx] Implement VJP for `ReduceLogSum` (dX -> Expand(dY) / Sum(X))
- [xx] Implement VJP for `ReduceLogSumExp` (dX -> Expand(dY) \* Softmax(X))
- [xx] Implement VJP for `ReduceSumSquare` (dX -> 2 _ X _ Expand(dY))
- [xx] Implement VJP for `LpNormalization` (Gradient of L1/L2 Norm)
- [xx] Implement VJP for `GlobalLpPool` (Gradient of Lp Norm)
- [xx] Implement VJP for `Einsum` (Symbolically generating the adjoint equation)
- [xx] Implement VJP for `LayerNormalization` dynamically resolving axis constraints
- [xx] Implement VJP for `MaxRoiPool` (dX -> ScatterND to ArgMax indices per RoI)
- [xx] Implement VJP for `RoiAlign` (dX -> ScatterND using bilinear interpolation weights natively)
- [xx] Prevent gradient flow into dynamic Shape tensors natively (`NonZero`, `Shape`, `Size`)
- [xx] Allow marking specific inputs dynamically as `requires_grad=False` inside Python API

### 9. Advanced Mixed Precision & Gradient Scaling (25+ items)

- [xx] Inject explicit `Cast` operators dynamically to cast Gradients to FP16
- [xx] Maintain Master Weights explicitly in FP32 format natively in the generated graph
- [xx] Cast Master Weights explicitly to FP16 ONLY during the Forward Pass evaluations
- [xx] Accumulate all gradients purely in FP32
- [xx] Implement dynamic Loss Scaling directly into the Graph topology (Multiplying Loss by S)
- [xx] Implement Un-Scaling directly into the Graph topology (Dividing Gradients by S)
- [xx] Implement `IsInf` / `IsNaN` checkers recursively across all un-scaled Gradients
- [xx] Implement `If` logic natively to skip Optimizer Updates if `NaN`/`Inf` is detected (Dynamic Graph Logic)
- [xx] Implement dynamic Loss Scale adjustment logic (Increase if successful, Halve if `NaN` detected)
- [xx] Validate standard AMP (Automatic Mixed Precision) PyTorch rules natively inside the AOT transpiler
- [xx] Implement BFloat16 (`bfloat16`) equivalents natively for training without dynamic Loss Scaling
- [xx] Optimize intermediate `Cast` nodes intelligently (canceling out sequential casts)
- [xx] Inject `MemcpyToHost` / `MemcpyToDevice` automatically across mixed precision boundaries if targeted
- [xx] Guarantee numerical equivalence of FP16 backwards pass against PyTorch AMP (atol=1e-3)

### 10. Federated & Distributed Training Native Graph Integrations (25+ items)

- [xx] Output isolated `CalculateGradient` Sub-graph natively (No Optimizer)
- [xx] Output isolated `AccumulateGradient` Sub-graph natively (No Forward Pass)
- [xx] Implement Ring AllReduce simulation topology natively using `Add` / `Div` ops
- [xx] Prepare model for Parameter Server deployments (Inputs = Weights/Grads, Outputs = New Weights)
- [xx] Embed unique `NodeArg` identifiers to coordinate distributed weight synchronization automatically
- [xx] Calculate theoretical gradient payload sizes natively using `onnx-tool` profiling
- [xx] Expose native Python/WASM API for calculating gradient deltas only (delta = NewWeight - OldWeight)
- [xx] Implement Differential Privacy natively (Adding `RandomNormal` noise to Gradients explicitly before export)
- [xx] Implement Gradient Clipping to L2 Norm (Local DP constraints) statically in the graph
- [xx] Compile multi-replica data parallel topologies into a single massive batched graph if requested
- [xx] Expose distributed synchronous barrier points statically inside the execution provider
- [xx] Extract and flatten all gradients cleanly into a single massive 1D Tensor dynamically for network transfer
- [xx] Expand received 1D Gradients back into topological layers natively (`Split` + `Reshape`)
- [xx] Expose an API to dynamically compress gradients via INT8 Quantization before transmission

### 11. LoRA & Parameter Efficient Fine Tuning (PEFT) (25+ items)

- [xx] Expose API to freeze arbitrary model layers mathematically (`requires_grad=False`)
- [xx] Inject Low Rank Adaptation (LoRA) A and B weight matrices statically into a pre-existing `MatMul`
- [xx] Rewrite standard `Gemm` into `Gemm(X, W) + Gemm(Gemm(X, LoRA_A), LoRA_B)`
- [xx] Extract gradients _exclusively_ for `LoRA_A` and `LoRA_B` (Saving 99% of backward pass memory)
- [xx] Compile LoRA-specific optimizer steps (only updating A and B tensors)
- [xx] Support generating isolated `LoRA.safetensors` dynamically containing only the trained adapters
- [xx] Support merging `LoRA` adapters back into the Master Weights statically inside `GraphSurgeon`
- [xx] Support injecting BitFit (Bias-only fine tuning) explicitly
- [xx] Support Prefix Tuning dynamically via ONNX `Concat` node injection
- [xx] Support Prompt Tuning natively via ONNX `Gather` / `Concat` node injection
- [xx] Profile peak memory allocation for LoRA vs Full Fine-Tuning mathematically
- [xx] Test LoRA backwards pass compilation across massive LLMs (Llama 3, Mistral) in under 5 seconds natively
- [xx] Emulate `peft` library configurations cleanly using dictionary configurations

### 12. Complete Testing & Opset Validations (25+ items)

- [xx] Ensure the AOT Autograd Compiler targets ONNX Opset 15 by default
- [xx] Ensure the AOT Autograd Compiler targets ONNX Opset 16 by default
- [xx] Ensure the AOT Autograd Compiler targets ONNX Opset 17 by default
- [xx] Ensure the AOT Autograd Compiler targets ONNX Opset 18 by default
- [xx] Ensure the AOT Autograd Compiler targets ONNX Opset 19 by default
- [xx] Ensure the AOT Autograd Compiler targets ONNX Opset 20 by default
- [xx] Ensure the AOT Autograd Compiler targets ONNX Opset 21 by default
- [xx] Unit Test: Differentiate `Reshape` recursively inside a deeply nested `If` node
- [xx] Unit Test: Differentiate `Split` recursively inside a `Loop` node (BPTT - Backprop Through Time)
- [xx] Verify execution exactly matches PyTorch Autograd traces for specific edge-case broadcasting tests
- [xx] Verify Execution exactly matches PyTorch Autograd traces for negative indices tests
- [xx] Catch explicitly un-differentiable operations recursively through CustomOps
- [xx] Test topological sorting guarantees the generated backward graph is strictly acyclic
- [xx] Test Optimizer graphs execute flawlessly sequentially across 1,000 steps without NaN
- [xx] Benchmark full PyTorch Native Training Loop vs compiled `onnx9000.Training` loop on CPU

### 13. Autograd Graph Memory Optimization (Memory Reuse & Caching) (25+ items)

- [xx] Implement Activation Checkpointing (Recomputation) natively in the AOT graph
- [xx] Inject explicit `If` logic or topological re-evaluations to discard intermediate activations (saving VRAM)
- [xx] Recompute `Relu` natively during the backward pass (zero memory footprint caching)
- [xx] Recompute `Silu` / `Swish` natively during the backward pass
- [xx] Recompute `Gelu` natively during the backward pass
- [xx] Analyze exactly which Forward activations MUST be cached for the Backward pass (e.g. `MatMul` inputs)
- [xx] Analyze exactly which Forward activations can be safely discarded (e.g. `Add` inputs if shapes match)
- [xx] Insert explicit `Yield` / `Return` nodes for cached activations to jump from Forward to Backward seamlessly
- [xx] Optimize intermediate gradient buffer reuse (`dY` -> `dX1` + `dX2` sharing memory statically)
- [xx] Inject `Inplace` operation hints recursively (`Add(A, B, out=A)`) for gradients where supported
- [xx] Analyze exact peak memory of the training graph vs inference graph using `onnx-tool` profiling
- [xx] Strip out dropout nodes automatically if tracing an inference-only `eval` subgraph
- [xx] Manage random seeds natively for `Dropout` reproducibility between Forward and Backward passes
- [xx] Ensure `BatchNormalization` running mean/var updates are completely detached from gradient tracking
- [xx] Guarantee no circular references exist within the activation cache routing logic

### 14. Advanced Loss & Numeric Stability Assertions (15+ items)

- [xx] Handle `SoftmaxCrossEntropyLoss` combined VJP explicitly (dY = Softmax(X) - Target) to avoid NaN
- [xx] Handle `BCEWithLogitsLoss` combined VJP explicitly (dY = Sigmoid(X) - Target) to avoid NaN
- [xx] Ensure `Log` inside losses is clamped (`Log(Clamp(X, eps, 1-eps))`) to prevent `Log(0) = -Inf`
- [xx] Ensure `Div` by zero inside `ReduceMean` tracking (e.g. dynamic batch sizes of 0) returns 0 gracefully
- [xx] Handle numerical underflow in `Exp` operations during backprop using `LogSumExp` tricks natively
- [xx] Support smoothing parameters natively inside `CategoricalCrossEntropy` (Label Smoothing)
- [xx] Emulate `FocalLoss` gradients natively (dY = (1-p)^gamma _ (gamma _ p \* log(p) + p - 1))
- [xx] Emulate `DiceLoss` gradients natively
- [xx] Prevent gradient explosion globally across the backward pass by injecting static `Clip` layers before optimizers
- [xx] Support explicit gradient penalty calculation natively (`Norm(dY) - 1.0` added to Loss)

### 15. Additional Specific Gradient Operations & Hooks (10+ items)

- [xx] Implement VJP for `SpaceToBatchND`
- [xx] Implement VJP for `BatchToSpaceND`
- [xx] Implement VJP for `BitShift` (Non-differentiable -> 0 gradient)
- [xx] Implement VJP for `Round` (Non-differentiable -> 0 gradient)
- [xx] Provide Python-level `register_vjp(op_type, custom_vjp_function)` API for extensibility
- [xx] Support tracing explicitly custom/unknown domains by relying on user-provided VJPs
- [xx] Provide analytical Jacobian Matrix generator explicitly (for tiny matrices only)

### 16. Final AOT Compiler Tests & Integration Rules (5+ items)

- [xx] Unit Test: Compile pure `Einsum` into VJP and test against standard `MatMul` representations
- [xx] Unit Test: Differentiate nested `Slice` operations with negative bounds accurately
- [xx] Unit Test: Track gradient of completely scalar operations accurately
- [xx] Unit Test: Test gradient flow correctly completely stops at `StopGradient` pseudo-nodes
- [xx] Ensure execution overhead of AOT training graph matches standard inference overhead completely (0 runtime tracing cost)
