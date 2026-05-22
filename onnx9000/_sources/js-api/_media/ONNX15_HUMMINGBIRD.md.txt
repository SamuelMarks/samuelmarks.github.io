# Hummingbird Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `Hummingbird` within the `onnx9000` ecosystem.
The original `Hummingbird` compiles traditional machine learning models (decision trees, random forests, linear models) into PyTorch or ONNX tensor operations. This allows traditional ML to leverage GPU hardware acceleration via matrix multiplications and advanced tensor operations rather than slow CPU branching.
Our `onnx9000` reimplementation focuses on transpiling traditional models into pure `ai.onnx` (core mathematical ops) natively in Python. This zero-dependency translation is incredibly powerful for **WebGPU** and **WASM** execution, as GPUs suffer massive performance penalties from branch divergence (if/else statements in trees). By lowering trees to tensor math, we unlock parallelized, hardware-accelerated inference for traditional ML directly inside the browser.

## Exhaustive Parity Checklist

### 1. Core Architecture & Strategy Selection (25+ items)

- [x] Implement zero-dependency transpilation engine architecture
- [x] Implement strategy selector based on hardware target (CPU vs GPU vs WebGPU)
- [x] Implement strategy selector based on tree depth / sparsity
- [x] Support `gemm` (General Matrix Multiply) execution strategy
- [x] Support `tree_trav` (Tree Traversal) execution strategy
- [x] Support `perf_tree_trav` (Perfect Tree Traversal) execution strategy
- [x] Implement memory-footprint estimator to auto-select optimal strategy
- [x] Allow explicit strategy overrides by the user
- [x] Map tree nodes to intermediate tensor abstractions natively
- [x] Support batch-size optimizations (switching strategy based on `batch_size`)
- [x] Implement backend registry for extensibility (e.g., pure ONNX, custom WGSL)
- [x] Ensure transpiled graphs use exclusively `ai.onnx` operators (no `ai.onnx.ml`)
- [x] Prevent generation of branch operators (`If`, `Loop`) in tree bodies
- [x] Provide dynamic batching support internally via symbolic dimensions
- [x] Handle models with a mix of categorical and continuous features natively
- [x] Transpile numerical threshold comparisons accurately
- [x] Handle structural missing values (NaN) within tensor operations
- [x] Extract global constants into contiguous initialized Tensors
- [x] Provide tree depth analysis utility (min, max, mean depths)
- [x] Provide tree leaf distribution utility
- [x] Parse ensemble weights natively (for AdaBoost / Weighted Forests)
- [x] Flatten nested ensemble structures into unified 2D/3D tensors
- [x] Resolve numerical precision mismatches (FP32 vs FP64) ahead of time
- [x] Cast FP64 parameters to FP32 natively to optimize WebGPU limits
- [x] Expose verbosity and logging to debug transpilation steps

### 2. GEMM (Matrix Multiplication) Strategy (35+ items)

- [x] Map internal tree decision nodes to Matrix A (feature indices)
- [x] Map internal tree thresholds to Matrix B (comparisons)
- [x] Map tree leaf values to Matrix C (predictions)
- [x] Map tree node routing logic to Matrix D (left/right path tracking)
- [x] Implement `MatMul` for feature selection
- [x] Implement `Less` / `Greater` ops for threshold evaluation natively
- [x] Map boolean path selections using `Sign` and `Relu` math tricks
- [x] Map bitwise routing using exact integer multiplications
- [x] Implement `ArgMax` for final leaf selection in the matrix space
- [x] Support multi-class prediction packing in GEMM C matrices
- [x] Compile Random Forest (multiple trees) into batched 3D `MatMul`
- [x] Compile Gradient Boosting into sequential 2D `MatMul` additions
- [x] Implement GEMM sparsity optimizations (removing dead matrix columns)
- [x] Compress Matrix A (feature selectors) using one-hot/sparse representations
- [x] Ensure GEMM generated graph avoids `Gather` ops for maximum ALU utilization
- [x] Exploit ONNX `Gemm` operator's built-in `alpha` and `beta` parameters
- [x] Transpile `Sum` reduction over ensemble outputs natively
- [x] Address intermediate tensor memory blow-up on extremely deep trees
- [x] Optimize GEMM memory using block-diagonal matrix representations for ensembles
- [x] Implement partial GEMM execution for trees evaluated in chunks
- [x] Map missing values to extreme matrix thresholds (`+inf` / `-inf`)
- [x] Utilize `Where` operator to conditionally zero-out path matrices
- [x] Support GEMM lowering for DecisionTreeRegressors
- [x] Support GEMM lowering for DecisionTreeClassifiers
- [x] Support GEMM lowering for IsolationForests
- [x] Validate GEMM output numerical equivalence against reference models
- [x] Measure and optimize peak VRAM usage of GEMM constants
- [x] Compile leaf node outputs into `Concat` + `MatMul`
- [x] Pre-compute scaling factors in GEMM matrices
- [x] Merge bias additions directly into GEMM Matrix C
- [x] Support batch dynamic sizes `[N, features]` strictly in GEMM `MatMul`
- [x] Resolve dimension broadcasting constraints statically
- [x] Pack tree indices to minimize matrix row sizes
- [x] Detect and eliminate redundant threshold checks across identical trees
- [x] Implement binary encoding of tree paths for logarithmic complexity

### 3. TreeTraversal Strategy (25+ items)

- [x] Map tree structures to flat 1D index arrays
- [x] Map feature threshold values to flat 1D arrays
- [x] Implement dynamic array indexing using ONNX `Gather`
- [x] Implement iterative gathering (simulating tree descent) without `Loop`
- [x] Track left child indices in a parallel tensor
- [x] Track right child indices in a parallel tensor
- [x] Track feature indices in a parallel tensor
- [x] Use `Less` / `Greater` to generate binary offsets (0 or 1)
- [x] Multiply binary offsets by jump strides
- [x] Use `Add` to compute next node index natively
- [x] Support batched gathering (batch of inputs traversing trees simultaneously)
- [x] Handle leaf node identification (e.g., negative index markers)
- [x] Use `Where` to freeze indices of rows that have reached a leaf
- [x] Handle uneven tree depths via masked padding
- [x] Optimize Gather operations by merging index tensors
- [x] Implement parallel traversal of all trees in an ensemble using batched Gathers
- [x] Transpile `TreeEnsembleRegressor` using TreeTraversal
- [x] Transpile `TreeEnsembleClassifier` using TreeTraversal
- [x] Manage ONNX sequence lengths statically for unrolled traversals
- [x] Extract maximum tree depth to dictate unrolling iterations
- [x] Pre-allocate output tensors for traversal aggregations
- [x] Handle missing value routing natively within gathered offsets
- [x] Implement categorical feature gathering (equality checks vs inequalities)
- [x] Flatten multi-class leaf outputs into parallel gathers
- [x] Test and validate latency of `Gather` bounds on WASM

### 4. PerfectTree Traversal Strategy (20+ items)

- [x] Pad all trees to perfectly balanced binary trees (depth $D$)
- [x] Calculate $2^D - 1$ required node capacities natively
- [x] Map perfect tree structure to implicit binary heap indices (e.g., $2i+1$, $2i+2$)
- [x] Eliminate explicit left/right index arrays (computed purely mathematically)
- [x] Transpile feature indices to perfect tree array
- [x] Transpile thresholds to perfect tree array
- [x] Transpile leaf values to perfect tree array
- [x] Use bitwise or arithmetic shifts (if supported/emulated) for node traversal
- [x] Implement unrolled loop of depth $D$ using pure arithmetic
- [x] Use `Gather` only for feature extraction and final leaf value
- [x] Heavily compress memory footprint of perfect trees vs GEMM
- [x] Support dynamic depth selection based on ensemble characteristics
- [x] Detect and trim physically unreachable perfect tree branches
- [x] Map categorical branches effectively within perfect node constraints
- [x] Handle multi-output regression perfectly aligned
- [x] Test memory constraints on WebGPU for deep PerfectTrees ($D > 15$)
- [x] Optimize padding values to bypass threshold evaluations cleanly
- [x] Implement early exit masking (simulated) for shallow branches in a perfect tree
- [x] Validate execution efficiency on wide arrays
- [x] Profile compilation time overhead of perfect tree padding

### 5. Scikit-Learn Translators (35+ items)

- [x] Parse `DecisionTreeClassifier` into Intermediate Representation
- [x] Parse `DecisionTreeRegressor` into Intermediate Representation
- [x] Parse `RandomForestClassifier` into Intermediate Representation
- [x] Parse `RandomForestRegressor` into Intermediate Representation
- [x] Parse `ExtraTreesClassifier` into Intermediate Representation
- [x] Parse `ExtraTreesRegressor` into Intermediate Representation
- [x] Parse `GradientBoostingClassifier` into Intermediate Representation
- [x] Parse `GradientBoostingRegressor` into Intermediate Representation
- [x] Parse `HistGradientBoostingClassifier` into Intermediate Representation
- [x] Parse `HistGradientBoostingRegressor` into Intermediate Representation
- [x] Parse `IsolationForest` into Intermediate Representation
- [x] Parse `AdaBoostClassifier` into Intermediate Representation
- [x] Parse `AdaBoostRegressor` into Intermediate Representation
- [x] Extract `n_estimators`, `max_depth`, and tree arrays automatically
- [x] Parse `LinearRegression` -> Tensor Math
- [x] Parse `LogisticRegression` -> Tensor Math + Sigmoid/Softmax
- [x] Parse `Ridge`, `Lasso`, `ElasticNet` -> Tensor Math
- [x] Parse `SGDClassifier` -> Tensor Math
- [x] Parse `LinearSVC` -> Tensor Math + Sign
- [x] Parse `SVC` (Poly) -> Tensor Math
- [x] Parse `SVC` (RBF) -> Tensor Math
- [x] Parse `SVC` (Sigmoid) -> Tensor Math
- [x] Parse `GaussianNB` -> Tensor Math
- [x] Parse `MultinomialNB` -> Tensor Math
- [x] Parse `BernoulliNB` -> Tensor Math
- [x] Parse `MLPClassifier` -> Tensor Math (Pure ONNX dense layers)
- [x] Extract classes and mapping them to output ZipMaps / Tensors
- [x] Parse pipeline structures seamlessly
- [x] Handle `predict_proba` via post-processing mathematical transformations
- [x] Handle multi-output regressors (n_targets > 1) natively
- [x] Handle multi-label classification natively
- [x] Bypass Scikit-Learn C++ extensions, extracting directly from Python object properties
- [x] Optimize Scikit-Learn `StandardScaler` to ONNX `Add` + `Mul`
- [x] Optimize Scikit-Learn `Binarizer` to ONNX `Greater` + `Cast`
- [x] Optimize Scikit-Learn `OneHotEncoder` to ONNX `Equal` / `ScatterND`

### 6. LightGBM Translators (25+ items)

- [x] Parse `LGBMClassifier` directly from Python memory
- [x] Parse `LGBMRegressor` directly from Python memory
- [x] Parse `LGBMRanker` directly from Python memory
- [x] Extract LightGBM booster dumps (JSON) strictly in memory
- [x] Transpile LightGBM default missing value behaviors to tensor masks
- [x] Handle LightGBM `max_bin` constraints within tensor limits
- [x] Transpile LightGBM categorical features (bitset evaluations) to `Gather` / `Equal` chains
- [x] Extract LightGBM `sigmoid` parameter for binary classification output
- [x] Map LightGBM multiclass Objective (Softmax) to ONNX `Softmax`
- [x] Map LightGBM multiclassova Objective to ONNX `Sigmoid`
- [x] Map LightGBM regression objectives (RMSE, L1, Huber) to raw outputs
- [x] Flatten LightGBM leaf weights into GEMM Matrix C
- [x] Transpile LightGBM Poisson objective -> `Exp` math node
- [x] Transpile LightGBM Tweedie objective -> `Exp` math node
- [x] Map LightGBM leaf output scaling (learning rate / base score) into matrix biases
- [x] Handle explicit `num_class` parameter conversions safely
- [x] Test performance on >5,000 tree LightGBM models
- [x] Provide error boundaries for unsupported custom LightGBM loss functions
- [x] Optimize `limit_max_depth` trees explicitly into PerfectTree strategies
- [x] Strip LightGBM specific feature names mapping to strict ONNX indices
- [x] Support boolean vs integer encoding for LightGBM split paths
- [x] Validate transpiled output against LightGBM native `predict()` (rtol=1e-5)
- [x] Validate transpiled output against LightGBM native `predict_proba()`
- [x] Parse categorical threshold subsets natively (simulating `in` operators)
- [x] Compress large categorical bitsets using int64 arithmetic in ONNX

### 7. XGBoost & CatBoost Translators (25+ items)

- [x] Parse `XGBClassifier` directly from Python memory
- [x] Parse `XGBRegressor` directly from Python memory
- [x] Parse `XGBRanker` directly from Python memory
- [x] Load XGBoost Booster JSON dumps natively
- [x] Extract XGBoost `base_score` and bake into output `Add` tensor
- [x] Transpile XGBoost `missing` child routing natively into `Where` masks
- [x] Support XGBoost `dart` (Dropout Additive Regression Trees) by applying static weight scales
- [x] Map XGBoost `binary:logistic` to `Sigmoid`
- [x] Map XGBoost `binary:logitraw` to Raw Output
- [x] Map XGBoost `multi:softmax` to `ArgMax`
- [x] Map XGBoost `multi:softprob` to `Softmax`
- [x] Map XGBoost `count:poisson` to `Exp`
- [x] Handle multi-target regression mapping cleanly
- [x] Parse CatBoost `CatBoostClassifier` directly
- [x] Parse CatBoost `CatBoostRegressor` directly
- [x] Leverage CatBoost Oblivious Trees to natively map to the PerfectTree Strategy
- [x] Extract symmetric CatBoost thresholds efficiently
- [x] Extract CatBoost leaf value arrays without duplication
- [x] Handle CatBoost one-hot encoded categorical variables mathematically
- [x] Bypass CTR (Categorical Target Encoding) constraints via explicit Gather maps
- [x] Map CatBoost `Logloss` -> `Sigmoid`
- [x] Map CatBoost `MultiClass` -> `Softmax`
- [x] Validate XGBoost transpilation against native Python outputs
- [x] Validate CatBoost transpilation against native Python outputs
- [x] Ensure strict dynamic batch size `N` capability across all compiled matrices

### 8. ONNX-ML Lowering (30+ items)

- [x] Provide explicit converter for `ai.onnx.ml.TreeEnsembleClassifier` -> `ai.onnx` Math
- [x] Provide explicit converter for `ai.onnx.ml.TreeEnsembleRegressor` -> `ai.onnx` Math
- [x] Provide explicit converter for `ai.onnx.ml.LinearClassifier` -> `ai.onnx` Math
- [x] Provide explicit converter for `ai.onnx.ml.LinearRegressor` -> `ai.onnx` Math
- [x] Provide explicit converter for `ai.onnx.ml.SVMClassifier` -> `ai.onnx` Math
- [x] Provide explicit converter for `ai.onnx.ml.SVMRegressor` -> `ai.onnx` Math
- [x] Provide explicit converter for `ai.onnx.ml.Scaler` -> `ai.onnx` Math
- [x] Provide explicit converter for `ai.onnx.ml.Normalizer` -> `ai.onnx` Math
- [x] Provide explicit converter for `ai.onnx.ml.Binarizer` -> `ai.onnx` Math
- [x] Provide explicit converter for `ai.onnx.ml.OneHotEncoder` -> `ai.onnx` Math
- [x] Provide explicit converter for `ai.onnx.ml.Imputer` -> `ai.onnx` Math
- [x] Provide explicit converter for `ai.onnx.ml.ArrayFeatureExtractor` -> `ai.onnx.Gather`
- [x] Provide explicit converter for `ai.onnx.ml.CategoryMapper` -> `ai.onnx` Gather/Where
- [x] Provide explicit converter for `ai.onnx.ml.ZipMap` -> standard Tensors + external dictionaries
- [x] Extract `nodes_treeids` natively from `TreeEnsemble`
- [x] Extract `nodes_nodeids` natively from `TreeEnsemble`
- [x] Extract `nodes_featureids` natively from `TreeEnsemble`
- [x] Extract `nodes_values` (thresholds) natively from `TreeEnsemble`
- [x] Extract `nodes_hitrates` (if applicable) natively
- [x] Extract `nodes_modes` (BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, etc.) natively
- [x] Extract `nodes_truenodeids` natively
- [x] Extract `nodes_falsenodeids` natively
- [x] Extract `nodes_missing_value_tracks_true` natively
- [x] Extract `class_treeids` natively
- [x] Extract `class_nodeids` natively
- [x] Extract `class_ids` natively
- [x] Extract `class_weights` natively
- [x] Extract `target_treeids`, `target_nodeids`, `target_ids`, `target_weights` for regressors
- [x] Support `post_transform` extraction (NONE, SOFTMAX, LOGISTIC) natively
- [x] Ensure lowered ONNX subgraphs are perfectly statically shaped

### 9. Advanced Feature Engineering & Mathematics (25+ items)

- [x] Map `CountVectorizer` to native `Equal`, `Cast`, and `ReduceSum` ops
- [x] Map `TfidfVectorizer` to native ONNX math ops using sparse dictionaries
- [x] Map polynomial expansions using `Pow` and `Mul` combinations
- [x] Implement MurmurHash3 purely in ONNX ops for `FeatureHasher` support (if feasible)
- [x] Support embedding lookups using dense `Gather` replacements
- [x] Handle explicit 64-bit to 32-bit integer casting natively
- [x] Resolve numerical instability in Softmax over highly weighted tree leaves
- [x] Optimize Sigmoid using mathematically equivalent faster operations if requested
- [x] Fold sequential `Scaler` -> `LinearRegressor` into a single affine transform natively
- [x] Map KNN distances to `ReduceSumSquare`, `TopK`, and `Gather` natively
- [x] Support Minkowski distances natively in ONNX math
- [x] Handle cosine distance via `LpNormalization` and `MatMul`
- [x] Transpile Naive Bayes log-probabilities natively using `Log` and `Add`
- [x] Replace `Mod` operations with `Div`, `Floor`, and `Sub` if target backend lacks `Mod`
- [x] Replace `Where` with arithmetic masking `(mask * A) + ((1-mask) * B)` for older Opsets
- [x] Enforce broadcast safety (always unsqueeze target tensors prior to arithmetic)
- [x] Test mathematical stability of `Exp` against float16 boundaries
- [x] Ensure `Softmax` numerical stability (subtract max) internally
- [x] Transpile PCA to pure `MatMul` + `Add`
- [x] Transpile TruncatedSVD to pure `MatMul`
- [x] Transpile LDA to dense linear operations
- [x] Optimize one-hot encoder dimensions dynamically
- [x] Provide graph utility to clamp NaN features to zero safely
- [x] Implement robust division by zero guards `Add(x, epsilon)`
- [x] Validate mathematical exactness using symbolic manipulation tools

### 10. Target Post-Processing (15+ items)

- [x] Parse `ZipMap` requirements and emit explicit output sequences
- [x] Provide configuration to omit `ZipMap` for raw tensor performance
- [x] Flatten multi-class `classlabels_strings` into metadata
- [x] Provide ONNX `Cast` nodes for specific output target requirements (e.g., bool outputs)
- [x] Extract predicted classes via `ArgMax`
- [x] Attach `classlabels_ints` to raw indices seamlessly
- [x] Map hierarchical probability distributions cleanly
- [x] Combine multi-output regression lists into contiguous vectors
- [x] Merge multi-label classification into 2D probability matrices
- [x] Emit specific named outputs (`label`, `probabilities`) reliably
- [x] Append top-K post-processing dynamically to the lowered graph
- [x] Output logits / pre-activation scores on demand (bypassing Sigmoid/Softmax)
- [x] Scale output probabilities by calibration factors statically
- [x] Correctly manage `batch_size=1` specific dimensional drops
- [x] Append confidence score derivations directly into the ONNX graph

### 11. WebGPU & WASM Execution Optimizations (35+ items)

- [x] Eliminate all branch divergence to maximize WebGPU warp efficiency
- [x] Ensure all generated constants fit within WebGPU max buffer sizes (128MB/256MB)
- [x] Auto-chunk massive GEMM matrices into smaller tiled `MatMul` sequences if needed
- [x] Validate memory alignment of `Float32` constants for WASM direct ingestion
- [x] Pre-transpose Matrix B dynamically during generation to leverage WebGPU layout efficiency
- [x] Pre-transpose Matrix A if required for optimal WGSL shader performance
- [x] Avoid `Gather` ops on highly fragmented indices to prevent WebGPU L1 cache misses
- [x] Utilize `ConstantOfShape` to dynamically allocate scratchpad memory inside the graph
- [x] Test WASM execution footprint of PerfectTree vs GEMM (WASM prefers GEMM)
- [x] Benchmark TreeTraversal strategy on WebAssembly CPU (usually faster than WebGPU for deep/sparse trees)
- [x] Expose heuristic flag `force_webgpu` to enforce GEMM regardless of tree depth
- [x] Expose heuristic flag `force_wasm` to enforce TreeTraversal
- [x] Support generating WebGPU compatible dynamic axes (using strict variables)
- [x] Verify maximum texture dimension limits for GEMM A/B matrices
- [x] Optimize scalar additions (fuse into `MatMul` beta where possible)
- [x] Prevent creation of heavily nested subgraphs (WebGPU prefers flattened execution)
- [x] Guarantee no usage of `If` or `Loop` anywhere in the transpiled tree structures
- [x] Strip ONNX metadata to compress `.onnx` payload size for network transfer (<1MB)
- [x] Serialize `Constant` arrays cleanly using little-endian standard
- [x] Prevent Out-of-Memory (OOM) on Pyodide by aggressively garbage collecting intermediate trees
- [x] Minimize peak RAM during the compilation phase
- [x] Support async loading hooks for massive constant arrays natively
- [x] Support INT8 quantization of GEMM matrices to halve WebGPU buffer sizes
- [x] Support FP16 downcasting of GEMM matrices natively
- [x] Ensure WGSL shader compatibility by avoiding `Float64` across the entire graph
- [x] Ensure WGSL shader compatibility by casting `Int64` to `Int32` natively
- [x] Test tree ensemble transpilation directly inside Chrome/V8
- [x] Test tree ensemble transpilation directly inside Safari/JavaScriptCore
- [x] Validate multi-threading (SharedArrayBuffer) concurrency with generated GEMM graphs
- [x] Evaluate cold-start latency of GEMM graph on AWS Lambda
- [x] Optimize node topology specifically for `onnxruntime-web` execution providers
- [x] Map tree structures to explicitly parallelized sub-graphs if hardware supports it
- [x] Pre-evaluate static shapes using `GraphSurgeon` tools automatically
- [x] Run constant folding automatically on transpiled graphs
- [x] Run dead-code elimination automatically on transpiled graphs

### 12. Testing, Validation & Edge Cases (25+ items)

- [x] Unit Test: 1-tree DecisionTreeClassifier
- [x] Unit Test: 100-tree RandomForestClassifier (binary)
- [x] Unit Test: 100-tree RandomForestClassifier (multiclass)
- [x] Unit Test: 100-tree RandomForestRegressor
- [x] Unit Test: LightGBM GBDT (1000 trees)
- [x] Unit Test: LightGBM DART (100 trees)
- [x] Unit Test: XGBoost gblinear
- [x] Unit Test: XGBoost gbtree (binary:logistic)
- [x] Unit Test: XGBoost gbtree (multi:softprob)
- [x] Unit Test: CatBoost (symmetric trees)
- [x] Unit Test: IsolationForest anomaly detection
- [x] Unit Test: Empty tree structure handling
- [x] Unit Test: Trees with depth > 50 (GEMM strategy fallback)
- [x] Unit Test: Trees with perfectly balanced properties
- [x] Test output equivalency with Scikit-Learn `predict` (atol=1e-5)
- [x] Test output equivalency with Scikit-Learn `predict_proba` (atol=1e-5)
- [x] Test output equivalency with LightGBM (atol=1e-5)
- [x] Test output equivalency with XGBoost (atol=1e-5)
- [x] Test output equivalency with `onnxruntime` native `ai.onnx.ml` providers
- [x] Stress Test: 10,000 tree Random Forest (compilation time < 2 seconds)
- [x] Stress Test: 10,000 tree Random Forest (WASM execution time < 10ms)
- [x] Handle identically named features in input datasets securely
- [x] Parse completely collinear features cleanly
- [x] Handle deeply imbalanced multi-class trees without NaNs
- [x] Prevent integer overflow during PerfectTree depth capacity calculations
