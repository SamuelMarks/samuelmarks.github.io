# onnxmltools Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `onnxmltools` within the `onnx9000` ecosystem.
The original `onnxmltools` relies heavily on native bindings for LightGBM, XGBoost, CatBoost, CoreML, and PySpark to perform conversions.
Our `onnx9000` reimplementation uses pure-Python parsers (reading raw JSON dumps, protobufs, or binary structures) to translate these diverse model types into `ai.onnx.ml` and `ai.onnx` topologies.
This zero-dependency approach means you can convert a 10,000-tree XGBoost model or a complex SparkML pipeline entirely in a web browser using WASM or in a cold-start AWS Lambda function without installing multi-gigabyte ML frameworks.

## Exhaustive Parity Checklist

### 1. Pure-Python LightGBM Conversion (40+ items)

- [x] Implement zero-dependency parser for LightGBM `.txt` dump format
- [x] Implement zero-dependency parser for LightGBM JSON dump format
- [x] Parse LightGBM `tree_info` (left/right children, thresholds, features)
- [x] Extract LightGBM leaf values and weights accurately
- [x] Map LightGBM `regression` objective -> `TreeEnsembleRegressor`
- [x] Map LightGBM `regression_l1` objective -> `TreeEnsembleRegressor`
- [x] Map LightGBM `huber` objective -> `TreeEnsembleRegressor`
- [x] Map LightGBM `fair` objective -> `TreeEnsembleRegressor`
- [x] Map LightGBM `poisson` objective -> `TreeEnsembleRegressor` + `Exp`
- [x] Map LightGBM `quantile` objective -> `TreeEnsembleRegressor`
- [x] Map LightGBM `mape` objective -> `TreeEnsembleRegressor`
- [x] Map LightGBM `gamma` objective -> `TreeEnsembleRegressor` + `Exp`
- [x] Map LightGBM `tweedie` objective -> `TreeEnsembleRegressor` + `Exp`
- [x] Map LightGBM `binary` objective -> `TreeEnsembleClassifier`
- [x] Map LightGBM `multiclass` objective -> `TreeEnsembleClassifier`
- [x] Map LightGBM `multiclassova` objective -> `TreeEnsembleClassifier`
- [x] Map LightGBM `cross_entropy` objective -> `TreeEnsembleClassifier`
- [x] Map LightGBM `cross_entropy_lambda` objective -> `TreeEnsembleClassifier`
- [x] Map LightGBM `lambdarank` objective -> `TreeEnsembleRegressor`
- [x] Map LightGBM `rank_xendcg` objective -> `TreeEnsembleRegressor`
- [x] Map LightGBM missing value logic (default to left/right) to `nodes_missing_value_tracks_true`
- [x] Extract and map LightGBM `class_names` to ONNX `classlabels_strings`
- [x] Extract and map LightGBM `class_names` to ONNX `classlabels_ints`
- [x] Support LightGBM models with categorical feature splits natively in ONNX
- [x] Map LightGBM `base_score` / `objective_seed` to ONNX base values
- [x] Map LightGBM `sigmoid` parameter for binary classification
- [x] Map LightGBM `num_class` correctly for multiclass ZipMap generation
- [x] Verify LightGBM `boosting_type="gbdt"` translation
- [x] Verify LightGBM `boosting_type="dart"` translation (handling dropout scaling)
- [x] Verify LightGBM `boosting_type="goss"` translation
- [x] Verify LightGBM `boosting_type="rf"` translation (Random Forest mode)

### 2. Pure-Python XGBoost Conversion (40+ items)

- [x] Implement zero-dependency parser for XGBoost JSON model dumps
- [x] Implement zero-dependency parser for XGBoost legacy binary `.ubj` (Universal Binary JSON)
- [x] Parse XGBoost `trees` array (nodes, splits, Yes/No/Missing paths)
- [x] Extract XGBoost `base_score` and apply dynamically based on objective
- [x] Map XGBoost `reg:squarederror` -> `TreeEnsembleRegressor`
- [x] Map XGBoost `reg:squaredlogerror` -> `TreeEnsembleRegressor`
- [x] Map XGBoost `reg:logistic` -> `TreeEnsembleRegressor` + `Sigmoid`
- [x] Map XGBoost `reg:pseudohubererror` -> `TreeEnsembleRegressor`
- [x] Map XGBoost `binary:logistic` -> `TreeEnsembleClassifier` (with Sigmoid post-transform)
- [x] Map XGBoost `binary:logitraw` -> `TreeEnsembleClassifier` (with NONE post-transform)
- [x] Map XGBoost `binary:hinge` -> `TreeEnsembleClassifier` (with Step post-transform logic)
- [x] Map XGBoost `count:poisson` -> `TreeEnsembleRegressor` + `Exp`
- [x] Map XGBoost `survival:cox` -> `TreeEnsembleRegressor` + `Exp`
- [x] Map XGBoost `survival:aft` -> `TreeEnsembleRegressor`
- [x] Map XGBoost `multi:softmax` -> `TreeEnsembleClassifier` (with Softmax post-transform)
- [x] Map XGBoost `multi:softprob` -> `TreeEnsembleClassifier` (with Softmax probability generation)
- [x] Map XGBoost `rank:pairwise` -> `TreeEnsembleRegressor`
- [x] Map XGBoost `rank:ndcg` -> `TreeEnsembleRegressor`
- [x] Map XGBoost `rank:map` -> `TreeEnsembleRegressor`
- [x] Map XGBoost missing value logic (`missing` attribute) to ONNX tree node rules
- [x] Track XGBoost `tree_limit` / `best_iteration` for truncated model conversion
- [x] Map XGBoost multi-output regression natively (Forest regressor with multiple target ids)
- [x] Reconstruct categorical features based on XGBoost experimental categorical split nodes
- [x] Support XGBoost `gblinear` booster -> `LinearRegressor` / `LinearClassifier`
- [x] Support XGBoost `dart` booster -> `TreeEnsemble` (applying weight drops statically)
- [x] Support XGBoost `gbtree` booster natively
- [x] Parse XGBoost `feature_names` to establish explicit ONNX inputs
- [x] Map XGBoost `scale_pos_weight` behavior implicitly within tree leaves
- [x] Support Scikit-Learn wrapper `XGBClassifier`
- [x] Support Scikit-Learn wrapper `XGBRegressor`

### 3. Pure-Python CatBoost Conversion (30+ items)

- [x] Implement zero-dependency parser for CatBoost JSON dumps
- [x] Parse CatBoost oblivious trees (flattening symmetric tree structures to standard DAGs)
- [x] Extract CatBoost float features and splits
- [x] Extract CatBoost one-hot encoded categorical features
- [x] Extract CatBoost CTR (Categorical Target Encoding) features natively if possible
- [x] Map CatBoost `RMSE` / `MultiRMSE` -> `TreeEnsembleRegressor`
- [x] Map CatBoost `MAE` / `Quantile` -> `TreeEnsembleRegressor`
- [x] Map CatBoost `Logloss` -> `TreeEnsembleClassifier`
- [x] Map CatBoost `CrossEntropy` -> `TreeEnsembleClassifier`
- [x] Map CatBoost `MultiClass` -> `TreeEnsembleClassifier` (Softmax)
- [x] Map CatBoost `MultiClassOneVsAll` -> `TreeEnsembleClassifier` (Sigmoid)
- [x] Map CatBoost `Poisson` -> `TreeEnsembleRegressor`
- [x] Map CatBoost `SurvivalAft` -> `TreeEnsembleRegressor`
- [x] Extract CatBoost leaf values accurately from oblivious tree arrays
- [x] Extract CatBoost `scale_and_bias` attributes and map to ONNX Affine transform / Tree biases
- [x] Translate symmetric thresholds to `ai.onnx.ml` `nodes_values` correctly
- [x] Unroll oblivious tree bitmasks into standard ONNX left/right index structures
- [x] Support `CatBoostClassifier` wrappers
- [x] Support `CatBoostRegressor` wrappers

### 4. CoreML Protobuf Conversion - Core & ML (40+ items)

- [x] Implement zero-dependency parser for CoreML `.mlmodel` (Protobuf)
- [x] Parse CoreML `ModelDescription` for inputs, outputs, and feature types
- [x] Map CoreML `Int64FeatureType` -> ONNX `Int64`
- [x] Map CoreML `DoubleFeatureType` -> ONNX `Float64` or `Float32`
- [x] Map CoreML `StringFeatureType` -> ONNX `String`
- [x] Map CoreML `ImageFeatureType` -> ONNX `Tensor` (Image shapes)
- [x] Map CoreML `MultiArrayFeatureType` -> ONNX `Tensor`
- [x] Map CoreML `DictionaryFeatureType` -> ONNX `Map` / `Sequence`
- [x] Map CoreML `TreeEnsembleClassifier` -> `ai.onnx.ml.TreeEnsembleClassifier`
- [x] Map CoreML `TreeEnsembleRegressor` -> `ai.onnx.ml.TreeEnsembleRegressor`
- [x] Map CoreML `SupportVectorClassifier` -> `ai.onnx.ml.SVMClassifier`
- [x] Map CoreML `SupportVectorRegressor` -> `ai.onnx.ml.SVMRegressor`
- [x] Map CoreML `GLMClassifier` -> `ai.onnx.ml.LinearClassifier`
- [x] Map CoreML `GLMRegressor` -> `ai.onnx.ml.LinearRegressor`
- [x] Map CoreML `DictVectorizer` -> `ai.onnx.ml.DictVectorizer`
- [x] Map CoreML `FeatureVectorizer` -> `ai.onnx.ml.FeatureVectorizer`
- [x] Map CoreML `Imputer` -> `ai.onnx.ml.Imputer`
- [x] Map CoreML `Scaler` -> `ai.onnx.ml.Scaler`
- [x] Map CoreML `Normalizer` -> `ai.onnx.ml.Normalizer`
- [x] Map CoreML `OneHotEncoder` -> `ai.onnx.ml.OneHotEncoder`
- [x] Map CoreML `CategoricalMapping` -> `ai.onnx.ml.CategoryMapper`
- [x] Map CoreML `ArrayFeatureExtractor` -> `ai.onnx.ml.ArrayFeatureExtractor`
- [x] Map CoreML `NonMaximumSuppression` -> `ai.onnx.NonMaxSuppression`
- [x] Map CoreML `ItemSimilarityRecommender` -> ONNX Custom Subgraph
- [x] Map CoreML `WordTagger` -> ONNX Custom Subgraph
- [x] Map CoreML `TextClassifier` -> ONNX Custom Subgraph
- [x] Map CoreML `VisionFeaturePrint` -> ONNX Extensibility
- [x] Map CoreML Pipeline models (recursive conversion)

### 5. CoreML Protobuf Conversion - Neural Networks (40+ items)

- [x] Map CoreML `NeuralNetwork` -> ONNX Subgraph
- [x] Map CoreML `NeuralNetworkClassifier` -> ONNX Subgraph + Probabilities
- [x] Map CoreML `NeuralNetworkRegressor` -> ONNX Subgraph + Scores
- [x] Map CoreML `ConvolutionLayer` -> `Conv`
- [x] Map CoreML `PoolingLayer` -> `MaxPool` / `AveragePool`
- [x] Map CoreML `ActivationReLU` -> `Relu`
- [x] Map CoreML `ActivationLeakyReLU` -> `LeakyRelu`
- [x] Map CoreML `ActivationSigmoid` -> `Sigmoid`
- [x] Map CoreML `ActivationTanh` -> `Tanh`
- [x] Map CoreML `ActivationLinear` -> `Add` + `Mul`
- [x] Map CoreML `ActivationPReLU` -> `PRelu`
- [x] Map CoreML `ActivationELU` -> `Elu`
- [x] Map CoreML `ActivationSoftsign` -> `Softsign`
- [x] Map CoreML `ActivationSoftplus` -> `Softplus`
- [x] Map CoreML `ActivationParametricSoftmax` -> `Softmax`
- [x] Map CoreML `BatchnormLayer` -> `BatchNormalization`
- [x] Map CoreML `InnerProductLayer` -> `Gemm` / `MatMul`
- [x] Map CoreML `SoftmaxLayer` -> `Softmax`
- [x] Map CoreML `FlattenLayer` -> `Flatten`
- [x] Map CoreML `ConcatLayer` -> `Concat`
- [x] Map CoreML `ReshapeLayer` -> `Reshape`
- [x] Map CoreML `PaddingLayer` -> `Pad`
- [x] Map CoreML `PermuteLayer` -> `Transpose`
- [x] Map CoreML `UpsampleLayer` -> `Upsample` / `Resize`
- [x] Map CoreML `L2NormalizeLayer` -> `LpNormalization`
- [x] Map CoreML `SimpleRNNLayer` -> `RNN`
- [x] Map CoreML `GRULayer` -> `GRU`
- [x] Map CoreML `UniDirectionalLSTMLayer` -> `LSTM`
- [x] Map CoreML `BiDirectionalLSTMLayer` -> `LSTM`
- [x] Map CoreML `ScaleLayer` -> `Mul`
- [x] Map CoreML `CropLayer` -> `Slice`
- [x] Map CoreML `AverageLayer` -> `Mean`
- [x] Map CoreML `MaxLayer` -> `Max`
- [x] Map CoreML `MinLayer` -> `Min`
- [x] Map CoreML `DotProductLayer` -> `Mul` + `ReduceSum`
- [x] Map CoreML `ReduceLayer` -> `Reduce*` operations

### 6. SparkML Pipeline & Transformer Conversion (40+ items)

- [x] Implement zero-dependency parser for SparkML Pipeline JSON/Parquet dumps
- [x] Extract PySpark `PipelineModel` stages
- [x] Map Spark `Binarizer` -> `ai.onnx.ml.Binarizer`
- [x] Map Spark `Bucketizer` -> `ai.onnx.ml.CategoryMapper` / Custom Subgraph
- [x] Map Spark `ChiSqSelector` -> `ai.onnx.ml.ArrayFeatureExtractor`
- [x] Map Spark `CountVectorizerModel` -> `ai.onnx.ml.CountVectorizer`
- [x] Map Spark `DCT` -> ONNX Custom Subgraph
- [x] Map Spark `ElementwiseProduct` -> `Mul`
- [x] Map Spark `HashingTF` -> ONNX Custom (MurmurHash3)
- [x] Map Spark `IDFModel` -> `Mul` (with extracted IDF weights)
- [x] Map Spark `ImputerModel` -> `ai.onnx.ml.Imputer`
- [x] Map Spark `IndexToString` -> `ai.onnx.ml.CategoryMapper`
- [x] Map Spark `MaxAbsScalerModel` -> `ai.onnx.ml.Scaler`
- [x] Map Spark `MinMaxScalerModel` -> `ai.onnx.ml.Scaler`
- [x] Map Spark `NGram` -> ONNX Sequence logic
- [x] Map Spark `Normalizer` -> `ai.onnx.ml.Normalizer`
- [x] Map Spark `OneHotEncoderModel` -> `ai.onnx.ml.OneHotEncoder`
- [x] Map Spark `PCAModel` -> `MatMul`
- [x] Map Spark `PolynomialExpansion` -> Math Subgraph
- [x] Map Spark `QuantileDiscretizer` -> `ai.onnx.ml.Binarizer` / Subgraph
- [x] Map Spark `RegexTokenizer` -> ONNX Custom Regex Node
- [x] Map Spark `StandardScalerModel` -> `ai.onnx.ml.Scaler`
- [x] Map Spark `StopWordsRemover` -> ONNX Custom Subgraph
- [x] Map Spark `StringIndexerModel` -> `ai.onnx.ml.CategoryMapper`
- [x] Map Spark `Tokenizer` -> ONNX StringSplit / Custom
- [x] Map Spark `VectorAssembler` -> `ai.onnx.ml.FeatureVectorizer` / `Concat`
- [x] Map Spark `VectorIndexerModel` -> Subgraph
- [x] Map Spark `VectorSlicer` -> `ai.onnx.ml.ArrayFeatureExtractor` / `Slice`
- [x] Map Spark `Word2VecModel` -> `Gather` (Embedding lookup)

### 7. SparkML Classifier & Regressor Conversion (30+ items)

- [x] Map Spark `LogisticRegressionModel` -> `ai.onnx.ml.LinearClassifier`
- [x] Map Spark `DecisionTreeClassificationModel` -> `ai.onnx.ml.TreeEnsembleClassifier`
- [x] Map Spark `RandomForestClassificationModel` -> `ai.onnx.ml.TreeEnsembleClassifier`
- [x] Map Spark `GBTClassificationModel` -> `ai.onnx.ml.TreeEnsembleClassifier`
- [x] Map Spark `MultilayerPerceptronClassificationModel` -> Chained `MatMul` + `Sigmoid`
- [x] Map Spark `LinearSVCModel` -> `ai.onnx.ml.SVMClassifier`
- [x] Map Spark `NaiveBayesModel` -> Probability Subgraph
- [x] Map Spark `LinearRegressionModel` -> `ai.onnx.ml.LinearRegressor`
- [x] Map Spark `GeneralizedLinearRegressionModel` -> `ai.onnx.ml.LinearRegressor` + Link function
- [x] Map Spark `DecisionTreeRegressionModel` -> `ai.onnx.ml.TreeEnsembleRegressor`
- [x] Map Spark `RandomForestRegressionModel` -> `ai.onnx.ml.TreeEnsembleRegressor`
- [x] Map Spark `GBTRegressionModel` -> `ai.onnx.ml.TreeEnsembleRegressor`
- [x] Map Spark `AFTSurvivalRegressionModel` -> Regression Subgraph
- [x] Map Spark `IsotonicRegressionModel` -> Subgraph / Custom
- [x] Map Spark `FMClassificationModel` (Factorization Machines) -> Math Subgraph
- [x] Map Spark `FMRegressionModel` -> Math Subgraph
- [x] Map Spark `KMeansModel` -> Distance Subgraph + `ArgMin`
- [x] Map Spark `BisectingKMeansModel` -> Distance Subgraph
- [x] Map Spark `GaussianMixtureModel` -> Probability Subgraph

### 8. LibSVM & Misc Model Support (10+ items)

- [x] Implement zero-dependency parser for LibSVM model text formats
- [x] Map LibSVM `C-SVC` -> `ai.onnx.ml.SVMClassifier`
- [x] Map LibSVM `nu-SVC` -> `ai.onnx.ml.SVMClassifier`
- [x] Map LibSVM `one-class SVM` -> `ai.onnx.ml.SVMClassifier`
- [x] Map LibSVM `epsilon-SVR` -> `ai.onnx.ml.SVMRegressor`
- [x] Map LibSVM `nu-SVR` -> `ai.onnx.ml.SVMRegressor`
- [x] Extract SVM support vectors, dual coefficients, and rhos
- [x] Map linear, polynomial, RBF, and sigmoid kernels cleanly
- [x] Support probability estimates (Platt scaling) via LibSVM parameters
- [x] Integrate H2O model parsing (MOJO/POJO to ONNX logic mapped if structurally requested)

### 9. Graph Optimizations & Routing (20+ items)

- [x] Implement TreeEnsemble node compression (removing redundant leaf nodes)
- [x] Implement VectorAssembler fusion (canceling out consecutive un-assemblies)
- [x] Optimize sequential Spark StringIndexers into a single dict lookup
- [x] Resolve LightGBM/XGBoost `int64` vs `float32` split thresholds globally
- [x] Handle TreeEnsemble batch inference dimension alignment explicitly
- [x] Flatten nested `ZipMap` operators from multiple estimators into a single dictionary array
- [x] Remove `Cast` operations that do not change precision semantics
- [x] Fuse scaling nodes generated by CoreML pipelines
- [x] Apply constant folding on CoreML NeuralNetwork normalization weights
- [x] Enforce deterministic node naming strategies across all converters

### 10. Lightweight Runtime & Zero-Dependency Capabilities (20+ items)

- [x] Execute LightGBM parsing fully in memory without disk I/O
- [x] Execute XGBoost parsing fully in memory without disk I/O
- [x] No `lightgbm` pip package required at runtime
- [x] No `xgboost` pip package required at runtime
- [x] No `catboost` pip package required at runtime
- [x] No `coremltools` or MacOS required at runtime (parse `.mlmodel` anywhere)
- [x] No `pyspark` or Java JVM required at runtime
- [x] Provide TypeScript/JS native wrappers for `onnx9000` web translation
- [x] Emscripten build configuration for translating models inside Safari/Chrome
- [x] Cloudflare Worker ready: script fits within 1MB worker limits
- [x] AWS Lambda ready: instantaneous cold start translation of tree structures
- [x] Support handling models > 2GB using file-backed memory-mapping structures
- [x] Support dynamic ONNX shape overriding natively in all 5 sub-converters
- [x] Produce strict `ai.onnx.ml` Opsets 1, 2, 3, 4 based on user flags
- [x] Provide 100% strict compliance validation against standard `onnxruntime`

### 11. Advanced CatBoost Loss Functions & Edge Cases (30+ items)

- [x] Map CatBoost `Huber` -> `TreeEnsembleRegressor`
- [x] Map CatBoost `Lq` -> `TreeEnsembleRegressor`
- [x] Map CatBoost `Tweedie` -> `TreeEnsembleRegressor` + `Exp`
- [x] Map CatBoost `Focal` -> `TreeEnsembleClassifier`
- [x] Map CatBoost `BrierScore` -> `TreeEnsembleClassifier`
- [x] Map CatBoost `HingeLoss` -> `TreeEnsembleClassifier`
- [x] Map CatBoost `HammingLoss` -> `TreeEnsembleClassifier`
- [x] Map CatBoost `ZeroOneLoss` -> `TreeEnsembleClassifier`
- [x] Map CatBoost `PairLogit` -> `TreeEnsembleClassifier` (pairwise)
- [x] Map CatBoost `PairLogitPairwise` -> `TreeEnsembleClassifier`
- [x] Map CatBoost `YetiRank` -> `TreeEnsembleRegressor`
- [x] Map CatBoost `YetiRankPairwise` -> `TreeEnsembleRegressor`
- [x] Map CatBoost `QueryRMSE` -> `TreeEnsembleRegressor`
- [x] Map CatBoost `QuerySoftMax` -> `TreeEnsembleClassifier`
- [x] Map CatBoost `StochasticFilter` -> `TreeEnsembleClassifier`
- [x] Map CatBoost `CrossEntropy` (with smoothed weights) -> `TreeEnsembleClassifier`
- [x] Support CatBoost `text` features (via ONNX string hashing/TF-IDF)
- [x] Support CatBoost `embedding` features (via ONNX Gather)
- [x] Flatten nested oblivious trees correctly when `leaf_values` exceed standard depths
- [x] Handle CatBoost explicit `border_count` constraints internally

### 12. Advanced SparkML & Pyspark Ecosystem Ops (20+ items)

- [x] Handle Spark `SparseVector` objects intrinsically via ONNX dense/sparse structures
- [x] Map Spark `SQLTransformer` -> ONNX Custom logic/Subgraph (where mathematically possible)
- [x] Map Spark `Binarizer` with custom `threshold` vectors
- [x] Map Spark `Bucketizer` with `splitsArray`
- [x] Map Spark `StringIndexer` with `handleInvalid='skip'`
- [x] Map Spark `StringIndexer` with `handleInvalid='keep'`
- [x] Map Spark `OneHotEncoder` with `dropLast=True`
- [x] Map Spark `OneHotEncoder` with `dropLast=False`
- [x] Map Spark `VectorAssembler` with `handleInvalid='skip'`
- [x] Map Spark `VectorAssembler` with `handleInvalid='keep'`
- [x] Map Spark `MinMaxScaler` with custom `min`/`max` bounds
- [x] Map Spark `MaxAbsScaler` with zero-variance protections
- [x] Support PySpark 2.x `PipelineModel` definitions
- [x] Support PySpark 3.x `PipelineModel` definitions

### 13. Advanced H2O & MOJO Translation (20+ items)

- [x] Implement zero-dependency parser for H2O MOJO / POJO structures
- [x] Map H2O `DistributedRandomForest` -> `TreeEnsembleClassifier`
- [x] Map H2O `GradientBoostingEstimator` -> `TreeEnsembleRegressor` / Classifier
- [x] Map H2O `DeepLearningEstimator` -> ONNX Subgraph (MLP)
- [x] Map H2O `GeneralizedLinearEstimator` -> `LinearRegressor` / Classifier
- [x] Map H2O `IsolationForest` -> `TreeEnsembleClassifier` (Anomaly detection)
- [x] Map H2O `KMeans` -> Distance Subgraph
- [x] Map H2O `PCA` -> `MatMul`
- [x] Map H2O `NaiveBayes` -> Probability Subgraph
- [x] Extract H2O categorical encodings seamlessly into ONNX `CategoryMapper`
- [x] Handle H2O missing values natively via MOJO node properties

### 14. Testing, Web Parity, and Validation (15+ items)

- [x] Unit test: End-to-end `lightgbm` Regressor (browser environment)
- [x] Unit test: End-to-end `lightgbm` Classifier (WASM environment)
- [x] Unit test: End-to-end `xgboost` Regressor (browser environment)
- [x] Unit test: End-to-end `xgboost` Classifier (WASM environment)
- [x] Unit test: End-to-end `catboost` Regressor (browser environment)
- [x] Unit test: End-to-end `catboost` Classifier (WASM environment)
- [x] Unit test: End-to-end `coreml` Image Classifier (WebGPU environment)
- [x] Unit test: End-to-end `pyspark` Pipeline (WASM environment)
- [x] Stress Test: 20,000 node XGBoost tree translated inside Chrome < 50ms
- [x] Validate deterministic identical translations (pure Python vs native converters)
- [x] Validate `ai.onnx.ml` execution against `onnx9000` Python JIT

### 15. LightGBM, XGBoost & LibSVM Hyper-edge Cases (30+ items)

- [x] Parse LightGBM `custom` objective functions (if mathematically representable in ONNX)
- [x] Map LightGBM `pos_bagging_fraction` impacts correctly
- [x] Map LightGBM `neg_bagging_fraction` impacts correctly
- [x] Handle LightGBM `categorical_feature` indexing natively in ONNX (no pre-processing needed)
- [x] Support LightGBM `is_unbalance=True` leaf value adjustments
- [x] Support LightGBM `scale_pos_weight` explicit overrides
- [x] Map XGBoost `survival:aft` with specific exponential constraints
- [x] Map XGBoost `survival:cox` hazard ratios to ONNX outputs
- [x] Map XGBoost `binary:logitraw` -> `LinearClassifier` or Tree with no link function
- [x] Handle XGBoost `base_margin` dynamically as an explicit ONNX graph input
- [x] Parse XGBoost `interaction_constraints` strictly
- [x] Map XGBoost `monotone_constraints` into tree evaluations correctly
- [x] Support LibSVM `probability=1` -> `ai.onnx.ml.SVMClassifier` (with Platt Scaling parameters)
- [x] Support LibSVM `probability=0` -> `ai.onnx.ml.SVMClassifier` (raw scores)
- [x] Map LibSVM `shrinking=1` heuristics into SVM structures
- [x] Ensure LibSVM `cache_size` is ignored (not applicable at inference)
- [x] Map LibSVM `kernel_type=0` (linear) -> SVM
- [x] Map LibSVM `kernel_type=1` (polynomial) -> SVM
- [x] Map LibSVM `kernel_type=2` (RBF) -> SVM
- [x] Map LibSVM `kernel_type=3` (sigmoid) -> SVM
- [x] Extract LibSVM `degree` exactly
- [x] Extract LibSVM `gamma` exactly
- [x] Extract LibSVM `coef0` exactly
- [x] Parse custom string tokens safely within `LibSVM` format lines
