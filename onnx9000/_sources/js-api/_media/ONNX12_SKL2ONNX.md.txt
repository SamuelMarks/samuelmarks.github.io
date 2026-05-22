# skl2onnx Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `skl2onnx` within the `onnx9000` ecosystem.
Unlike the original project, our implementation focuses heavily on translating traditional machine learning models (Scikit-Learn) into highly optimized, zero-dependency `ai.onnx.ml` structures.
By generating these models via pure Python, the resulting pipelines can be executed instantly in the browser natively using WASM without needing heavy Python runtimes or C++ ML libraries, and they can be dispatched instantly in high-concurrency server environments.

## Exhaustive Parity Checklist

### 1. Core Architecture & Pipeline Parsing (Zero-Dependency)

- [x] Implement zero-dependency Scikit-Learn object introspection logic
- [x] Traverse `Pipeline` objects and topologically sort steps
- [x] Traverse `FeatureUnion` objects and map to parallel execution graphs
- [x] Traverse `ColumnTransformer` and map to `ArrayFeatureExtractor` / `Concat`
- [x] Traverse nested meta-estimators (e.g., `GridSearchCV`, `RandomizedSearchCV`)
- [x] Auto-infer input tensor shapes and types from `X` training arrays or Pandas DataFrames
- [x] Support explicit user-defined initial types (e.g., `FloatTensorType`, `Int64TensorType`)
- [x] Support explicit user-defined string types (`StringTensorType`)
- [x] Map Scikit-Learn `predict` methods to ONNX graph outputs
- [x] Map Scikit-Learn `predict_proba` methods to `ZipMap` probabilities
- [x] Map Scikit-Learn `transform` methods to intermediate tensor representations
- [x] Map Scikit-Learn `decision_function` to raw score outputs
- [x] Handle missing value imputation (`np.nan` -> ONNX scalar handling)
- [x] Handle sparse matrix inputs (SciPy sparse matrices to ONNX dense/sparse translations)

### 2. Preprocessing Transformers & Scaling (30+ items)

- [x] Map `StandardScaler` -> `Scaler`
- [x] Map `MinMaxScaler` -> `Scaler`
- [x] Map `MaxAbsScaler` -> `Scaler`
- [x] Map `RobustScaler` -> `Scaler` (with median/IQR params)
- [x] Map `Normalizer` -> `Normalizer` (L1, L2, Max norms)
- [x] Map `Binarizer` -> `Binarizer`
- [x] Map `PolynomialFeatures` -> ONNX Math Subgraph
- [x] Map `PowerTransformer` (Yeo-Johnson) -> ONNX Math Subgraph
- [x] Map `PowerTransformer` (Box-Cox) -> ONNX Math Subgraph
- [x] Map `QuantileTransformer` -> ONNX Math Subgraph (Interpolation)
- [x] Map `KBinsDiscretizer` (ordinal) -> `Binarizer` / Custom
- [x] Map `KBinsDiscretizer` (onehot) -> `OneHotEncoder`
- [x] Map `OneHotEncoder` -> `OneHotEncoder` (handling handle_unknown)
- [x] Map `OrdinalEncoder` -> `CategoryMapper`
- [x] Map `LabelEncoder` -> `CategoryMapper`
- [x] Map `LabelBinarizer` -> `Binarizer` or `CategoryMapper`
- [x] Map `MultiLabelBinarizer` -> ONNX Sequence / Math
- [x] Map `SimpleImputer` -> `Imputer` (mean, median, most_frequent, constant)
- [x] Map `MissingIndicator` -> `IsNaN` / `IsInf`
- [x] Map `IterativeImputer` -> Subgraph of Estimators
- [x] Map `KNNImputer` -> Subgraph of KNN
- [x] Map `FunctionTransformer` -> ONNX Math / Extensibility hooks
- [x] Map `SplineTransformer` -> ONNX Math Subgraph
- [x] Map `DictVectorizer` -> `DictVectorizer`
- [x] Map `FeatureHasher` -> ONNX Custom (MurmurHash mapping)
- [x] Map `CountVectorizer` -> `CountVectorizer` (vocabulary mapping)
- [x] Map `TfidfTransformer` -> `TfIdfVectorizer` or Math ops
- [x] Map `TfidfVectorizer` -> `TfIdfVectorizer`
- [x] Map `HashingVectorizer` -> ONNX Custom Hash

### 3. Linear Models & Classifiers (30+ items)

- [x] Map `LinearRegression` -> `LinearRegressor`
- [x] Map `Ridge` -> `LinearRegressor`
- [x] Map `RidgeCV` -> `LinearRegressor`
- [x] Map `Lasso` -> `LinearRegressor`
- [x] Map `LassoCV` -> `LinearRegressor`
- [x] Map `ElasticNet` -> `LinearRegressor`
- [x] Map `ElasticNetCV` -> `LinearRegressor`
- [x] Map `Lars` -> `LinearRegressor`
- [x] Map `LassoLars` -> `LinearRegressor`
- [x] Map `OrthogonalMatchingPursuit` -> `LinearRegressor`
- [x] Map `BayesianRidge` -> `LinearRegressor`
- [x] Map `ARDRegression` -> `LinearRegressor`
- [x] Map `LogisticRegression` -> `LinearClassifier` (binary)
- [x] Map `LogisticRegression` -> `LinearClassifier` (multiclass / OvR)
- [x] Map `LogisticRegression` -> `LinearClassifier` (multinomial / softmax)
- [x] Map `LogisticRegressionCV` -> `LinearClassifier`
- [x] Map `PassiveAggressiveClassifier` -> `LinearClassifier`
- [x] Map `PassiveAggressiveRegressor` -> `LinearRegressor`
- [x] Map `Perceptron` -> `LinearClassifier`
- [x] Map `RidgeClassifier` -> `LinearClassifier`
- [x] Map `RidgeClassifierCV` -> `LinearClassifier`
- [x] Map `SGDClassifier` -> `LinearClassifier`
- [x] Map `SGDRegressor` -> `LinearRegressor`
- [x] Map `HuberRegressor` -> `LinearRegressor`
- [x] Map `TheilSenRegressor` -> `LinearRegressor`
- [x] Map `QuantileRegressor` -> `LinearRegressor`
- [x] Map `PoissonRegressor` -> `LinearRegressor` + Exp Link
- [x] Map `GammaRegressor` -> `LinearRegressor` + Exp Link
- [x] Map `TweedieRegressor` -> `LinearRegressor` + Link

### 4. Support Vector Machines (SVM) (20+ items)

- [x] Map `SVC` (linear kernel) -> `SVMClassifier`
- [x] Map `SVC` (poly kernel) -> `SVMClassifier`
- [x] Map `SVC` (rbf kernel) -> `SVMClassifier`
- [x] Map `SVC` (sigmoid kernel) -> `SVMClassifier`
- [x] Map `SVR` (linear kernel) -> `SVMRegressor`
- [x] Map `SVR` (poly kernel) -> `SVMRegressor`
- [x] Map `SVR` (rbf kernel) -> `SVMRegressor`
- [x] Map `SVR` (sigmoid kernel) -> `SVMRegressor`
- [x] Map `NuSVC` -> `SVMClassifier`
- [x] Map `NuSVR` -> `SVMRegressor`
- [x] Map `OneClassSVM` -> `SVMClassifier` (Anomaly detection)
- [x] Map `LinearSVC` -> `LinearClassifier`
- [x] Map `LinearSVR` -> `LinearRegressor`
- [x] Extract dual coefficients (`dual_coef_`) accurately
- [x] Extract support vectors (`support_vectors_`) accurately
- [x] Extract intercept (`intercept_`) accurately
- [x] Map probability estimates (`predict_proba`) via Platt scaling logic natively in graph
- [x] Support multi-class SVM (`ovo` strategy) via ONNX topology mappings
- [x] Support multi-class SVM (`ovr` strategy) via ONNX topology mappings

### 5. Tree & Ensemble Methods (30+ items)

- [x] Map `DecisionTreeClassifier` -> `TreeEnsembleClassifier`
- [x] Map `DecisionTreeRegressor` -> `TreeEnsembleRegressor`
- [x] Map `ExtraTreeClassifier` -> `TreeEnsembleClassifier`
- [x] Map `ExtraTreeRegressor` -> `TreeEnsembleRegressor`
- [x] Map `RandomForestClassifier` -> `TreeEnsembleClassifier`
- [x] Map `RandomForestRegressor` -> `TreeEnsembleRegressor`
- [x] Map `ExtraTreesClassifier` -> `TreeEnsembleClassifier`
- [x] Map `ExtraTreesRegressor` -> `TreeEnsembleRegressor`
- [x] Map `GradientBoostingClassifier` -> `TreeEnsembleClassifier`
- [x] Map `GradientBoostingRegressor` -> `TreeEnsembleRegressor`
- [x] Map `HistGradientBoostingClassifier` -> `TreeEnsembleClassifier`
- [x] Map `HistGradientBoostingRegressor` -> `TreeEnsembleRegressor`
- [x] Map `AdaBoostClassifier` -> `TreeEnsembleClassifier` (SAMME/SAMME.R)
- [x] Map `AdaBoostRegressor` -> `TreeEnsembleRegressor`
- [x] Map `BaggingClassifier` -> Ensemble Subgraph
- [x] Map `BaggingRegressor` -> Ensemble Subgraph
- [x] Map `IsolationForest` -> `TreeEnsembleClassifier` (Anomaly detection)
- [x] Map `VotingClassifier` -> `TreeEnsembleClassifier` / Subgraph
- [x] Map `VotingRegressor` -> `TreeEnsembleRegressor` / Subgraph
- [x] Map `StackingClassifier` -> Full Pipeline Subgraph
- [x] Map `StackingRegressor` -> Full Pipeline Subgraph
- [x] Efficient tree traversal parsing (recursively parsing left/right child structures)
- [x] Extracting tree thresholds accurately
- [x] Extracting tree values (leaf node outputs)
- [x] Handling missing value branches in HistGradientBoosting
- [x] Convert `n_classes_` handling natively in `TreeEnsembleClassifier`
- [x] Correctly aggregate probabilities for Random Forests (mean of trees)
- [x] Correctly accumulate decision paths for predictions

### 6. Decomposition, PCA, and Clustering (30+ items)

- [x] Map `PCA` -> `MatMul` (components) + `Add` (mean)
- [x] Map `IncrementalPCA` -> `MatMul` + `Add`
- [x] Map `KernelPCA` (linear) -> Subgraph
- [x] Map `KernelPCA` (rbf) -> Subgraph
- [x] Map `KernelPCA` (poly) -> Subgraph
- [x] Map `TruncatedSVD` -> `MatMul`
- [x] Map `FastICA` -> `MatMul` + `Add`
- [x] Map `NMF` (Non-Negative Matrix Factorization) -> `MatMul`
- [x] Map `LatentDirichletAllocation` -> Subgraph
- [x] Map `KMeans` -> Distance Computation Subgraph + `ArgMin`
- [x] Map `MiniBatchKMeans` -> Distance Computation Subgraph + `ArgMin`
- [x] Map `BisectingKMeans` -> Distance Computation Subgraph + `ArgMin`
- [x] Map `DBSCAN` (Inference mode if possible, or Custom)
- [x] Map `OPTICS` (Inference mode if possible)
- [x] Map `MeanShift` (Inference mode if possible)
- [x] Map `SpectralClustering` (Inference mapping)
- [x] Map `AgglomerativeClustering` (Inference mapping)
- [x] Map `GaussianMixture` -> Subgraph (Log probabilities)
- [x] Map `BayesianGaussianMixture` -> Subgraph
- [x] Compute pairwise distances in ONNX (Euclidean)
- [x] Compute pairwise distances in ONNX (Manhattan/L1)
- [x] Compute pairwise distances in ONNX (Cosine)

### 7. Naive Bayes & Nearest Neighbors (20+ items)

- [x] Map `GaussianNB` -> Subgraph (probability densities)
- [x] Map `MultinomialNB` -> `LinearClassifier` or Subgraph
- [x] Map `ComplementNB` -> `LinearClassifier` or Subgraph
- [x] Map `BernoulliNB` -> `LinearClassifier` or Subgraph
- [x] Map `CategoricalNB` -> Subgraph
- [x] Map `KNeighborsClassifier` -> Distance Subgraph + `TopK` + `Gather` + Mode
- [x] Map `KNeighborsRegressor` -> Distance Subgraph + `TopK` + `Gather` + Mean
- [x] Map `RadiusNeighborsClassifier` -> Distance + Condition + Mode
- [x] Map `RadiusNeighborsRegressor` -> Distance + Condition + Mean
- [x] Map `NearestCentroid` -> Distance + `ArgMin`
- [x] Implement KdTree/BallTree inference logic purely in ONNX ops (if statically known)
- [x] Support uniform weighting in KNN
- [x] Support distance weighting in KNN

### 8. Neural Networks (Scikit-Learn) (10+ items)

- [x] Map `MLPClassifier` -> Chained `MatMul` + `Add` + `Relu`/`Tanh`/`Logistic`
- [x] Map `MLPRegressor` -> Chained `MatMul` + `Add` + Activations
- [x] Handle `MLPClassifier` with multi-label output
- [x] Extract hidden layer weights (`coefs_`)
- [x] Extract hidden layer biases (`intercepts_`)
- [x] Map `identity` activation -> No-op
- [x] Map `logistic` activation -> `Sigmoid`
- [x] Map `tanh` activation -> `Tanh`
- [x] Map `relu` activation -> `Relu`
- [x] Map `softmax` output layer -> `Softmax`

### 9. Feature Selection & Cross-Validation Meta-Estimators (15+ items)

- [x] Map `SelectKBest` -> `ArrayFeatureExtractor`
- [x] Map `SelectPercentile` -> `ArrayFeatureExtractor`
- [x] Map `SelectFpr` -> `ArrayFeatureExtractor`
- [x] Map `SelectFdr` -> `ArrayFeatureExtractor`
- [x] Map `SelectFwe` -> `ArrayFeatureExtractor`
- [x] Map `GenericUnivariateSelect` -> `ArrayFeatureExtractor`
- [x] Map `VarianceThreshold` -> `ArrayFeatureExtractor`
- [x] Map `RFE` (Recursive Feature Elimination) -> `ArrayFeatureExtractor`
- [x] Map `RFECV` -> `ArrayFeatureExtractor`
- [x] Map `SelectFromModel` -> `ArrayFeatureExtractor`
- [x] Map `SequentialFeatureSelector` -> `ArrayFeatureExtractor`
- [x] Evaluate meta-estimators post-fit (they behave as standard models at inference)

### 10. `ai.onnx.ml` Domain Mapping Operators (30+ items)

- [x] Validate `ai.onnx.ml.ArrayFeatureExtractor` semantics
- [x] Validate `ai.onnx.ml.Binarizer` semantics
- [x] Validate `ai.onnx.ml.Cast` semantics (ML domain)
- [x] Validate `ai.onnx.ml.CategoryMapper` semantics
- [x] Validate `ai.onnx.ml.DictVectorizer` semantics
- [x] Validate `ai.onnx.ml.FeatureVectorizer` semantics
- [x] Validate `ai.onnx.ml.Imputer` semantics
- [x] Validate `ai.onnx.ml.LabelEncoder` semantics
- [x] Validate `ai.onnx.ml.LinearClassifier` semantics
- [x] Validate `ai.onnx.ml.LinearRegressor` semantics
- [x] Validate `ai.onnx.ml.Normalizer` semantics
- [x] Validate `ai.onnx.ml.OneHotEncoder` semantics
- [x] Validate `ai.onnx.ml.Scaler` semantics
- [x] Validate `ai.onnx.ml.SVMClassifier` semantics
- [x] Validate `ai.onnx.ml.SVMRegressor` semantics
- [x] Validate `ai.onnx.ml.TreeEnsembleClassifier` semantics
- [x] Validate `ai.onnx.ml.TreeEnsembleRegressor` semantics
- [x] Validate `ai.onnx.ml.ZipMap` semantics
- [x] Map classlabels_strings for ML classifiers
- [x] Map classlabels_int64s for ML classifiers
- [x] Handle post_transform specifications (e.g., NONE, LOGISTIC, SOFTMAX)
- [x] Emit `ZipMap` strictly conforming to dictionary structures for probabilities

### 11. Graph Optimizations (skl2onnx specific) (20+ items)

- [x] Fuse adjacent `Scaler` operations into a single affine transform
- [x] Fuse `Normalizer` operations where mathematically redundant
- [x] Fuse `ArrayFeatureExtractor` chains
- [x] Optimize one-hot encoded outputs fed directly into linear classifiers
- [x] Eliminate identity transforms in pipelines
- [x] Optimize tree ensembles by removing mathematically dead branches
- [x] Prune unused inputs to the `Pipeline`
- [x] Consolidate multiple string-to-int `CategoryMapper` instances
- [x] Fold static mathematical operations in `FunctionTransformer`
- [x] Reduce dimensions automatically for sparse matrix assumptions

### 12. Opset Compliance & Versions

- [x] Target ONNX Standard Opset 9-21
- [x] Target `ai.onnx.ml` Opset 1
- [x] Target `ai.onnx.ml` Opset 2
- [x] Target `ai.onnx.ml` Opset 3
- [x] Target `ai.onnx.ml` Opset 4
- [x] Ensure backward compatibility with ONNX Runtime v1.10+ ML executor

### 13. Zero-Dependency & Lightweight Runtime Features (20+ items)

- [x] Pipeline conversion strictly via pure Python introspection (no C++ bindings)
- [x] Emscripten/WASM compilation capability for deploying ML pipelines directly in the browser
- [x] Convert `scikit-learn` objects trained in Python to WASM graphs instantaneously
- [x] WebWorker compatible for parallelized translation of large random forests
- [x] Output strictly typed `ai.onnx.ml` topologies validated by `onnx9000`
- [x] Memory-efficient parsing of ensembles with >10,000 trees
- [x] Support executing ONNX ML ops natively via JavaScript fallback if WebGPU not available
- [x] Enable zero-copy prediction in Pyodide bridging
- [x] Enable serverless execution of tree ensembles on AWS Lambda without scikit-learn installed

### 14. Additional Preprocessing, Cross-Decomposition & Covariance (30+ items)

- [x] Map `PLSRegression` -> `MatMul` + `Add`
- [x] Map `PLSCanonical` -> `MatMul` + `Add`
- [x] Map `CCA` (Canonical Correlation Analysis) -> `MatMul`
- [x] Map `PLSSVD` -> `MatMul`
- [x] Map `EmpiricalCovariance` -> Subgraph (Mahalanobis distance)
- [x] Map `EllipticEnvelope` -> Subgraph (Mahalanobis + Thresholding)
- [x] Map `MinCovDet` -> Subgraph
- [x] Map `OAS` -> Subgraph
- [x] Map `ShrunkCovariance` -> Subgraph
- [x] Map `LedoitWolf` -> Subgraph
- [x] Map `IsotonicRegression` -> Subgraph or Custom Operator
- [x] Map `ColumnTransformer` with `remainder="passthrough"`
- [x] Map `ColumnTransformer` with `remainder="drop"`
- [x] Map `Binarizer` with custom `threshold` parameter
- [x] Map `FunctionTransformer` with `inverse_func` handling (when requested)
- [x] Map `IterativeImputer` with `sample_posterior=True` (if applicable)
- [x] Extract vocabulary from `HashingVectorizer` effectively
- [x] Validate TF-IDF with `use_idf=False`
- [x] Validate TF-IDF with `smooth_idf=False`
- [x] Validate TF-IDF with `sublinear_tf=True`

### 15. Exhaustive Neural & Kernel Features (20+ items)

- [x] Map `Nystroem` -> Feature approximation subgraph
- [x] Map `RBFSampler` -> Feature approximation subgraph
- [x] Map `SkewedChi2Sampler` -> Feature approximation subgraph
- [x] Map `AdditiveChi2Sampler` -> Feature approximation subgraph
- [x] Support custom kernels in `SVC` (if explicitly registered)
- [x] Support custom kernels in `GaussianProcessRegressor` (if translatable)
- [x] Map `GaussianProcessRegressor` -> Subgraph (mean predictions)
- [x] Map `GaussianProcessRegressor` -> Subgraph (standard deviation output)
- [x] Map `GaussianProcessClassifier` -> Subgraph (probabilities)
- [x] Handle `MLPClassifier` with `solver='lbfgs'` (inference mapping)
- [x] Handle `MLPClassifier` with `solver='sgd'` (inference mapping)
- [x] Handle `MLPClassifier` with `solver='adam'` (inference mapping)
- [x] Support `MLPRegressor` multi-output (`n_outputs_ > 1`)
- [x] Support `MultiOutputClassifier` meta-estimator
- [x] Support `MultiOutputRegressor` meta-estimator
- [x] Support `ClassifierChain` meta-estimator
- [x] Support `RegressorChain` meta-estimator

### 16. Extensive Opset & Backend Testing (40+ items)

- [x] Test integration with `onnx9000` Python Native Executor (Accelerate)
- [x] Test integration with `onnx9000` WebGPU WASM Executor
- [x] Validate strict type-checking on `float32` vs `float64` boundaries
- [x] Ensure double precision (`float64`) preservation for high-accuracy Linear Models
- [x] Ensure single precision (`float32`) downcasting for neural net/tree performance
- [x] Unit test: `RandomForestClassifier` (1 tree, depth 1)
- [x] Unit test: `RandomForestClassifier` (100 trees, arbitrary depth)
- [x] Unit test: `SVC` (RBF kernel, 100 support vectors)
- [x] Unit test: `LogisticRegression` (binary, multinomial)
- [x] Unit test: `StandardScaler` (1D, 2D arrays)
- [x] Unit test: `Pipeline` (`StandardScaler` -> `PCA` -> `LogisticRegression`)
- [x] Unit test: `FeatureUnion` (`PCA`, `NMF`)
- [x] Unit test: `TfidfVectorizer` (100-word vocabulary)
- [x] Unit test: `CountVectorizer` (binary=True, ngram_range=(1,2))
- [x] Implement `ai.onnx.ml` compliance tests per ONNX specification
- [x] Compare `onnx9000` inference outputs with pure `scikit-learn` predict outputs (rtol=1e-5)
- [x] Test `TreeEnsembleClassifier` serialization size efficiency against native Pickling
- [x] Test `TreeEnsembleRegressor` memory loading latency via WASM

### 17. Hyper-Specific Tree Metrics & Target Class Mapping (20+ items)

- [x] Map `classlabels_ints` -> Output `Int64` type
- [x] Map `classlabels_strings` -> Output `String` type
- [x] Translate `classes_` of boolean type to `classlabels_ints` (0, 1)
- [x] Support `RandomForestClassifier` with multi-label indicator matrix output
- [x] Support `RandomForestClassifier` with `class_weight='balanced'`
- [x] Support `DecisionTreeClassifier` with `criterion='entropy'` structure maps
- [x] Support `DecisionTreeClassifier` with `criterion='gini'` structure maps
- [x] Support `DecisionTreeRegressor` with `criterion='mse'` structure maps
- [x] Support `DecisionTreeRegressor` with `criterion='mae'` structure maps
- [x] Support `ExtraTreesClassifier` random threshold mappings
- [x] Support `ExtraTreesRegressor` random threshold mappings
- [x] Ensure `n_estimators_` scales appropriately during WebGPU/WASM evaluation
- [x] Flatten nested tree structures for linear WASM memory loading
- [x] Support `HistGradientBoostingClassifier` binning strategies (mapping to thresholds)
- [x] Support `HistGradientBoostingClassifier` multi-class probability outputs
- [x] Calculate `TreeEnsembleClassifier` tree weight sums natively in graph (`post_transform='NONE'`)
- [x] Calculate `TreeEnsembleClassifier` tree weight sums natively in graph (`post_transform='LOGISTIC'`)
- [x] Calculate `TreeEnsembleClassifier` tree weight sums natively in graph (`post_transform='SOFTMAX'`)

### 18. Edge-Case Matrix Transformations (20+ items)

- [x] Ensure `MatrixFactorization` outputs correct shape constraints
- [x] Handle `SparseMatrix` density optimizations (if supported by `ai.onnx.ml`)
- [x] Handle 1D array fallback behavior (scikit-learn allows some 1D `y`, ONNX requires 2D matrices)
- [x] Support implicit broadcasting of `sample_weight` in regressions
- [x] Ensure `Binarizer` threshold applies strictly (> vs >= mapping)
- [x] Ensure `KBinsDiscretizer` handles `strategy='uniform'` natively
- [x] Ensure `KBinsDiscretizer` handles `strategy='quantile'` natively
- [x] Ensure `KBinsDiscretizer` handles `strategy='kmeans'` natively
- [x] Map `Normalizer` L1 sum correctly to 0 for all-zero arrays
- [x] Map `Normalizer` L2 sum correctly to 0 for all-zero arrays
- [x] Map `RobustScaler` IQR zero-division handling
- [x] Support `OneHotEncoder` with `drop='first'`
- [x] Support `OneHotEncoder` with `drop='if_binary'`
- [x] Support `LabelBinarizer` with `neg_label=-1, pos_label=1`
- [x] Explicit error throwing for unsupported estimators (e.g., `LocalOutlierFactor` with `novelty=False`)
