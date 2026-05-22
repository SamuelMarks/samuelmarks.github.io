# Supported Frameworks Coverage

This file tracks the level of support for various ML frameworks in ONNX9000.

## Summary

| Target | Supported | Total | Percentage |
|---|---|---|---|
| ONNX Spec | 200 | 200 | 100.00% |
| Torch | 933 | 933 | 100.00% |
| Tensorflow | 8095 | 31717 | 25.52% |
| Keras | 7790 | Unknown | N/A |
| Jax | 0 | 5598 | 0.00% |
| Flax | 0 | 25727 | 0.00% |
| Paddle | 96 | 14217 | 0.68% |
| Coremltools | 0 | 4339 | 0.00% |
| Sklearn | 115 | 1203 | 9.56% |
| Xgboost | 2 | 298 | 0.67% |
| Lightgbm | 2 | 113 | 1.77% |
| Catboost | 2 | 168 | 1.19% |
| Pyspark | 1 | 7741 | 0.01% |
| H2o | 1 | 1653 | 0.06% |
| Libsvm | 1 | 40 | 2.50% |
| Cntk | 0 | 1377 | 0.00% |
| Mxnet | 0 | 2611 | 0.00% |
| Caffe | 0 | 149 | 0.00% |
| Gguf | 2 | 383 | 0.52% |
| Safetensors | 2 | 53 | 3.77% |

## Detailed Support Breakdown

Below are the exhaustive API tracking files. They contain the specific list of classes, functions, and models we must implement adapters for to be fully compliant with each framework.

- [**ONNX Standard** (`200/200` supported)](compliance/ONNX_SUPPORT.md)
- [**Torch** (`933/933` supported)](compliance/TORCH_SUPPORT.md)
- [**Tensorflow** (`8095/31717` supported)](compliance/TENSORFLOW_SUPPORT.md)
- [**Jax** (`0/5598` supported)](compliance/JAX_SUPPORT.md)
- [**Flax** (`0/25727` supported)](compliance/FLAX_SUPPORT.md)
- [**Paddle** (`96/14217` supported)](compliance/PADDLE_SUPPORT.md)
- [**Coremltools** (`0/4339` supported)](compliance/COREMLTOOLS_SUPPORT.md)
- [**Sklearn** (`115/1203` supported)](compliance/SKLEARN_SUPPORT.md)
- [**Xgboost** (`2/298` supported)](compliance/XGBOOST_SUPPORT.md)
- [**Lightgbm** (`2/113` supported)](compliance/LIGHTGBM_SUPPORT.md)
- [**Catboost** (`2/168` supported)](compliance/CATBOOST_SUPPORT.md)
- [**Pyspark** (`1/7741` supported)](compliance/PYSPARK_SUPPORT.md)
- [**H2o** (`1/1653` supported)](compliance/H2O_SUPPORT.md)
- [**Libsvm** (`1/40` supported)](compliance/LIBSVM_SUPPORT.md)
- [**Cntk** (`0/1377` supported)](compliance/CNTK_SUPPORT.md)
- [**Mxnet** (`0/2611` supported)](compliance/MXNET_SUPPORT.md)
- [**Caffe** (`0/149` supported)](compliance/CAFFE_SUPPORT.md)
- [**Gguf** (`2/383` supported)](compliance/GGUF_SUPPORT.md)
- [**Safetensors** (`2/53` supported)](compliance/SAFETENSORS_SUPPORT.md)