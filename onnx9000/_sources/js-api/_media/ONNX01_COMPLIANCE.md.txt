# ONNX Standard Compliance & Testing Tracker

## Description

This document tracks the complete compliance and validation of the `onnx9000` ecosystem against the official ONNX Standard specifications.
To ensure our zero-dependency pure-Python implementation is numerically and structurally identical to the massive C++ reference implementation (`onnxruntime`), we must execute the thousands of standard ONNX Backend Test Cases. These tests guarantee that `onnx9000` evaluates every single operator, shape, broadcasting rule, and mathematical edge case precisely according to the strict `ai.onnx` and `ai.onnx.ml` specifications across Opsets 1 through 21.

<!-- COVERAGE_SUMMARY_START -->
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
<!-- COVERAGE_SUMMARY_END -->

## Exhaustive Parity Checklist

### 1. Test Runner Architecture & Environment (30+ items)

- [xx] Implement pure-Python test case runner targeting `.pb` directories
- [xx] Parse `test_data_set_N/input_X.pb` seamlessly without `onnx` native package
- [xx] Parse `test_data_set_N/output_X.pb` seamlessly without `onnx` native package
- [xx] Support loading `model.onnx` from each standard backend test directory
- [xx] Execute loaded model dynamically via `onnx9000.InferenceSession`
- [xx] Compare inference outputs to reference `.pb` outputs natively using NumPy
- [xx] Handle explicit type matching (`dtype` validation must be exact)
- [xx] Handle explicit shape matching (`shape` validation must be exact)
- [xx] Implement `rtol` (Relative Tolerance) logic identical to `numpy.testing.assert_allclose`
- [xx] Implement `atol` (Absolute Tolerance) logic identical to `numpy.testing.assert_allclose`
- [xx] Support explicitly overriding tolerance limits for mathematically loose ops (e.g. `Gelu`, `Erf`)
- [xx] Support explicitly defining skip-lists for recognized un-supported tests (regex patterns)
- [xx] Parse expected error assertions (tests designed to fail structurally)
- [xx] Output JUnit-compatible XML reports natively
- [xx] Output human-readable console summary (Pass/Fail/Skip/Error)
- [xx] Execute tests sequentially inside `pytest` (using parametrization)
- [xx] Execute tests concurrently to speed up CI/CD pipelines
- [xx] Track backend execution time per test
- [xx] Support testing specific Opsets (e.g., `--opset=17`)
- [xx] Expose an API to inject custom Execution Providers (e.g., test Apple Accelerate vs standard CPU)
- [xx] Support downloading standard ONNX test repositories natively via HTTP if missing
- [xx] Extract ZIP archives containing standard tests automatically
- [xx] Support generating `bfloat16` and `float16` tolerance validations safely in JS/Python
- [xx] Handle large external data payloads (`.bin`) associated with backend tests
- [xx] Verify node topological sorting capabilities within the runner implicitly
- [xx] Handle string extraction correctly (`String` Tensor types)
- [xx] Prevent Python Garbage Collector from sweeping large test tensors prematurely
- [xx] Expose the runner logic to a WASM environment (test ONNX compliance inside Chrome/Safari)
- [xx] Implement identical `np.nan` and `np.inf` checks natively
- [xx] Support strictly executing `ai.onnx.ml` domain test sets

### 2. Math & Elementwise Operator Compliance (40+ items)

- [xx] Pass `test_abs`
- [xx] Pass `test_acos`
- [xx] Pass `test_acosh`
- [xx] Pass `test_add` (standard)
- [xx] Pass `test_add_bcast` (broadcasting 1D to ND)
- [xx] Pass `test_add_uint8`
- [xx] Pass `test_and` (standard)
- [xx] Pass `test_and_bcast`
- [xx] Pass `test_asin`
- [xx] Pass `test_asinh`
- [xx] Pass `test_atan`
- [xx] Pass `test_atanh`
- [xx] Pass `test_bitshift_left`
- [xx] Pass `test_bitshift_right`
- [xx] Pass `test_bitwise_and`
- [xx] Pass `test_bitwise_not`
- [xx] Pass `test_bitwise_or`
- [xx] Pass `test_bitwise_xor`
- [xx] Pass `test_ceil`
- [xx] Pass `test_clip`
- [xx] Pass `test_clip_default`
- [xx] Pass `test_clip_default_min`
- [xx] Pass `test_clip_default_max`
- [xx] Pass `test_cos`
- [xx] Pass `test_cosh`
- [xx] Pass `test_div` (standard)
- [xx] Pass `test_div_bcast`
- [xx] Pass `test_erf`
- [xx] Pass `test_exp`
- [xx] Pass `test_floor`
- [xx] Pass `test_isinf`
- [xx] Pass `test_isinf_negative`
- [xx] Pass `test_isinf_positive`
- [xx] Pass `test_isnan`
- [xx] Pass `test_log`
- [xx] Pass `test_mod_broadcast`
- [xx] Pass `test_mod_fmod_mixed_sign`
- [xx] Pass `test_mod_uint8`
- [xx] Pass `test_mul` (standard)
- [xx] Pass `test_mul_bcast`

### 3. Math & Elementwise (Continued) (30+ items)

- [xx] Pass `test_neg`
- [xx] Pass `test_not`
- [xx] Pass `test_or` (standard)
- [xx] Pass `test_or_bcast`
- [xx] Pass `test_pow`
- [xx] Pass `test_pow_bcast_array`
- [xx] Pass `test_pow_bcast_scalar`
- [xx] Pass `test_pow_types_float32_int64`
- [xx] Pass `test_reciprocal`
- [xx] Pass `test_round`
- [xx] Pass `test_sign`
- [xx] Pass `test_sin`
- [xx] Pass `test_sinh`
- [xx] Pass `test_sqrt`
- [xx] Pass `test_sub` (standard)
- [xx] Pass `test_sub_bcast`
- [xx] Pass `test_tan`
- [xx] Pass `test_tanh`
- [xx] Pass `test_xor` (standard)
- [xx] Pass `test_xor_bcast`
- [xx] Pass `test_equal`
- [xx] Pass `test_equal_bcast`
- [xx] Pass `test_greater`
- [xx] Pass `test_greater_bcast`
- [xx] Pass `test_greater_equal`
- [xx] Pass `test_greater_equal_bcast`
- [xx] Pass `test_less`
- [xx] Pass `test_less_bcast`
- [xx] Pass `test_less_equal`
- [xx] Pass `test_less_equal_bcast`

### 4. Matrix & Neural Network Operators Compliance (40+ items)

- [xx] Pass `test_matmul_2d`
- [xx] Pass `test_matmul_3d`
- [xx] Pass `test_matmul_4d`
- [xx] Pass `test_gemm_default_matrix_bias`
- [xx] Pass `test_gemm_default_no_bias`
- [xx] Pass `test_gemm_default_scalar_bias`
- [xx] Pass `test_gemm_default_single_elem_vector_bias`
- [xx] Pass `test_gemm_default_vector_bias`
- [xx] Pass `test_gemm_default_zero_bias`
- [xx] Pass `test_conv_with_autopad_same`
- [xx] Pass `test_conv_with_strides_padding`
- [xx] Pass `test_conv_with_strides_no_padding`
- [xx] Pass `test_conv_with_strides_and_asymmetric_padding`
- [xx] Pass `test_conv_1d`
- [xx] Pass `test_conv_3d`
- [xx] Pass `test_conv_depthwise`
- [xx] Pass `test_convtranspose`
- [xx] Pass `test_convtranspose_1d`
- [xx] Pass `test_convtranspose_3d`
- [xx] Pass `test_maxpool_1d_default`
- [xx] Pass `test_maxpool_2d_default`
- [xx] Pass `test_maxpool_2d_strides`
- [xx] Pass `test_maxpool_3d_default`
- [xx] Pass `test_averagepool_1d_default`
- [xx] Pass `test_averagepool_2d_default`
- [xx] Pass `test_averagepool_2d_strides`
- [xx] Pass `test_averagepool_3d_default`
- [xx] Pass `test_globalaveragepool`
- [xx] Pass `test_globalmaxpool`
- [xx] Pass `test_globallppool`
- [xx] Pass `test_batchnorm_epsilon`
- [xx] Pass `test_batchnorm_example`
- [xx] Pass `test_dropout_default` (should pass via inference mode Identity mapping)
- [xx] Pass `test_dropout_random` (should pass via inference mode)
- [xx] Pass `test_lrn`
- [xx] Pass `test_lrn_default`
- [xx] Pass `test_roialign`
- [xx] Pass `test_maxroipool`
- [xx] Pass `test_celu`
- [xx] Pass `test_celu_expanded`

### 5. Activations & Reductions Compliance (30+ items)

- [xx] Pass `test_elu`
- [xx] Pass `test_hardmax_axis_0`
- [xx] Pass `test_hardmax_axis_1`
- [xx] Pass `test_hardmax_axis_2`
- [xx] Pass `test_hardmax_default_axis`
- [xx] Pass `test_hardsigmoid`
- [xx] Pass `test_hardswish`
- [xx] Pass `test_leakyrelu`
- [xx] Pass `test_logsoftmax_axis_0`
- [xx] Pass `test_logsoftmax_axis_1`
- [xx] Pass `test_mish`
- [xx] Pass `test_prelu_broadcast`
- [xx] Pass `test_relu`
- [xx] Pass `test_selu`
- [xx] Pass `test_shrink_hard`
- [xx] Pass `test_shrink_soft`
- [xx] Pass `test_sigmoid`
- [xx] Pass `test_softmax_axis_0`
- [xx] Pass `test_softmax_axis_1`
- [xx] Pass `test_softplus`
- [xx] Pass `test_softsign`
- [xx] Pass `test_reduce_max_do_not_keepdims`
- [xx] Pass `test_reduce_max_keepdims`
- [xx] Pass `test_reduce_mean_do_not_keepdims`
- [xx] Pass `test_reduce_mean_keepdims`
- [xx] Pass `test_reduce_min_do_not_keepdims`
- [xx] Pass `test_reduce_min_keepdims`
- [xx] Pass `test_reduce_prod_do_not_keepdims`
- [xx] Pass `test_reduce_sum_keepdims`
- [xx] Pass `test_reduce_sum_square_do_not_keepdims`

### 6. Shape, Tensor Manipulation & Selection (40+ items)

- [xx] Pass `test_argmax_default_axis_example`
- [xx] Pass `test_argmax_keepdims_example`
- [xx] Pass `test_argmin_default_axis_example`
- [xx] Pass `test_argmin_keepdims_example`
- [xx] Pass `test_cast_FLOAT_to_FLOAT16`
- [xx] Pass `test_cast_FLOAT_to_STRING`
- [xx] Pass `test_cast_STRING_to_FLOAT`
- [xx] Pass `test_concat_1d_axis_0`
- [xx] Pass `test_concat_2d_axis_0`
- [xx] Pass `test_concat_2d_axis_1`
- [xx] Pass `test_constant`
- [xx] Pass `test_constantofshape_float_ones`
- [xx] Pass `test_constantofshape_int_zeros`
- [xx] Pass `test_cumsum_1d`
- [xx] Pass `test_cumsum_1d_exclusive`
- [xx] Pass `test_cumsum_1d_reverse`
- [xx] Pass `test_cumsum_2d_axis_0`
- [xx] Pass `test_depthtospace_crd_mode`
- [xx] Pass `test_depthtospace_dcr_mode`
- [xx] Pass `test_expand_dim_changed`
- [xx] Pass `test_expand_dim_unchanged`
- [xx] Pass `test_flatten_axis0`
- [xx] Pass `test_flatten_axis1`
- [xx] Pass `test_flatten_default_axis`
- [xx] Pass `test_gather_0`
- [xx] Pass `test_gather_1`
- [xx] Pass `test_gather_negative_indices`
- [xx] Pass `test_gatherelements_0`
- [xx] Pass `test_gathernd_example_int32`
- [xx] Pass `test_nonzero_example`
- [xx] Pass `test_pad_edge`
- [xx] Pass `test_pad_reflect`
- [xx] Pass `test_pad_constant`
- [xx] Pass `test_reshape_extended_dims`
- [xx] Pass `test_reshape_negative_dim`
- [xx] Pass `test_reshape_one_dim`
- [xx] Pass `test_reshape_reduced_dims`
- [xx] Pass `test_shape`
- [xx] Pass `test_shape_example`
- [xx] Pass `test_size`
- [xx] Pass `test_slice_default_axes`

### 7. Tensor Manipulation (Continued) & Control Flow (30+ items)

- [xx] Pass `test_slice_negative_axes`
- [xx] Pass `test_slice_negative_steps`
- [xx] Pass `test_split_equal_parts_1d`
- [xx] Pass `test_split_equal_parts_2d`
- [xx] Pass `test_split_variable_parts_1d`
- [xx] Pass `test_squeeze`
- [xx] Pass `test_squeeze_negative_axes`
- [xx] Pass `test_tile`
- [xx] Pass `test_tile_precomputed`
- [xx] Pass `test_top_k`
- [xx] Pass `test_top_k_negative_axis`
- [xx] Pass `test_top_k_smallest`
- [xx] Pass `test_transpose_all_permutations_0`
- [xx] Pass `test_transpose_all_permutations_1`
- [xx] Pass `test_transpose_all_permutations_2`
- [xx] Pass `test_transpose_all_permutations_3`
- [xx] Pass `test_transpose_default`
- [xx] Pass `test_unsqueeze_axis_0`
- [xx] Pass `test_unsqueeze_axis_1`
- [xx] Pass `test_unsqueeze_axis_2`
- [xx] Pass `test_unsqueeze_negative_axes`
- [xx] Pass `test_where_example`
- [xx] Pass `test_where_long_example`
- [xx] Pass `test_if_seq`
- [xx] Pass `test_loop13_seq`
- [xx] Pass `test_scan_sum`
- [xx] Pass `test_scan9_sum`
- [xx] Pass `test_sequence_insert_at_back`
- [xx] Pass `test_sequence_insert_at_front`
- [xx] Pass `test_sequence_construct`

### 8. `ai.onnx.ml` Classical ML Domain Compliance (30+ items)

- [xx] Pass `test_ml_array_feature_extractor`
- [xx] Pass `test_ml_binarizer`
- [xx] Pass `test_ml_cast`
- [xx] Pass `test_ml_category_mapper_int64_to_string`
- [xx] Pass `test_ml_category_mapper_string_to_int64`
- [xx] Pass `test_ml_dict_vectorizer`
- [xx] Pass `test_ml_feature_vectorizer`
- [xx] Pass `test_ml_imputer_float_mean`
- [xx] Pass `test_ml_imputer_float_median`
- [xx] Pass `test_ml_label_encoder_string`
- [xx] Pass `test_ml_label_encoder_tensor`
- [xx] Pass `test_ml_linear_classifier`
- [xx] Pass `test_ml_linear_regressor`
- [xx] Pass `test_ml_normalizer_l1`
- [xx] Pass `test_ml_normalizer_l2`
- [xx] Pass `test_ml_normalizer_max`
- [xx] Pass `test_ml_one_hot_encoder`
- [xx] Pass `test_ml_scaler`
- [xx] Pass `test_ml_svm_classifier`
- [xx] Pass `test_ml_svm_regressor`
- [xx] Pass `test_ml_tree_ensemble_classifier`
- [xx] Pass `test_ml_tree_ensemble_classifier_binary`
- [xx] Pass `test_ml_tree_ensemble_classifier_multiclass`
- [xx] Pass `test_ml_tree_ensemble_regressor`
- [xx] Pass `test_ml_zipmap_int64`
- [xx] Pass `test_ml_zipmap_string`
- [xx] Pass `test_ai_onnx_ml_tree_ensemble_multi_target`
- [xx] Validate ML topologies output shapes flawlessly matches Opset 2 rules
- [xx] Validate ML topologies handle sparse structures if tested
- [xx] Catch expected execution failures natively on invalid `ai.onnx.ml` topologies

### 9. Advanced Operators, Sequences & Quantization Compliance (30+ items)

- [xx] Pass `test_sequence_at_negative`
- [xx] Pass `test_sequence_empty`
- [xx] Pass `test_sequence_erase_at_back`
- [xx] Pass `test_sequence_erase_at_front`
- [xx] Pass `test_sequence_length`
- [xx] Pass `test_split_to_sequence_1`
- [xx] Pass `test_split_to_sequence_2`
- [xx] Pass `test_split_to_sequence_n_parts`
- [xx] Pass `test_concat_from_sequence_1d`
- [xx] Pass `test_concat_from_sequence_2d`
- [xx] Pass `test_dynamicquantizelinear`
- [xx] Pass `test_dynamicquantizelinear_expanded`
- [xx] Pass `test_dequantizelinear`
- [xx] Pass `test_dequantizelinear_axis`
- [xx] Pass `test_quantizelinear`
- [xx] Pass `test_quantizelinear_axis`
- [xx] Pass `test_qlinearconv`
- [xx] Pass `test_qlinearmatmul_2D`
- [xx] Pass `test_qlinearmatmul_3D`
- [xx] Pass `test_matmulinteger`
- [xx] Pass `test_convinteger_with_padding`
- [xx] Pass `test_convinteger_without_padding`
- [xx] Pass `test_resize_downsample_scales_cubic`
- [xx] Pass `test_resize_downsample_scales_linear`
- [xx] Pass `test_resize_downsample_scales_nearest`
- [xx] Pass `test_resize_tf_crop_and_resize`
- [xx] Pass `test_resize_upsample_scales_cubic`
- [xx] Pass `test_resize_upsample_scales_linear`
- [xx] Pass `test_resize_upsample_scales_nearest`
- [xx] Pass `test_rnn_seq_length`
- [xx] Pass `test_gru_seq_length`
- [xx] Pass `test_lstm_with_initial_bias`

### 10. Explicit Edge Cases, Precision & Subgraphs (25+ items)

- [xx] Validate 0D scalar broadcasts successfully against ND tensors (`test_add_scalar_ND`)
- [xx] Validate empty tensor execution (`test_empty_tensor_1d`)
- [xx] Validate empty tensor execution propagates properly through math ops
- [xx] Validate `Float16` numerical tolerance (`rtol=1e-3`, `atol=1e-3`)
- [xx] Validate `BFloat16` numerical tolerance
- [xx] Handle `Float64` precision retention across intermediate ops
- [xx] Ensure `Softmax` returns mathematically sound ranges (`[0, 1]` with sum 1.0) on extreme inputs
- [xx] Execute deeply nested `If` subgraphs returning dynamic tensor shapes
- [xx] Execute `Loop` where loop body modifies loop carried dependencies flawlessly
- [xx] Execute `Scan` over dimension 1 (sequence scanning)
- [xx] Execute `Trilu` extracting upper and lower bounds correctly
- [xx] Test negative axis indices (`axis=-1`) behave exactly like Python/NumPy logic
- [xx] Validate `ScatterND` handles duplicate indices exactly per the specification
- [xx] Ensure `GatherND` extracts multi-dimensional slices successfully
- [xx] Test `Unique` correctly emits inverse indices
- [xx] Test `Unique` correctly emits sorted results
- [xx] Evaluate `Mod` utilizing fmod (float logic) vs standard modulo
- [xx] Ensure `BitShift` properly clamps out-of-bounds shift amounts if specified
- [xx] Check `Einsum` parses standard NumPy format equations identically
- [xx] Confirm `CumSum` exclusive flag operates as specified
- [xx] Confirm `CumSum` reverse flag operates as specified
- [xx] Execute standard ONNX ResNet compliance test seamlessly
- [xx] Execute standard ONNX BERT compliance test seamlessly
- [xx] Execute standard ONNX VGG compliance test seamlessly
