# GERO numerical stability report

Run: `ecf59104f767379863b0c7913f410831f5fbde7515a943bb22588fcb9332eee3`

Local synthetic-graph CPU benchmark. A divergence is an investigation candidate, not an upstream bug or novelty claim.

ONNX 1.19.0 / ONNX Runtime 1.22.1 / NumPy 2.2.6 / arm64.

Summary: {'pass': 232, 'divergence': 14}. Excluded: QuantizeLinear and DynamicQuantizeLinear.

Regressions are assessed only against a compatible explicit baseline. Both-NaN is never a pass.

| Case | Optimization | Status | Max absolute error | Failed checks |
|---|---|---|---:|---|
| CosineSimilarity/large_offset/float16/4x17/seed-20260906 | all | divergence | None | reference:batch_duplicate, reference:batch_partition, reference:batch_permutation, reference:batch_reshape, reference:cosine_float64_oracle, reference:cosine_range, reference:finite_output |
| CosineSimilarity/large_offset/float16/4x17/seed-20260906 | disabled | divergence | None | reference:batch_duplicate, reference:batch_partition, reference:batch_permutation, reference:batch_reshape, reference:cosine_float64_oracle, reference:cosine_range, reference:finite_output |
| CosineSimilarity/tiny/float16/4x17/seed-20260906 | all | divergence | 1.9234619140625 | reference:cosine_float64_oracle, reference:cosine_range |
| CosineSimilarity/tiny/float16/4x17/seed-20260906 | disabled | divergence | 1.9234619140625 | reference:cosine_float64_oracle, reference:cosine_range |
| LayerNormalization/large_offset/float32/4x17/seed-20260906 | all | divergence | 2.307088240981102 | ort:batch_duplicate, ort:batch_partition, ort:batch_permutation, ort:batch_reshape, ort:centered_float64_oracle, ort:equivalent_graph, ort:finite_output, ort:normalized_variance, reference:centered_float64_oracle, reference:normalized_variance |
| LayerNormalization/large_offset/float32/4x17/seed-20260906 | disabled | divergence | 2.307088240981102 | ort:batch_duplicate, ort:batch_partition, ort:batch_permutation, ort:batch_reshape, ort:centered_float64_oracle, ort:equivalent_graph, ort:finite_output, ort:normalized_variance, reference:centered_float64_oracle, reference:normalized_variance |
| LogSoftmax/random-001/float32/4x105/seed-20260906 | all | divergence | 0.413299560546875 | reference:batch_duplicate, reference:batch_partition, reference:batch_permutation, reference:batch_reshape, reference:equivalent_graph, reference:finite_output |
| LogSoftmax/random-001/float32/4x105/seed-20260906 | disabled | divergence | 0.413299560546875 | reference:batch_duplicate, reference:batch_partition, reference:batch_permutation, reference:batch_reshape, reference:equivalent_graph, reference:finite_output |
| LogSoftmax/random-022/float32/4x127/seed-20260906 | all | divergence | 0.2647857666015625 | reference:batch_duplicate, reference:batch_partition, reference:batch_permutation, reference:batch_reshape, reference:equivalent_graph, reference:finite_output |
| LogSoftmax/random-022/float32/4x127/seed-20260906 | disabled | divergence | 0.2647857666015625 | reference:batch_duplicate, reference:batch_partition, reference:batch_permutation, reference:batch_reshape, reference:equivalent_graph, reference:finite_output |
| LpNormalization/large_offset/float16/4x17/seed-20260906 | all | divergence | 0.2431640625 | reference:l2_unit_or_zero, reference:scaled_float64_oracle |
| LpNormalization/large_offset/float16/4x17/seed-20260906 | disabled | divergence | 0.2431640625 | reference:l2_unit_or_zero, reference:scaled_float64_oracle |
| LpNormalization/tiny/float16/4x17/seed-20260906 | all | divergence | 0.23779296875 | reference:batch_duplicate, reference:batch_partition, reference:batch_permutation, reference:batch_reshape, reference:equivalent_graph, reference:finite_output, reference:l2_unit_or_zero, reference:scaled_float64_oracle |
| LpNormalization/tiny/float16/4x17/seed-20260906 | disabled | divergence | 0.23779296875 | reference:batch_duplicate, reference:batch_partition, reference:batch_permutation, reference:batch_reshape, reference:equivalent_graph, reference:finite_output, reference:l2_unit_or_zero, reference:scaled_float64_oracle |
