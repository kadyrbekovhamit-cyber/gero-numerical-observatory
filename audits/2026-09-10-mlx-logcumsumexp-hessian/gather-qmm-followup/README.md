# GatherQMM mixed-derivative follow-up

This extends validation of the [previously published affine-VJP orientation repair](https://www.gero.uz/research/articles/mlx-gather-qmm-transpose-gradients.html). It is not a separate new finding.

For all-one quantization codes, differentiating a scalar output sum along the zero coordinate of x and scales/biases gives:

| GatherQMM | transpose | x then parameter | parameter then x | Reference |
|---|---|---:|---:|---:|
| Original | true | 32 | 1 | 1 |
| Previous repair | true | 1 | 1 | 1 |
| Original | false | 32 | 1 | 32 |
| Previous repair | false | 32 | 32 | 32 |

The result holds for scales and biases. Central differences of the actual first gradient confirm the reference. Ordinary quantized_matmul passes both differentiation orders as a control. Although transpose=true did not reveal the parameter orientation issue at first order, its x-VJP invokes GatherQMM with transpose=false at the next order.

Fresh native execution: 24 comparisons, four mismatches before and none after. [Test](mixed_regression.cpp), [runner](build_mixed.py), [before log](run-before.log), [after log](run-after.log), [build records](mixed-build-results.json). These results reuse the earlier repair's isolated before.o and after.o translation units, linked against the existing CPU archive. They do not use the new logcumsumexp patch. To rerun, first execute the earlier audit's build_and_test.py, set QMM_PREVIOUS_OBJECTS to that audit directory, and MLX_CPU_BUILD to its CPU build directory, then run python3 build_mixed.py. One thread, CPU only.

The [saved Python probe](probe-results.json) and [probe code](probe.py) also retain a corrected reference: transpose=false must yield 32 rather than 1 because one parameter affects a physical group of 32 output weights. The initial expected field was mistaken and was corrected; the measured values were unchanged. The fresh native rerun uses the corrected reference.

An unresolved observation in the saved scan probe is the gradient [1,1] at a common input offset 1e8 where [1.5,0.5] is expected. This accuracy observation has no completed fix or applicability audit here and is excluded from the count of completed findings. The [main report](../README.md) concerns a different zero-cotangent higher-derivative defect.
