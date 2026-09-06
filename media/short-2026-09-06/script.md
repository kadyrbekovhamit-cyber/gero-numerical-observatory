# Narration — verified against sources.md

Can a reference implementation lose numerical accuracy? Yes. GERO tests the mathematical contract as well as comparing the two outputs.

We ran one hundred twenty-three synthetic graphs with runtime optimizations off and on. Two hundred thirty-two evaluations passed. Fourteen divergences came from seven graphs.

All fourteen flagged archives reproduced. We also checked the seven graph formulas with eighty-digit decimal calculations.

One layer normalization graph returned seventeen non-finite values out of sixty-eight in ONNX Runtime. The high-precision calculation was finite.

In other cases, the official reference evaluator produced non-finite values. A disagreement needs investigation before assigning blame.

These are version-specific observations. We have not established new bugs or regressions between versions. Quantize Linear is excluded from this benchmark.

Explore the tolerances, test graphs, and reproducible reports at gero dot uz.
