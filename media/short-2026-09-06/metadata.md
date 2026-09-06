# YouTube release metadata

Title: Can the ONNX Reference Lose Accuracy? | GERO Numerical Observatory

Description:
A reference implementation is useful evidence, but it is still a floating-point program. This 71-second GERO overview explains a reproduced CPU benchmark: 123 synthetic graphs, 246 evaluations, 232 passes and 14 divergence observations across seven graphs.

All 14 flagged archives replayed successfully. A separate 80-digit Decimal calculation examined the seven graph formulas. One float32 LayerNormalization graph produced 17 non-finite outputs out of 68 in ONNX Runtime. In other cases, the official reference evaluator produced non-finite outputs.

Scope: ONNX Runtime 1.22.1, ONNX 1.19.0, NumPy 2.2.6; Python 3.12.13, macOS 15.5 arm64 CPU, one thread, opset 19 / IR 10. These are version-specific observations, not seven newly confirmed bugs, latest-version claims or demonstrated regressions between versions. The Decimal check is algorithmically independent, not third-party validation. QuantizeLinear and DynamicQuantizeLinear are excluded; DequantizeLinear remains covered.

Interactive reports and graph downloads:
https://www.gero.uz/stability/

Full method and limitations:
https://www.gero.uz/research/articles/onnx-numerical-observatory-reproduced-snapshot.html

Source, tests and exact measured snapshot:
https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory

Primary diagnostic record:
https://www.gero.uz/stability/reports/reverification.json

Official documentation:
https://onnx.ai/onnx/api/reference.html
https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html

Narration uses a synthetic voice for the fictional digital narrator Alex Vector. No real-person voice cloning or avatar. Original diagrams and synthetic test data; no music. Research and editorial review by Xamit Kadirbekov / GERO, published on Technology Product. GERO is the author's own research product.

Reviewed: 6 September 2026. Corrections will be dated in the research note.

#ONNX #MachineLearning #NumericalComputing #GERO #Shorts

Audience: Not made for children; technical educational content.
Language: English.
Tags: ONNX, ONNX Runtime, numerical stability, floating point, machine learning, software testing, GERO, LayerNormalization, LogSoftmax.
Captions: output/captions.en.srt.
Chapters: omitted for this short format.
Suggested comment: Inspect the exact tolerances and download a runnable bundle at https://www.gero.uz/stability/ . The full scope and limits are in the linked research note.
