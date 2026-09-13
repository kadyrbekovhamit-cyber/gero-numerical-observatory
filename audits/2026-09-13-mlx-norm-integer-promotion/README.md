# Apple MLX: integer norms can become zero, negative, or NaN

Independent numerical audit — Xamit Kadirbekov / GERO Research, 13 September 2026.

Native CPU reproduction: int8 [100] has L2 norm 4 instead of 100; int32 [65536] has L2 norm 0 instead of 65536; int8 [-128] has L1 norm -128 instead of 128. The cause is integer arithmetic before float promotion.

The clean-source regression covers 1,356 bounded scenarios: 244 fail before the input-promotion candidate, none fail after. Very large uint64 values and large powers retain separate floating-point overflow limitations. No general norm repair, GPU result, model impact, or priority claim is made.

[Complete report and evidence archive on Hugging Face](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/mlx-norm-integer-promotion.md).

Source build ce916dbbcaa88e433b6fd1e60a17f766d49c27fe (version header 0.32.3). AI-assisted research.
