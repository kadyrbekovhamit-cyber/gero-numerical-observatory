# Apple MLX can sample tokens with probability zero

NUMERICAL AUDIT · 8 NOVEMBER 2026

With num_samples, mx.random.categorical takes an inverse-CDF path whose float32 total and boundaries drift apart; on CPU a token with logit −inf was drawn 207 times in 1,000,000 samples over a 151,936-token vocabulary.

Xamit Kadirbekov, Daniyal Kadirbekov · AI-assisted investigation and writing

**Checked 8 November 2026: PR #4847 is open and not merged. It awaits maintainer review; no review has been posted yet. No upstream acceptance or released fix is claimed.**

[Upstream pull request #4847](https://github.com/ml-explore/mlx/pull/4847) · [GitHub catalog](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/mlx-categorical-num-samples-zero-probability.md)

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO Numerical Assurance). Published 8 November 2026.

- **207** — banned picks per 1,000,000 on CPU (expected 0)
- **~1 in 4,300** — CPU rate, N(0, 9) logits
- **0** — banned picks on a local CPU build with the fix

## Published proposal

- [ml-explore/mlx PR #4847](https://github.com/ml-explore/mlx/pull/4847) — head commit `94c398a008a9a7f1a880c8baf91806539055cc28` on main `fc344c9`; open at submission on 8 November 2026 (Tashkent time). Submitted by Xamit Kadirbekov.

## What the function does

A language model produces one score (logit) per token. Sampling turns the scores into probabilities and draws tokens at random. Setting a logit to −inf is the standard way to ban a token: its probability is exactly zero, so it must never be drawn. This is how padding ids, disallowed tokens and top-k / top-p cut-offs are usually applied.

`mx.random.categorical(logits, num_samples=k)` draws k samples from one distribution. Since [MLX PR #4177](https://github.com/ml-explore/mlx/pull/4177) (August 2026; releases 0.32.1 and later) this call uses a fast inverse-CDF path: the probabilities are laid out one after another on a line, a uniform random point is thrown on the line, and the token whose segment contains the point is returned.

## The discrepancy

In `categorical_inverse_cdf` (`mlx/random.cpp`) the weights are `w = exp(logits − max)`. The length of the line is taken from `sum(w)`, while the segment boundaries come from a separate exclusive `cumsum(w)`, followed by `searchsorted(cdf, u, "right") − 1`. In float32 the two reductions round differently, so they drift apart:

- **Past the last boundary.** When `sum(w)` is larger than the end of the cumulative sum, a point can land beyond the last boundary. It is then assigned to the last index, even if that token's logit is −inf.

- **Inside the vocabulary (GPU).** The parallel `cumsum` on the GPU is not monotone after rounding, so a banned token in the middle of the vocabulary can receive a tiny nonzero segment.

## Minimal reproduction

```python
import mlx.core as mx, numpy as np
V = 151_936                                            # Qwen2/2.5 vocabulary size
logits = mx.array(np.random.default_rng(0).standard_normal(V).astype(np.float32) * 3)
logits[-1] = -float("inf")                             # last token: probability exactly 0
for dev in (mx.cpu, mx.gpu):
    s = mx.random.categorical(logits, num_samples=1_000_000, key=mx.random.key(0), stream=dev)
    print(dev, "banned token drawn", (s == V - 1).sum().item(), "times in 1e6 (expected 0)")
```

## Measured results

MLX 0.32.4 from pip, MacBook Air M4. In every row the correct number of banned tokens drawn is zero.

| Scenario | CPU | GPU |
|---|---|---|
| Vocabulary 151,936, last token −inf, 1,000,000 samples (reproduction above) | 207 | 0 |
| Last token and 500 middle tokens −inf, 1,000,000 samples (the regression test in the PR) | 195 | 1 |
| Rate, logits N(0, 9) | about 1 in 4,300 samples | — |
| Rate, LLM-like logits with masked padding ids | about 1 in 6,200 samples | — |
| Banned tokens in the middle of the vocabulary, 20,000,000 samples | — | 8–12 |

## Who is affected

**Not affected:** ordinary text generation in mlx-lm, which does not pass `num_samples`; without it `categorical` uses a different method.

**Affected:** code that draws many samples from one distribution in a single call — best-of-n and pass@k sampling, reinforcement-learning rollouts, and ports of `torch.multinomial(probs, k)`. In these settings a token that was deliberately masked out can still appear.

## Fix and tests

- Scale the random point by the last value of the inclusive cumulative sum, so the line length and the boundaries come from the same computation.

- Clamp the index to at most n − 1.

- If rounding still gives a zero-weight entry a nonzero segment, move the draw to the nearest entry at or before it that has a positive weight.

- Regression test in `python/tests/test_random.py` (`test_categorical`): 151,936 logits, last token and 500 middle tokens banned, 1,000,000 samples on CPU and, when Metal is available, on GPU; no banned token may appear.

Local CPU build from source with the fix: 0 banned tokens in 1,000,000 samples; `test_random` 15 of 15 tests pass.

## Upstream status

As of 8 November 2026, PR #4847 is open and requires review; no review has been posted. Nothing has been merged or released.

## Limits

The GPU path with the fix was not built locally (no Metal compiler on the test machine); it is left to the project's CI, which has not run yet. All measurements come from one machine and one MLX version; rates depend on the vocabulary size and the shape of the logits. The rates are measured frequencies, not a bound.

## Attribution and license

AI tools (Claude) assisted with the analysis, the reproduction and measurement scripts, the patch and test code, and the text of this report; the results were re-run before publication. In line with the MLX contribution rules, the pull request was submitted by Xamit Kadirbekov. Xamit Kadirbekov and Daniyal Kadirbekov are the report authors; upstream software remains credited to its developers. Report: CC BY 4.0. Upstream patches retain their upstream licenses.

GERO page: [https://www.gero.uz/research/articles/mlx-categorical-num-samples-zero-probability.html](https://www.gero.uz/research/articles/mlx-categorical-num-samples-zero-probability.html)

_Report CC BY 4.0. Upstream code retains its own licence. Synthetic logits; no measured model-quality or customer-loss claim._
