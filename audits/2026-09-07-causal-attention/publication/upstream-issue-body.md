## Issue Description

On main `a7ea056e79ab8e14447ea305c1b634e233343258`, native FP32
`AttentionLayer` with `causal_mask=true` can include future values. A compact
case uses only zero queries/keys, so the expected result needs no approximate
exponential calculation:

```
Q: [1,1,2,1] = [0,0]
K: [1,1,3,1] = [0,0,0]
V: [1,1,3,1] = [2,10,50]
causal_mask=true, scaled_dot_product=false
```

Original output: `[20.66666794,20.66666794]`.
The derivative of the first output with respect to V is `[1/3,1/3,1/3]`.

## Expected Result

With the existing upper-triangular causal convention, query i can use keys j≤i.
The output should be `[2,6]`; the first output's V gradient should be `[1,0,0]`.
Changing future V₂ must not affect either output. For a different intended
rectangular convention, an explicit alignment contract or rejected input would
be preferable to silently running unmasked attention.

Two related failures in the same implementation:

- Square scores, Q=[1,1], K=[0,2e10], V=[2,10]: original first output is 10
  rather than 2 because subtracting 1e10 does not exclude the future logit.
- Q=K=[0,0,0], V=[2,10,50]: full causal output is `[2,6,20.6667]`, while
  incremental ranges [0,1), then [1,3) give `[2,20.6667,20.6667]`.

## How to Reproduce

[Immutable evidence package](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/dbd6a98d7c4213d1df26252d4b5cc053c18c05d3/audits/2026-09-07-causal-attention) · [Native build instructions](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/dbd6a98d7c4213d1df26252d4b5cc053c18c05d3/audits/2026-09-07-causal-attention/evidence/BUILD.md) · [Test patch](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/dbd6a98d7c4213d1df26252d4b5cc053c18c05d3/audits/2026-09-07-causal-attention/evidence/patches/nntrainer-tests.patch) · [Implementation patch](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/dbd6a98d7c4213d1df26252d4b5cc053c18c05d3/audits/2026-09-07-causal-attention/evidence/patches/nntrainer-attention.patch)

Apply the test patch to the pinned checkout. Build the native
layer test target using the commands in BUILD.md, then run from its build root:

```sh
test/unittest/layers/unittest_layers \
  --gtest_filter='AttentionNumericalAudit.*:CausalShapes/*:Attention/*' \
  --gtest_color=no
```

Before the repair: **21 failed / 21 passed**.
Apply `patches/nntrainer-attention.patch`, rebuild and rerun: **42 passed**.
The four smallest cases were also repeated identically in three fresh processes
per version. Full logs, XML and SHA-256 manifests accompany the reproduction pack.

The 28 new tests cover FP64 values and derivatives, finite differences of native
forward, 16 shape/scaling combinations, batch independence, masked probability
support, future perturbations and eight incremental partitions. All 14 existing
attention semantics/golden tests pass before and after.

## Further Information

`forwarding` creates an Nk×Nk mask for Nq×Nk scores. The failed `add_i` return
code is ignored on rectangular shapes. The mask uses a finite additive penalty.
`incremental_forwarding` applies it only when `from==0`.

The proposed local repair overwrites forbidden score entries with negative
infinity, using actual score dimensions and the absolute query offset. It adds
no dependency or parameter and removes the temporary mask allocation. The patch
applies to the exact pinned files and matches the tested source byte for byte.

Environment: macOS 15.5 arm64 / Apple M4; native C++ CPU, FP32, NCHW, channel=1;
BLAS disabled, one thread. Incremental validation uses batch=1. The reused build
contains earlier unrelated loss/activation/LayerNorm changes; the attention
baseline itself matches pinned main. Full clean repository builds, other OSes,
FP16/GPU, performance and model-level impact have not been validated.

Related public work considered: #1713, #2407/#2409, #3780, #4040, #4198 and
#3989. No exact match was found within ten retained searches covering 191 unique
issues/PRs. Please link a duplicate if one is known. This is a local reproduction
and repair proposal, not a claim of maintainer confirmation or version regression.

Prepared with AI assistance. Feedback from the attention-layer maintainers is welcome, especially on rectangular causal alignment and the intended incremental block contract.
