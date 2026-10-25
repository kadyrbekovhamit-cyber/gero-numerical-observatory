# decimal.js: rounding boundaries in log2 and log10

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO). 25 October 2026.

**Status:** Open upstream PRs; not merged or released. Scoped checks and limitations documented.

Sixteen independently checked references expose early termination at logarithm guard-digit boundaries.

## Source and proposed changes

Base: `b1d1e311eb272b16f5b2ca0e0d1f4c4fa8558edf` in MikeMcl/decimal.js.

- [decimal-log PR #271](https://github.com/MikeMcl/decimal.js/pull/271) — proposed commit `e73eff7aa461bac37c87258af5d88cc89f5b5f54`.

## Technical evidence

This addresses two rounding-boundary cases in `log(base)` as used by `log2` and `log10`, whose API documentation promises correctly rounded results.

A compact example on base `b1d1e311eb272b16f5b2ca0e0d1f4c4fa8558edf`:

```js
const D = Decimal.clone({ precision: 5, rounding: Decimal.ROUND_DOWN });
D.log2("590295810358705651711").toString(); // "69", expected "68.999"
```

The argument is exactly `2^69 - 1`, so its logarithm is strictly less than 69. Rounding toward zero cannot return 69. A default-rounding example is included among the regression fixtures as well.

Two changes are proposed in the CommonJS and ES-module implementations:

- During repeated precision extension, inspect the subsequent guard window for a run of nines/zeros. The initial rounding digit is no longer in that window, so using the original half-rounding mode can stop refinement too soon.
- For base 2, distinguish exact integer powers of two from non-terminating logarithms before final rounding. The power check rejects implausible exponents from input precision, then performs an exact comparison and restores constructor settings.

This is separate from #261, which changes the repeating branch used by natural logarithm/exponential. That work is acknowledged; no claim is made to its `ln()` fix.

Validation: all 10 added `log2` and 6 added `log10` fixtures fail before the change and pass after. Expected results independently agree with Python Decimal calculations at both 300 and 450 significant digits. The full existing decimal.js suite passes (the `random` test changes the total between runs; exact output is retained in our evidence). This does not establish correctness for every possible input or rounding mode.

The local full suite recorded 22,644 passing checks. The total can vary because its random test generates cases. Python Decimal at 300 and 450 digits independently agrees on all 16 expected results.

## Reproduction and boundaries

The archive contains exact public commit patches, upstream licenses, and selected validation metadata. Check out the pinned base, apply each proposal independently, install the project’s declared test dependencies, and run the affected tests described above. The metadata records the actual narrower local environment; a full clean build or unlisted tests are not implied. Existing local paths in logs are descriptive, not required paths.

Novelty checking covered related open/closed GitHub issues and PRs and relevant source history on 25 October 2026. Related reports are acknowledged above; an exhaustive search of every community channel or first-discovery priority is not claimed. No real customer records were used, no deployed product was tested, and no monetary or clinical loss was measured.

## Attribution and licensing

Claude assisted the initial investigation and patch preparation. OpenAI Codex independently repeated the scoped before/after checks, checked the stated reference calculations, and helped prepare the proposed changes and this report. Research/report authors are Xamit Kadirbekov and Daniyal Kadirbekov. Independent human code or legal review is not claimed. Upstream projects retain credit for their software. Report: CC BY 4.0; original reproduction code: MIT; patches retain the upstream licenses.
