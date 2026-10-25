# ethers: negative FixedNumber rounding can change integral values

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO). 25 October 2026.

**Status:** Open upstream PRs; not merged or released. Scoped checks and limitations documented.

A proposed correction to signed quotient rounding, with 16 regressions and 104 related checks.

## Source and proposed changes

Base: `3ea4c226dd0b3a074abf90748de9d11192027f4f` in ethers-io/ethers.js.

- [ethers PR #5207](https://github.com/ethers-io/ethers.js/pull/5207) — proposed commit `b4b6bb03d6b836c66f5f9cbf19416030640091be`.

## Technical evidence

`FixedNumber.round()` adds the half-unit before dividing by a power of ten. BigInt division truncates toward zero, so a negative adjusted value is rounded in the wrong direction. This can even change an already integral value:

```js
FixedNumber.fromString("-3").round().toString();    // current: "-2.0", expected: "-3.0"
FixedNumber.fromString("-1.7").round().toString();  // current: "-1.0", expected: "-2.0"
FixedNumber.fromString("-1.2").round().toString();  // current: "0.0", expected: "-1.0"
```

The patch floors the adjusted quotient when division has a negative remainder. It preserves the documented tie direction: `-1.5` rounds to `-1.0`, just as `1.5` rounds to `2.0`. It does not change `floor()` or `ceiling()`; those are separately addressed by #5160. Older #1629/#1749 concern format compatibility, not this arithmetic.

Six of the 16 new regression cases fail on base `3ea4c226dd0b3a074abf90748de9d11192027f4f`; all 16 pass after the patch. The regression, units and maths modules pass 104 tests together. Tests execute JavaScript emitted from the modified TypeScript with the existing TypeScript runtime (ES2022 target); a full project typecheck/build and network-dependent suites were not run. Generated library files are not included in the PR.

No wallet, deployed contract or customer loss is asserted; this report concerns the public `FixedNumber` API.

## Reproduction and boundaries

The archive contains exact public commit patches, upstream licenses, and selected validation metadata. Check out the pinned base, apply each proposal independently, install the project’s declared test dependencies, and run the affected tests described above. The metadata records the actual narrower local environment; a full clean build or unlisted tests are not implied. Existing local paths in logs are descriptive, not required paths.

Novelty checking covered related open/closed GitHub issues and PRs and relevant source history on 25 October 2026. Related reports are acknowledged above; an exhaustive search of every community channel or first-discovery priority is not claimed. No real customer records were used, no deployed product was tested, and no monetary or clinical loss was measured.

## Attribution and licensing

Claude assisted the initial investigation and patch preparation. OpenAI Codex independently repeated the scoped before/after checks, checked the stated reference calculations, and helped prepare the proposed changes and this report. Research/report authors are Xamit Kadirbekov and Daniyal Kadirbekov. Independent human code or legal review is not claimed. Upstream projects retain credit for their software. Report: CC BY 4.0; original reproduction code: MIT; patches retain the upstream licenses.
