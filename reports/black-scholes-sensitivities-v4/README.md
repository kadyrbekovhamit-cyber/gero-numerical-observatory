# Black-Scholes sensitivities and information loss, v4.0

**Khamit Kadyrbekov and Daniyal Kadirbekov**, 26 September 2026.
Working paper: https://doi.org/10.5281/zenodo.22976212
Preceding price version: https://doi.org/10.5281/zenodo.22973900

Read TECHNICAL_NOTE_V4.md (or its PDF), SUMMARY_RU_V4.md and REVIEW_GUIDE.md.
This focused release adds a separate analytic Greek API and two diagnostic
examples. Standard BSM mathematics; no new market model, novelty, trading
return, universal error guarantee or independent human review is claimed.

## Results and limits

- New locally frozen confirmation: **384/384** exact-input cases pass the
  declared mixed value/log/range gates; all primary references converge at
  100/180 digits. Development: 1084 old cases; unit tests: 34 methods.
- The five ordinary outputs are call Delta, put Delta, Gamma, Vega and inverse
  Vega. Vega is per unit sigma. Five corresponding logs remain available
  when ordinary output rounds to zero or overflows. Explicit domain failures
  cover zero time/volatility and unsupported internal arithmetic.
- Sixty exploratory finite-difference stencils include five negative v3 Gamma
  estimates and six in the rounded-oracle control. Tiny steps deliberately
  amplify rounding; these are not market failure probabilities.
- An ATM-specialized formula has the wrong derivative if extended off its
  validity surface. This is an analytic warning for future AD ports, not an
  executed AD defect in the current Python API or a vendor product.
- Six distinct volatilities yield the same rounded ITM call of 100. The
  approximate transition 0.0894077 is illustrative, not interval-certified.
  No general interval-IV solver has yet been implemented.
- Branch-family construction is not exhaustive two-sided seam coverage.
  No specialist Greek performance comparison is made.

## Reproduce

Python 3.9+ and **mpmath==1.3.0**. No compiled comparator, GPU or network is
needed after dependencies are available. From this extracted directory:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s tests -v
python3 reproduce_greeks_v1.py --output /tmp/greeks-replay.json
python3 reproduce_sensitivity_audit.py --output /tmp/sensitivity-replay.json
```

Use fresh output paths. Compare `rows` and `summary` of the Greek replay with
`evidence/benchmark-greeks-v1.json`. Compare `stencils`, `summary`,
`atm_branch_derivative` and `iv_information_loss` with the exploratory receipt.
Timestamps and harness metadata differ by design. Same-host replay matched;
other-machine/human reproduction remains open.

The post-result portable harnesses check frozen source/input hashes and the
mpmath source tree, replacing only original-machine verification. The original
`lab.corpus_greeks_v1.verify` also references the original price/native/worker
setup; use the portable commands above on another machine. The old freeze
receipt is retained transitively, with original paths, but this focused package
does not include all historical C++ experiments or a native binary. Those
remain available unchanged at the preceding version DOI.

`MANIFEST.json` gives SHA256 for every release file except itself. The original
frozen candidate, corpus, tests and protocol were not changed after confirmation.

Original text/data: CC BY 4.0; original Python code: MIT. See AUTHORSHIP.md for
roles and AI assistance. Reviewers are invited to reproduce and find failures.
