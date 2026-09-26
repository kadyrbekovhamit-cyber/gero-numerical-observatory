# Positive-payoff quadrature for Black–Scholes: a frozen binary64 prototype

**Khamit Kadyrbekov and Daniyal Kadirbekov** · Working paper v2.0 · 26 September 2026.

Research on numerical evaluation within the Black–Scholes–Merton model.
This release does not alter the financial model or establish improved market prediction.

## Read and reproduce

- [Full working paper](TECHNICAL_NOTE_V2.md) and [PDF](black-scholes-quadrature-v2.pdf).
- [Russian summary](SUMMARY_RU_V2.md).
- [Frozen protocol](EXPERIMENT_V2.md), [all 288 confirmation rows](evidence/benchmark-v2.json),
  [timing](evidence/python-timing-v2.json), [candidate source](lab/black_scholes_v2.py).
- [Earlier negative result](TECHNICAL_NOTE_V1.md), preserved development attempts under evidence/.

Among 202 nonzero reference prices, v2 is closer than our v0 in 192 cases,
farther in 3, equal in 7. Against a historical Jäckel source: 146/30/26.
All 288 registered practical gates pass. The Python v2 API is about 15.4 times
slower than v0 on a separate timing subset; native C++ speed was not measured.
Synthetic selected inputs, empirical high-precision reference, no universal guarantee.

Python 3.9+, mpmath==1.3.0 and clang++ are needed. From this folder:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python3 -m lab.build_jackel
python3 -m unittest discover -s tests -v
python3 -m lab.run_benchmark_v2 --output /tmp/black-scholes-v2-reproduction.json
```

Use a new output path. The build records the local native binary, whose hash
may differ by compiler/platform. Source/protocol/corpus hashes stay frozen.
The runtime is sequential on CPU. There is no GPU workload or production-trading integration.

## Licensing

Original text/data: CC BY 4.0, see LICENSE-TEXT.md. Original lab/tests: MIT,
see LICENSE-CODE.txt. Third-party Jäckel/Cody files retain their original notices
and terms; see vendor/jackel/PROVENANCE.md and file headers. No blanket relicensing
of third-party code is intended.

## Authors and research transparency

Khamit Kadyrbekov and Daniyal Kadirbekov. Independent research, Tashkent, Uzbekistan.
Correspondence: kadirbekov@gmail.com. Software contact: daniyal.kadirbekov@gmail.com.

Khamit initiated and directed the research programme. Daniyal is a coauthor and
the designated lead for its continuing software development. That prospective
responsibility does not assert that he independently implemented or validated
the frozen v2 code. The recorded experiment was prepared and executed with
OpenAI Codex assistance, including mathematical analysis, implementation,
test generation, result checking, drafting and editorial review. AI-role
reviews are not human peer review. The work is an interim, non-peer-reviewed
computational study; independent reproduction and human technical review are
invited. No academic or financial institution endorsement is claimed.

