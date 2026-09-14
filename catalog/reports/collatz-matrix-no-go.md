> Archival mirror. [Original report](https://github.com/kadyrbekovhamit-cyber/collatz-matrix-no-go/blob/main/README.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# Collatz matrix-interpretation no-go artifact

Version 1.0.1 artifact DOI:
[10.5281/zenodo.22098492](https://doi.org/10.5281/zenodo.22098492).

## Publication package

The repository now also contains the manuscript-facing artifact for the
unbounded low-dimensional matrix-interpretation result:

- `submission/main.tex` — English Journal of Automated Reasoning manuscript;
- `submission/JAR_CHECKLIST.md` — current submission requirements and open items;
- `ARTIFACT.md` — exact reproduction commands and trust boundary;
- `MATRIX_ATTACK.md` — full derivation and discovery record;
- `tests/test_matrix_structural.py` — exact structural-certificate regression tests;
- `tools/verify_relative_cubic.cpp` — independent bounded C++ enumerator.

Run `make test-fast` for the manuscript-facing exact tests.  The result is
computer-assisted, not a Lean/Coq formalization: universal coverage is proved
algebraically in the manuscript, while code reconstructs and checks the local
integer certificates.

This repository implements a deliberately finite and auditable experiment for
the accelerated Collatz map

\[
T(n)=\frac{3n+1}{2^{v_2(3n+1)}}\qquad(n\text{ positive and odd}).
\]

It does **not** claim to prove the Collatz conjecture.  It proves a collection
of universal statements about residue classes and leaves every class without a
valid certificate explicitly unresolved.

## Reproduce

```bash
python3 -m unittest discover -s tests -v
python3 -m sag_collatz.experiment --k 20 --limit 63 --output-dir artifacts
python3 -m sag_collatz.verifier \
  artifacts/collatz-k20-l63.bin \
  artifacts/collatz-k20-l63.json
python3 -m sag_collatz.stress \
  artifacts/collatz-k20-l63.bin \
  artifacts/collatz-k20-l63.json
python3 -m sag_collatz.frontier --k 26 --limit 63 --output-dir artifacts
python3 -m sag_collatz.frontier_verifier \
  artifacts/open-frontier-k26-l63.bin \
  artifacts/open-frontier-k26-l63.json
python3 -m sag_collatz.global_attack
python3 -m sag_collatz.matrix_attack
python3 -m sag_collatz.matrix_attack --matrix-entry-max 2 --relative
```

The generator and verifier are separate implementations.  A certificate byte
encodes only the first descent step and proof mode; the verifier reconstructs
the full exact arithmetic for every odd residue.

See `REPORT.md` for the theorem, derivation, results, and limitations.  The
file `CLAUDE_AUDIT_PROMPT.md` is a ready-made adversarial review request.
The completed independent verdict is recorded in `EXTERNAL_AUDIT.md`.
`METHOD_LIMIT.md` proves why no bounded-horizon first-descent graph can settle
the full conjecture.
The independently checked depth-26 extension is summarized in
`FRONTIER_REPORT.md`.
`GLOBAL_ATTACK.md` tests the next natural proof strategy, gives exact
counterexamples to local binary ranking functions, and specifies what a
genuinely global certificate would still have to do.

The global attack now also proves a sharp boundary for factor-map searches:
every finite-image factor fails, but asking merely for a noninjective
finite-state factor with a well-founded quotient is already equivalent to the
Collatz conjecture—even if the factor has an infinite fiber.  Exact quotient
cycle certificates refute all boundary-aware local substring-count maps of
radii 1 through 6.  These are no-go theorems for specified proof languages,
not a proof of Collatz.

`MATRIX_ATTACK.md` gives an exact symbolic no-go theorem for direct
one-dimensional natural affine interpretations of the 11-rule mixed-base
Collatz rewriting system.  It also gives an exhaustive integer-certified
no-go result for dimension 2 with Boolean symbol matrices and unbounded natural
offset vectors.  The exact exhaustive certificate is extended to every matrix
entry in `{0,1,2}`, including upper-left entries 1 and 2 (over 1.3 trillion raw
assignments, pruned to 1,169,880 survivors and closed by 14 integer Farkas
templates).  The original eight templates already refute an eight-rule
weakening obtained by deleting `carry-t0`, `carry-t1`, and `left-1`.  A second
exact check uses the rule-removal formulation that is actually sufficient for
Collatz: only the two dynamic rules are strict and the nine terminating
auxiliary rules are weak.  Scalar affine interpretations are ruled out
symbolically, and the full two-dimensional domain in which every matrix entry
is in `{0,1,2}` and the upper-left entry is positive is exhausted by 42 exact
integer Farkas templates (1,169,880 surviving coefficient assignments and
unbounded natural offsets).  The relative no-go is further extended through
coefficient 3 with upper-left entry 1 or 2: exact enumeration prunes
128^7 raw assignments to 33,099,480 survivors, all closed by 96 exact integer
templates.  The complete coefficient cube `{0,1,2,3}` is now exhausted as
well: 192^7 raw assignments prune to 72,169,932 survivors; eleven exact
parameterized Farkas constructors cover 72,165,108 of them, and a twelfth
structural constructor closes the former residual 4,824.  The 103 fixed
integer templates are retained as a regression corpus but are no longer used
on that cube.  A widened slice in which only the left-marker entries may reach
4 has 130,631,724 survivors; the same twelfth formula closes all 44,064 cases
left by the first eleven.  Entries above 3 are not enumerated; they are covered
instead by the structural theorem below.  Higher dimensions and other staged
variants remain outside the theorem's scope.
Beyond those finite boxes, all twelve parametric Farkas applicability lemmas
are now proved with no
coefficient ceiling.  The first closes `ell M_f >= ell`; the next three give
exact branches inside the strict deficit region, including an unbounded integer
parameter solved by interval arithmetic, and family 5 has an exact reduced
characterization on the same complement.  Families 6--12 likewise have exact
reduced conditions rather than bounded searches.  Perron--Frobenius
rigidity now closes all four zero patterns with `f00>=2` in the strict deficit
region.  The fully off-diagonal pattern is in fact impossible already for
`f00>=1`: summing the carry slacks makes every carry rule an equality, and a
rank split on `M_2` forces incompatible bounds `delta>=lambda^2` and
`delta<=lambda`.  Hence every remaining assignment has `f00=1`, at least one
of `f01,f10` zero, and therefore `M_f^2=M_f`.  The three idempotent forms are
then covered symbolically: the lower form by family 5, the projection by
families 2--6, and the upper form by families 4 and 12.  This proves the
joint unbounded completeness of the twelve families.
The exact unsimplified applicability predicates for all fixed-weight families,
including family 12, are recorded in `families-conditions.md`.
An exact constructor tests all twelve parametric rows on arbitrary natural
coefficients, including an unbounded integer parameter solved by interval
arithmetic.  The resulting theorem rules out the complete one-stage relative
natural affine proof language in dimension 2.  It does not rule out higher
dimensions or other termination methods, and it does not prove Collatz.
