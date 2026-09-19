# Same Itô integral, fewer operations: a reproducible computational study

Xamit Kadirbekov · GERO · 19 September 2026 · Version 1.0

**Status:** computational research note about known methods, with proofs, executable code and negative controls. Not peer reviewed. No new stochastic integral, correction to Itô's formula, improved market-pricing formula, or newly discovered implementation defect is claimed. Analysis and code review used separate AI-agent contexts; this is internal checking, not external peer review.

## What was tested

Can algebraic reorganization reduce the work of computing the same stochastic estimate, without changing its information or mathematical error? The primary task is reconstructing the realized integral

\[
I_f=\int_0^T f(W_t)\,dW_t
\]

from the same deterministic observations of a standard one-dimensional Brownian motion, with W₀=0. We study fixed polynomial integrands. Estimating expectations and solving general stochastic differential equations are different tasks; the option experiment below is only an expectation control.

This is reconstruction after an interval has been observed. Using both endpoints is legitimate here and does not make the estimate a predictable trading position at the start of that interval. Adaptive observation rules were not tested.

## A known estimator, two algebraic forms

For f(x)=x², Itô's formula gives I=W_T³/3−∫W_t dt. Conditional on the observed grid, each Brownian bridge has zero mean. Therefore the optimal endpoint estimate is

\[
J=E[I\mid W_{t_0},\ldots,W_{t_n}]
=\frac{W_T^3}{3}-\frac12\sum_i h_i(W_{t_i}+W_{t_{i+1}}).
\]

Writing x_i=W_{t_i}, d_i=W_{t_{i+1}}−x_i gives the identical local expression

\[
J=\sum_i\left[x_i^2d_i+x_i(d_i^2-h_i)+d_i^3/3-h_i d_i/2\right].
\]

The cubic differences telescope. This is an identity for each input array, not a convergence heuristic. The simplified implementation removes repeated local products while retaining a linear pass over observations.

The residual I−J is minus the sum of independent bridge areas. Each has variance h_i³/12, so

\[
E|I-J|^2=\frac1{12}\sum_i h_i^3.
\]

On a uniform grid this equals T³/(12n²). The left-point sum has MSE T³/n, and the local Milstein estimate has MSE T³/n². For Milstein, the local remainder is ∫₀ʰB_s²dB_s; Itô isometry gives variance h³. These accuracy differences belong to known estimators. Rewriting J does not improve its MSE.

The information limit is exact: for every square-integrable estimate Z based on the same observations,

\[
E|I-Z|^2=E|I-J|^2+E|J-Z|^2.
\]

Consequently no reorganization can beat the exact conditional expectation in this error metric without changing the information. Among deterministic grids with n intervals, Jensen's inequality makes the uniform grid optimal for this particular integral. This is not a lower bound for every adaptive problem.

## Reproducible measurements and costs

The main standard-library Python benchmark uses seeds 17,43,101,271,577, 256 paths each, and nested grids n=8,16,32,64: 1,280 coupled paths, 13 methods and 52 accuracy rows. Exact fine-grid bridge areas are used only by the reference to sample ∫W dt jointly with the endpoints. No candidate receives these hidden areas. Thus the reference has no time-quadrature bias; PRNG and floating-point limitations remain.

Five deterministic identity checks passed. The largest absolute difference between the two forms of J was 2.67×10⁻¹⁵. At T=1,n=64, their common observed MSE was 2.02234×10⁻⁵, versus the theoretical 2.03451×10⁻⁵. An approximate interval from five seed-level means was [1.85768×10⁻⁵,2.18699×10⁻⁵]. These are descriptive t intervals, not guaranteed finite-sample coverage. One GBM/Milstein diagnostic at n=16 missed its approximate interval and was retained.

Timing uses 64 frozen paths, three evaluations per path, five repetitions with alternating method order. Median paired kernel speed ratios were 2.79,3.21,3.55,3.77 across the four grid sizes. At n=64, the two median batch times were 1.904 and 0.520 milliseconds. They are local observations about these Python implementations, not superiority over the fastest known implementation.

The apparent advantage shrinks when common data preparation is included: the ratio of accounted full benchmark costs at n=64 was about 1.45. Those costs sum separately measured components; they are not contiguous deployment end-to-end timings. They include validation-only bridge generation. Exact terminal controls are also charged shared grid extraction that an optimized standalone implementation would not need. The saved time/error frontier describes this harness only.

Both J implementations use O(n) reads and O(1) additional working storage. Source-level arithmetic changes from 7n multiplications,2n divisions,6n additions/subtractions to 3 multiplications,2 divisions,2n+1 additions/subtractions in the measured trapezoid loop. Both had 296 bytes of peak temporary Python allocations on the timing subset, excluding inputs. Tracemalloc is not total process memory.

All numerical experiments ran sequentially, one configured local computational worker, no GPU or multithreaded numerical library.

## Batching: useful reuse, with a losing case

For polynomial left sums L_j=Σ_i f_j(W_i)ΔW_i and f_j(x)=Σ_k c_jk x^k, ordinary linearity gives

\[
L_j=\sum_k c_{jk}M_k,\qquad M_k=\sum_i W_i^k\Delta W_i.
\]

Shared moments cost O(nD+mD) arithmetic for m degree-D queries, versus O(mnD) repeated Horner passes. This preserves the same discrete left sums; it does not reduce their stochastic discretization error. Moment aggregation is a known method.

The separate batch experiment has six scenarios, five paths per scenario, five timing repetitions. It compares direct Horner, a shared feature table, and compensated streamed moment accumulation. All cache preparation is included. It is **not** a benchmark of the more expensive conditional-projector batching construction in the analytical notes.

| Grid,degree | Queries | Kernel direct/cache | Including common input preparation |
|---|---:|---:|---:|
| 64,4 | 1 | 0.324 | 0.541 |
| 64,4 | 8 | 2.313 | 1.861 |
| 64,4 | 32 | 8.055 | 4.849 |
| 256,8 | 1 | 0.277 | 0.442 |
| 256,8 | 8 | 1.968 | 1.755 |
| 256,8 | 32 | 7.509 | 5.794 |

Values below one mean the cache is slower. The baseline copies a coefficient slice in every Horner evaluation; optimizing it can change the measured ratios. This experiment does not beat the nearest known moment-aggregation algorithm: that algorithm is the candidate itself. Raw timings and approximate seed-level intervals are included.

The maximum absolute difference among all pairs on ordinary test inputs was 2.22×10⁻¹⁶. That does **not** establish universal relative stability. For the deterministic stress polynomial (x−1)^8 near one, a 70-digit reference is approximately 8.2458×10⁻²⁹; the monomial cache gives −2.1034×10⁻¹⁷, with relative error about 2.55×10¹¹. The known factored representation has relative error about 1.69×10⁻¹⁷. This is a rounding stress test, not a Brownian sample. Compensated summation does not repair an ill-conditioned basis.

## Extra information and historical priority

Observing true bridge areas can improve polynomial reconstruction, but changes the information budget. For ∫W³dW on a uniform grid, endpoint MSE is T⁴(3n−1)/(8n³); with true areas it is 7T⁴/(400n³). The conditional parabola and central time-integral formulas already appear in Foster–Lyons–Oberhauser (2020), Theorems 1.2,1.3,3.7. We include six exact rational checks, not a novelty claim.

If the hidden true area is replaced by an independent sample, the reconstruction MSE becomes 2V₀−V₁≥V₀. Sampling a correct conditional distribution does not recover the realized hidden remainder of the original path.

Selected Doeblin manuscript sections concern random time, changes of coordinates and the Kolmogorov generator. They do not supply a documented ready-made numerical replacement for Itô integration. The entire manuscript corpus was not reviewed.

## Black–Scholes control, including the unfavorable result

For S=K=100,r=0.05,σ=0.2,T=1, the analytical European call is 10.450583572185565. Exact terminal GBM simulation removes time discretization error. Plain and antithetic Monte Carlo use 256 payoff evaluations per seed; antithetic uses 128 independent pairs rather than 256 independent normals, so equal payoff counts are not identical full costs.

Across five seeds, mean prices were 10.1288 and10.0238. Estimated within-seed price variance decreased from0.7824 to0.4218, but the observed MSE of five price estimates **increased** from0.5223 to0.7195. The approximate paired squared-error-difference interval [−0.2806,0.6750] includes zero. We do not replace this unfavorable sample outcome with a claim of empirical superiority.

This does not contradict the known variance-reduction property of antithetic sampling for a monotone payoff. No market data were used. The exact BSM formula remains unchanged; the study establishes no advantage for market prices or general stochastic-volatility models.

## Reproduce and interpret

The accompanying archive contains the original standard-library code, saved JSON outputs, source hashes, Russian detailed proofs and internal critic/auditor notes. Run sequentially from its root:

```sh
python3 programme/benchmark.py
python3 programme/batch_moments.py
python3 area_enriched_check.py
```

Results distinguish identities, theoretical errors, observed timings, hypotheses and refuted universal claims. PASS concerns finite checks, not mathematical novelty, stable relative accuracy on every input, or guaranteed statistical superiority. The next justified investigation is centered/scaled polynomial batching, including coefficient conversion cost and comparison with an optimized known moment cache at a prespecified absolute-error target.

## Primary references

1. Itô (1944), *Stochastic Integral*, §2 and §4. https://www.jstage.jst.go.jp/article/pjab1912/20/8/20_8_519/_pdf
2. Milstein (1974), *Approximate integration of stochastic differential equations*, §§2–3. https://www.mathnet.ru/eng/tvp2929
3. Foster, Lyons, Oberhauser (2020), *An optimal polynomial approximation of Brownian motion*, Theorems1.2,1.3,3.7. https://arxiv.org/abs/1904.06998v3
4. Glynn, Szechtman (2002), conditional Monte Carlo and control variates, §2,§5. https://web.stanford.edu/~glynn/papers/2002/GSzechtman02.pdf
5. Giles (2008), *Multilevel Monte Carlo Path Simulation*, Theorem3.1. https://people.maths.ox.ac.uk/gilesm/files/OPRE_2008.pdf
6. Müller-Gronbach (2004), discrete information and adaptive approximation, §§2–3. https://arxiv.org/pdf/math/0503531
7. Doeblin, published transcription (2000), VIII–IX,XV,XVII. https://djalil.chafai.net/docs/M2/history-brownian-motion/CRAS%20Doeblin%20-%202000%20-%20French/Doeblin-Partie-II-Equation-de-Kolmogoroff.pdf

External papers are referenced, not bundled or relicensed. GERO-authored text and figures: CC BY4.0. Original code: MIT. Separate notices accompany the package. Video and verified publication links are supplied with the release metadata.

## Release links

- [github](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/ito-computation-reorganization)
- [gero](https://www.gero.uz/research/articles/ito-computation-reorganization.html)
- [linkedin](https://www.linkedin.com/feed/update/urn:li:share:7507110937569419264/)
- [43-second English video](https://www.gero.uz/research/media/ito-computation-reorganization.mp4)
- [Evidence archive](https://raw.githubusercontent.com/kadyrbekovhamit-cyber/gero-numerical-observatory/refs/heads/main/reports/ito-computation-reorganization/gero-ito-computation-evidence-v1-2026-09-19.zip)

Additional mirrors pending: youtube, zenodo, huggingface, reddit.
