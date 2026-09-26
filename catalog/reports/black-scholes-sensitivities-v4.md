# Black-Scholes beyond price accuracy: model sensitivities and lost volatility information

**Khamit Kadyrbekov and Daniyal Kadirbekov**

Working paper v4.0, 26 September 2026. Numerical research, not peer reviewed.

## Abstract

We audit what the published hybrid Black-Scholes price evaluator does not establish: reliable derivatives and identifiable implied volatility. Sixty exploratory spot stencils separate model curvature, finite-step bias, rounded-price noise and price-kernel error. An exact-ATM simplification illustrates why differentiating a selected formula can return the wrong partial derivative. A separate analytic sensitivity API passes all predeclared mixed value/log and range gates on 384 new locally frozen cases after development on 1,084 disclosed inputs. Finally, six distinct volatilities from 3% to 8% have the same correctly rounded in-the-money call price. This is an engineering and educational extension of established mathematics, not a new financial model or a novelty claim.

## 1. Three different numerical contracts

The existing v3 returns prices and an OTM log price. Its reported accuracy and speed apply to that interface, not automatically to Delta, Gamma or inversion. We preserve its source hash and every published price experiment. The new API computes derivatives of the smooth BSM model directly, separately from the rounded price graph.

Write v = sigma sqrt(T), m = log(S/K) + (r-q)T and d1 = m/v + v/2. For positive S, K, T and sigma, with the other contract inputs held fixed, the standard model identities are:

\[
\Delta_C=e^{-qT}\Phi(d_1),\qquad \Delta_P=-e^{-qT}\Phi(-d_1).
\]

\[
\Gamma=\frac{e^{-qT}\phi(d_1)}{S\sigma\sqrt T},\qquad \mathcal V=S e^{-qT}\phi(d_1)\sqrt T.
\]

Vega is per unit sigma, not per percentage point. Both option sides share Gamma and Vega. The local inverse sensitivity, when a positive finite implied volatility exists, is 1/Vega. It is not a guarantee that a rounded quote identifies that volatility.

## 2. Exact-ATM prices do not define off-surface derivatives

At S=K and r=q=0, C=S erf(v/(2 sqrt(2))). The equality holds on the ATM surface. Extending this selected expression as a function of S while holding its coefficient fixed differentiates the wrong function for spot risk at fixed K.

For S=K=100, T=1 and sigma=0.3, the resulting leaf derivative is 0.1192353847404850 and its second derivative is zero. The actual call Delta is 0.5596176923702425 and Gamma is 0.0131493110302630. Along the path K=S, the chain rule gives dC(S,S)/dS=C_S+C_K; this explains the smaller derivative. Spot Delta instead holds K fixed.

This is an analytic counterexample to blindly porting the special price branch into autodiff. The frozen Python v3 has no native AD interface, and we do not report an executed AD failure in that API or in a third-party library. The new sensitivity API uses the model identities, not derivatives of the ATM branch. Boost's documentation illustrates the established approach of differentiating an unspecialized Black-Scholes expression [1].

## 3. Finite differences need a control

The exploratory audit uses ten contracts and six relative spot steps, from 1e-2 through 1e-12. For each of 60 three-point stencils it records the actual unequal binary64 spot increments. We compare frozen v2/v3 price differences with the analytic derivative, a high-precision price stencil, and the same floating stencil fed correctly rounded oracle prices. This avoids blaming a price kernel for all cancellation introduced by differentiation.

Across the 60 stencils, five v3 Gamma estimates are negative. Six estimates are negative even in the rounded-oracle control. Thirty v3 stencils cross a price-method branch. These counts are descriptive, include deliberately unsuitable steps and are not failure probabilities for market use.

An example near the negative-tail switch has S=100 exp(-4.88), K=100, T=1, r=q=0 and sigma=0.4, with S interpreted as the recorded binary64 input. At relative step 1e-8, v3 Gamma has about 10.83% relative error; the rounded-oracle control has about 0.21%. At step 1e-4 both are near 6.71e-7, mostly finite-step bias. Kernel errors are amplified at tiny steps, but reducing the step is not a universal remedy. All stencils, including unfavorable controls, are in `evidence/sensitivity-audit-v3.json`.

## 4. Separate analytic sensitivity API

`lab.black_scholes_greeks.greeks(S,K,T,r,q,sigma)` returns call Delta, put Delta, Gamma, Vega, inverse Vega and five corresponding absolute-value logarithms. Every ordinary output has a normal/subnormal/rounded-zero/overflow tag. The implementation postpones products and quotients through binary exponent decomposition so an intermediate density, reciprocal or cash scale does not decide the final range prematurely.

Normal tail probabilities use an erfc expression centrally and a Mills expansion for arguments at or below -12. The expansion is standard [2]; its omitted-term argument concerns exact-arithmetic truncation, not certified total floating-point error. Logarithms remain useful when the ordinary Greek rounds to zero or overflows. Inverse Vega can overflow legitimately; it is reported rather than capped to an apparently safe number.

The declared domain requires positive T and sigma, representable positive total volatility and finite internal logarithmic arithmetic. Zero maturity, zero volatility, underflow of positive total volatility and unsupported exponent arithmetic raise explicit exceptions. Boundary sensitivities need a separate mathematical contract. The implementation has no oracle fallback and does not modify the v2/v3 price engines.

## 5. Frozen confirmation and exact denominators

All 1,084 disclosed v1/v2/v3 price fixtures pass the development gates. Thirty-four unit-test methods pass, including six new methods with multiple fixtures. Three fixtures also differentiate an unspecialized multiprecision BSM price expression. The new candidate, protocol, oracle, runner, tests, generator, inputs and local dependency/runtime hashes were fixed before the new outputs. This is a local freeze, not an external preregistration or a hermetic operating-system image.

The confirmation consists of 384 new exact-binary64 input tuples: 96 regular, 96 price-branch neighborhoods, 96 tail/monetary/discount range, 48 exact-ATM range and 48 normal-tail switch cases. Family labels describe how inputs were constructed; they do not guarantee a particular rounded branch or exhaustive two-sided coverage of every seam.

All **384/384 cases** pass the ten output gates and range checks. All 384 primary references converge at 100/180 decimal digits. For finite ordinary values, the absolute budget is max(2e-11 times the exact magnitude, one ULP of its rounded value), plus correct range classification and no unexpected zero. For each logarithm the budget is max(2e-11, 32 ULP of the reference log). These are mixed gates, not universal one-ULP or relative-error guarantees.

| Output | Normal | Subnormal | Rounded zero | Overflow |
|---|---:|---:|---:|---:|
| Call Delta | 356 | 1 | 27 | 0 |
| Put Delta | 349 | 1 | 34 | 0 |
| Gamma | 309 | 1 | 68 | 6 |
| Vega | 318 | 3 | 63 | 0 |
| Inverse Vega | 318 | 0 | 0 | 66 |

The maximum normal relative errors are approximately 1.91e-13 for call Delta, 2.73e-13 for put Delta, 3.09e-13 for Gamma, and 2.73e-13 for both Vega and inverse Vega. The largest absolute log error in the new corpus is 3.41e-12. These maxima do not describe every possible binary64 input. In the older development corpus, enormous log magnitudes make absolute log errors much larger while still meeting its ULP-scaled budget.

## 6. A correctly rounded price may have no unique inverse

Consider S=200, K=100, T=1 and r=q=0. The intrinsic call value is exactly 100. For each of sigma=0.03, 0.04, 0.05, 0.06, 0.07 and 0.08, the correctly rounded call and the v3 call are exactly the same binary64 number 100. The positive put/time value and its logarithm distinguish the cases; the rounded ITM call alone has discarded them.

The next larger float is separated from 100 by 2^-46. Consequently time values smaller than 2^-47 are absorbed by rounding to 100. High-precision bisection estimates the volatility transition near 0.0894076873552444, or 8.94% annualized in this one-year example. The recorded estimate is illustrative, not a certified interval endpoint. The six explicit equal-price examples suffice to demonstrate non-injectivity.

More generally, let I(p) be the rounding cell or explicit quote-uncertainty interval for price p. Since a positive-volatility BSM price is strictly increasing in sigma, the compatible set is:

\[
\Sigma(p)=\{\sigma\ge0:P(\sigma)\in I(p)\}.
\]

After intersection with the model's no-arbitrage price range, this set is an interval, possibly empty, unbounded or touching zero. A small repricing residual cannot turn a wide compatible interval into a unique accurate sigma. Likewise, replacing Vega by an arbitrary floor changes a gradient policy; it does not recover lost information. Related work already treats analytic inverse differentiation and low-Vega gating, including PIVOT [3]. Our tests do not reproduce its GPU or market-data results.

## 7. Reproduction, attribution and remaining work

Use Python 3.9+ and mpmath 1.3.0. `python3 reproduce_greeks_v1.py --output /tmp/greeks-replay.json` checks frozen source hashes and replays the 384 rows into a fresh file. The portable harness was added after the original results and changes only verification of local paths/runtime; it does not change the numeric implementation, reference, inputs or metrics. The release includes raw rows, tests, protocol and original freeze receipt. No native comparator build is needed for the Greek experiment.

Khamit Kadyrbekov directs the programme. Daniyal Kadirbekov is coauthor and designated lead for continuing software development; this does not assert independent implementation or validation of the frozen prototype by him. OpenAI Codex materially assisted derivation, implementation, experiment design and execution, analysis and drafting. No independent human peer review, current-library superiority, production readiness, market-accuracy improvement, trading return or university endorsement is claimed.

Next work is an explicit quote-interval IV interface, derivative behavior at mathematical boundaries, comparisons with pinned specialist Greek implementations and independent reproduction on another machine. Original price-engine performance evidence remains limited to its earlier experiments. All local runs used one sequential numerical CPU worker, no GPU.

## References

[1] Boost.Math, automatic differentiation documentation, Black-Scholes examples. Accessed 26 September 2026: https://www.boost.org/doc/libs/latest/libs/math/doc/html/math_toolkit/autodiff0.html

[2] NIST Digital Library of Mathematical Functions, section 7.12(i), complementary-error-function asymptotics and remainder bounds. Accessed 26 September 2026: https://dlmf.nist.gov/7.12

[3] R. Saqur, Y. Limmer, A. Kratsios, B. Horvath and H. Buehler, PIVOT: Bridging Black-Scholes Implied-Volatility and Price Objectives via Differentiable Jaeckel Operator, arXiv:2606.17065v1, 4 June 2026. Abstract and relevant differentiable-layer/conditioning sections consulted; experiments not reproduced: https://arxiv.org/abs/2606.17065
