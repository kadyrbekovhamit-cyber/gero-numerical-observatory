# Zero observed false positives is not a zero false-positive rate

This note reproduces a narrow statistical fact used by GERO when evaluating
small validation studies. It does not evaluate or name any third-party product.

If a validation sample contains `n` independent, representative negative cases
and observes zero false positives, the point estimate is `0/n = 0`. The point
estimate is not an uncertainty statement.

For zero observed failures, the one-sided exact 95% upper confidence bound is

```text
p_U = 1 - 0.05^(1/n).
```

At `n = 2`, this upper bound is `0.776393`: data compatible with zero observed
failures still permit a very large underlying rate. The upper endpoint of a
two-sided 95% Wilson score interval is `0.657620`. These values differ because
the intervals use different constructions and sidedness; both must be labelled.

With zero observed failures, at least 59 representative negative cases are
needed before the one-sided exact 95% upper bound falls to 5%, and 299 are needed
before it falls to 1%.

Run the calculation:

```bash
python3 tools/zero_failure_bounds.py
```

Run the tests:

```bash
python3 -m pytest tests/test_zero_failure_bounds.py -q
```

## Evidence boundary

- The binomial calculation assumes independent, representative cases with a
  stable failure probability.
- Correlation, repeated entities, hand-picked negatives, distribution shift or
  subgroup imbalance can make the effective sample smaller.
- A confidence interval is a property of a repeated procedure. It is not a
  posterior probability that the true rate lies in one realised interval.
- A larger benchmark should report sensitivity, specificity, precision, class
  prevalence, subgroup results and uncertainty, not only a zero-error headline.

## Sources

- NIST/SEMATECH, “Confidence intervals”: https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm
- NIST AI RMF, “AI Risks and Trustworthiness”: https://airc.nist.gov/airmf-resources/airmf/3-sec-characteristics/
- C. J. Clopper and E. S. Pearson (1934), “The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial,” DOI: 10.1093/biomet/26.4.404

