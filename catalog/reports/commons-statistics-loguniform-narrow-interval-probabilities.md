# Apache Commons Statistics: a finite interior probability becomes NaN

Date: 23 September 2026. Independently reproduced numerical-conformance report. A technical message was sent to the official Apache Commons development list before this publication; delivery through list moderation and maintainer acknowledgment are not yet verified. No qualifying bounty or security impact is established.

## Reproduction

Official Maven artifact `org.apache.commons:commons-statistics-distribution:1.3`, with its official direct runtime dependencies, executed through the unmodified public Java API. Maven SHA1 sidecars match all seven artifacts; locally recorded SHA256 hashes are in SOURCE_LEDGER.json. Runtime: existing OpenJDK 26.0.2, interpreted mode, Serial GC, ActiveProcessorCount=1, coordinated through the existing single-worker lock. No network during numerical execution.

```java
LogUniformDistribution d = LogUniformDistribution.of(
    0x1.0p100, 0x1.0000000000002p100);
d.cumulativeProbability(0x1.0000000000001p100); // NaN
d.survivalProbability(0x1.0000000000001p100);   // NaN
```

The lower bound, evaluation point and upper bound are distinct finite positive binary64 values with a finite interval width. Both probabilities should be close to one half. Let t=2^-52. The independent analytic CDF is `ln(1+t)/ln(1+2*t)` = approximately `0.50000000000000005551115123125781469523`; the SF is its complement. The correctly rounded CDF is 0.5. The proposed candidate returns the neighbouring value 0.5000000000000001 for this input: it resolves the invalid result, but is not claimed correctly rounded.

## Cause and current-source check

The constructor subtracts separately rounded `log(upper)` and `log(lower)`. They coincide for the minimal input, producing a zero normalization denominator; CDF/SF then evaluate zero divided by zero. Current master `5d8d9e10d876d9a913a054154146f36a18490824` differs from the release target class only by one Javadoc summary line. The release API was executed; a complete current-master build was not.

## Verification and candidate

The grid contains 177 calls, 176 distinct (lower, upper, x) inputs, spanning exponents -1020 through 900, narrow intervals, endpoints, outside-support points and ordinary/wide controls. Decimal analytic references at 120 and 200 digits agree after conversion to binary64.

At an absolute CDF/SF tolerance of 1e-12, unique failing inputs are **48 -> 0 -> 48** for release/candidate/restored. Unique invalid probability pairs are **30 -> 0 -> 30**. Counting the deliberately repeated minimal input gives 49/0/49 failing calls and 31/0/31 invalid calls. These are one numerical mechanism, not 48 independent defects.

The candidate computes nearby logarithmic ratios with log1p and uses the log difference for an overflowing ratio. Candidate maximum absolute CDF/SF error on this grid is 1.1102230246251565e-16. All 95 endpoint/outside/ordinary control calls pass; one ordinary control improves by roughly one ulp, so bitwise preservation of every control is not claimed. Restoring the original class returns byte-identical original CSV output and original source bytes.

Candidate execution replaces only this class in an otherwise official release runtime. It is a review candidate, not a released fix. The full upstream test suite, inverse probabilities, all moments and sampling have not been validated; the reused normalization field can also affect those methods. Density and mean are recorded diagnostically, not exhaustively assessed.

## Measured consequence and limits

The measured chain is **rounded-log cancellation → invalid probability → a changed CSV/JSON worksheet → an incorrect or blocked decision**. Apache computes the probabilities. The worksheet and its two read-back consumers are explicitly GERO demonstration code.

For each of the 176 unique inputs, a synthetic exposure of 1,000,000 units is multiplied by the survival probability and compared with a 400,000-unit review threshold. The unguarded CSV consumer parses `NaN` as a float and executes `cost > threshold`. A separate guarded JSON consumer encodes invalid numbers as `null` with a numeric-validity marker and blocks them before comparison.

| Variant | Invalid probabilities | Wrong unguarded decisions | Guarded calculations blocked |
|---|---:|---:|---:|
| Official release 1.3 | 30 | 30 | 30 |
| Local candidate | 0 | 0 | 0 |
| Restored original | 30 | 30 | 30 |

For the minimal input, the reference worksheet amount rounds to **500,000.00 units**, requiring `REVIEW`. Release/restored return `NaN`, so the unguarded consumer chooses `WITHIN_LIMIT`; the guarded consumer returns `BLOCKED_INVALID_NUMBER`. The candidate returns SF=0.49999999999999994, giving 499999.99999999994 and the correct `REVIEW` branch. Every originally finite result gives the expected decision in this particular threshold experiment. Original/restored CSV and JSON documents are byte-identical.

This measures a possible downstream consequence under a declared application choice, not actual deployment, a real invoice, customer loss, a denied claim, or observed incidence. **500,000 is a hypothetical calculated amount, not a measured financial loss.** The 30 failures are a deliberately selected stress grid, not a frequency estimate. Checking finiteness prevents the wrong decision but leaves those computations unavailable. Candidate validation remains limited to this CDF/SF grid.

## Duplicate review and reporting

A bounded fresh review covers target-class history (three commits), STATISTICS-88 and both comments, the initial implementation commit, JIRA uniform/numerical searches, GitHub exact/broad searches and PR54's diff, plus the current canonical GERO catalogue. No exact prior report or proposed fix was found in those reviewed sources. STATISTICS-88 introduced the feature; PR54 concerns unrelated descriptive-statistics Javadocs. This does not establish universal priority or rule out unindexed reports.

The 23 September pre-send refresh confirmed the same master commit and reviewed STATISTICS-96's body: it concerns nonfinite constructor parameters, whereas this report uses finite ordered parameters. A separate mailing-list archive search failed at the network layer. No exact prior report or fix was found in the sources actually reviewed; global novelty is not proved.

A single technical email was sent at 08:50 Asia/Tashkent on 23 September to `dev@commons.apache.org`, with a minimal reproducer and measured limits. Gmail's sent confirmation and sent-folder copy were checked. No maintainer acknowledgment, accepted patch, Jira ticket or award is claimed. The official list disallows attachments, so the initial message included inline reproduction and offered the packet. Apache states that it has no own bug bounty program; this is an ordinary developer report, not an unpaid debt or a promised reward.

Sources: https://commons.apache.org/proper/commons-statistics/commons-statistics-distribution/apidocs/org/apache/commons/statistics/distribution/LogUniformDistribution.html ; https://commons.apache.org/proper/commons-statistics/developers.html ; https://issues.apache.org/jira/browse/STATISTICS-88 . Exact fetched URLs and hashes are in SOURCE_LEDGER.json.

## Local evidence

The frozen packet contains `reproducer/` (official artifacts, source, candidate and scripts), `expected/` (raw outputs, independent references, worksheets and receipts), and `review/` (bounded duplicate/source checks). A fresh temporary-directory replay of the portable packet completed successfully before publication, rerunning release, candidate, restoration and document consumers without network calls. Set `GERO_JAVA_HOME` to your JDK home and follow the archive README; Python 3.11+ with standard library and JDK 11+ are required. The recorded execution used OpenJDK 26.0.2 and Python 3.12.14; other runtimes are not claimed tested.

Third-party license and notice files are retained. This report and the GERO tests were generated with an AI assistant under Xamit Kadirbekov's direction. No independent human code review or personal execution by the owner is claimed.

## Public records and delivery status

- [github](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/commons-statistics-loguniform-narrow-interval-probabilities.md)
- [gero](https://www.gero.uz/research/articles/commons-statistics-loguniform-narrow-interval-probabilities.html)

[Complete evidence ZIP](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/raw/refs/heads/main/reports/commons-statistics-loguniform-narrow-interval-probabilities/gero-commons-loguniform-evidence-2026-09-23-v1.0.1.zip). SHA-256: `523062c72ecb4090396858eab417b75b4687f8cc58a435b03bd3d3b2a49b48f7`.

Pending distribution: zenodo, huggingface, linkedin, youtube. One technical email was sent to dev@commons.apache.org before publication; delivery, moderation, acknowledgment and acceptance remain unverified.
