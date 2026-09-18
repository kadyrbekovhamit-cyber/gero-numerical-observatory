# Strata normal implied volatility: zero and tiny initial guesses

This archive accompanies GERO's issue [OpenGamma/Strata#2796](https://github.com/OpenGamma/Strata/issues/2796), submitted before public distribution. It describes one residual initialization defect associated with the earlier correction in issue 2238 / PR 2372, not a new report of their negative-forward problem. No upstream acceptance is claimed.

## Reproduce

Requirements: Python 3.9+ standard library, an existing Java Development Kit with `java` and `javac`, and access to the official Strata 2.12.74 release ZIP. Tested with Temurin 25.0.4.1 on macOS arm64. Other JDK/platform combinations are not claimed tested.

From this folder:

```sh
python3 -B reproduce.py --java-home /path/to/existing/jdk --output ./replay-output
```

The script downloads only the exact official release ZIP, verifies its SHA-256, extracts the named dependency jar and verifies its SHA-256. It does not install a JDK. To avoid a download, add `--release-zip /path/to/strata-report-tool-2.12.74.zip`.

The harness executes four configurations sequentially:

1. The unmodified official released jar.
2. The separately compiled pinned current `NormalFormulaRepository` class before the same released dependency jar on the classpath.
3. The same current class with the single local `Math.min` → `Math.max` initializer candidate.
4. The original class restored and freshly compiled.

Compiler/runtime use `ActiveProcessorCount=1`, `UseSerialGC`, and interpreter mode. No GPU, model training, media playback or parallel worker is used. These configure the Java worker; they do not cap the operating system or browser.

Expected: 1,750 rows per CSV; failures 864/864/0/864. The release/current/restored CSV files match byte for byte. Each CSV records arguments, independent expected value, actual value, tolerance and pass/fail. The fresh portable run also matched all four archived CSV files byte for byte. Absolute platform-independent bit identity is not guaranteed; `REPLAY_RECEIPT.json` records actual file matching and validation results.

## Contents and limits

- `StrataGrid.java`: 1,728 independent analytical ATM cases, 18 zero-intrinsic controls and 4 negative-forward controls transcribed from the existing upstream test.
- `Repro.java`: minimal positive-price / zero-start example and repricing residual.
- `source/`: exact current upstream class, related upstream test, original license and notices. The candidate is supplied as a separate patch.
- `observations/`: four recorded 1,750-row grids.
- `evidence/`: original verification, portable replay, source/duplicate review and vendor submission receipts.
- `REPORT_EN.md` and `CLAIMS.json`: public explanation and claim boundaries.

ATM oracle: `price = numeraire * sigma * sqrt(T) / sqrt(2*pi)`; hence positive price 1, forward = strike 100, T = 1, numeraire = 1 implies `sigma=sqrt(2*pi)=2.5066282746310002`. This is normal volatility in the forward's units per square-root year, not 2.5066 percent implied Black volatility.

This is a focused component audit, not a full build of Strata main, full JUnit suite, non-ATM coverage survey, performance study or bank deployment assessment. Four upstream regression scenarios are direct calls, not a claim of running their entire original test class. Of 886 formerly passing observations, all still pass; 450 actual output strings are identical and 436 change within declared tolerances. The candidate is not established correct for all inputs. Research and preparation were AI-assisted; real Java executions produced the results.

No customer records, measured losses, exploited systems or regulator allegations are involved. Scientific artifacts remain useful independently of whether the vendor accepts the proposal.
