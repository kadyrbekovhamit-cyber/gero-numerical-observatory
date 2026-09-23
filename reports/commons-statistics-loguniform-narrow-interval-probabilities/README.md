# Reproduce the Commons Statistics LogUniform report

Read REPORT_EN.md for exact claims and limits. One numerical mechanism; not 48 separate bugs.

## Offline replay

Extract the complete ZIP, install Python3.11+ and JDK11+ from their official providers if not already present, then set GERO_JAVA_HOME to your JDK home and run:

```sh
cd reproducer
python3 replay.py
```

The included seven original Maven JARs are sufficient; no dependency installation or network is used by the replay. On the measured system Python3.12.14 and OpenJDK26.0.2 were used. Java uses interpreted execution, one configured active processor and SerialGC. The scripts write generated results in the extracted reproducer directory. expected/ preserves the recorded results; compare numerical values, not machine-dependent timing/path strings.

Review RESULTS.csv, candidate/RESULTS.csv, restored/RESULTS.csv, VERIFICATION_RECEIPT.json, worksheet-*.csv/json and IMPACT_RECEIPT.json. The worksheet is a synthetic GERO consumer, not an Apache application. The candidate changes the target class only; no full suite or all-method claim.

Original Apache code/JARs retain their Apache2.0 licenses/notices inside the archives; explicit copies are included. GERO-authored test/report material is offered under MIT where copyright applies. Review metadata copies redact machine-local paths; numeric inputs and outputs are unchanged. Source URLs and cryptographic hashes permit verification.
