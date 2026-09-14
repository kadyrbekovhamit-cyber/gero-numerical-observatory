> Archival mirror. [Original report](https://github.com/kadyrbekovhamit-cyber/gero-fineract-calculation-contracts/blob/main/README.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

---
title: GERO Financial Calculation Checks
emoji: 🧮
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Public calculation contracts and a scoped Fineract case study
---

# GERO financial calculation checks

**Turn written calculation rules into checks your engineering team can rerun.**

This public Apache Fineract example shows how GERO checks a selected version,
records mismatches and tests scoped correction candidates. It is independent
research using public source and synthetic inputs, not a customer audit.

[Read the customer overview](https://www.gero.uz/research/fineract-calculation-contracts/) ·
[Watch the 40-second demonstration](https://github.com/kadyrbekovhamit-cyber/gero-fineract-calculation-contracts/blob/main/media/gero-calculation-checks.mp4) ·
[Inspect the evidence](https://github.com/kadyrbekovhamit-cyber/gero-fineract-calculation-contracts/blob/main/evidence.json)

| Source state | Common cases passed / tested | Additional calendar cases |
|---|---:|---:|
| Historical baseline | 8 / 25 | Incomplete unpatched run; not claimed |
| Local candidate corrections | 25 / 25 | 144 / 144 |
| Selected upstream revision | 10 / 25 | Not run |

For example, seven grace days out of fourteen should be 0.5 under the declared
contract; the tested source returns 2. This establishes a component mismatch,
not a bank loss or production transaction impact. The inverse-helper correction
is already [merged upstream](https://github.com/apache/fineract/pull/6409).

A customer engagement starts with agreed conventions, version and scope. Its
outputs are a report, executable checks and a correction candidate where the
evidence supports one. Confidential customer work is handled separately from
this public background research.

Version: **0.1.0** · Published by **Xamit Kadirbekov / GERO Research** · 13 September 2026.
See [LICENSE](https://github.com/kadyrbekovhamit-cyber/gero-fineract-calculation-contracts/blob/main/LICENSE), [NOTICE](https://github.com/kadyrbekovhamit-cyber/gero-fineract-calculation-contracts/blob/main/NOTICE), and [provenance](https://github.com/kadyrbekovhamit-cyber/gero-fineract-calculation-contracts/blob/main/PROVENANCE.md).
The short video uses original graphics and disclosed synthetic English narration.

---

# Reproduce the scoped calculation checks

This publication copy contains test sources, candidate patches and a selected-field evidence summary. It excludes original native logs, local configuration and original XML. Hashes identify retained original reports; they are not third-party signatures.

Use an isolated Apache Fineract checkout and the required JDK. The recorded runs used Temurin 25.0.4.1+1 and Gradle 9.7.1. Supply dependencies through your normal approved build setup.

1. Check out one of the full commits stated in `evidence.json`.
2. Copy the three common test files into the matching paths below.
3. Run the targeted Gradle command below. Test failures on unmodified source are the reported outcomes; inspect the individual native XML results.

## Common test destinations

- `rules/GeroInverseContractTest.java` → `fineract-loan/src/test/java/org/apache/fineract/portfolio/loanaccount/loanschedule/domain/GeroInverseContractTest.java`
- `rules/GeroGraceFractionContractTest.java` → `fineract-loan/src/test/java/org/apache/fineract/portfolio/loanaccount/loanschedule/domain/GeroGraceFractionContractTest.java`
- `rules/GeroCalendarContractTest.java` → `fineract-core/src/test/java/org/apache/fineract/portfolio/savings/domain/interest/GeroCalendarContractTest.java`

```sh
./gradlew --no-daemon --no-parallel --max-workers=1 --continue \
  :fineract-loan:test --tests "*GeroInverseContractTest" --tests "*GeroGraceFractionContractTest" \
  :fineract-core:test --tests "*GeroCalendarContractTest"
```

## Local correction control

Use the declared historical baseline in a separate checkout. Apply only the production changes from the three supplied patches, then copy the common tests. The full patches also contain earlier test names; avoid compiling both old and renamed copies as if they were independent cases.

```sh
git apply --include="*/src/main/*" "patches/FINERACT-2809.patch"
git apply --include="*/src/main/*" "patches/grace-fraction.patch"
git apply --include="*/src/main/*" "patches/fiscal-compounding-calendar.patch"
```

Patch paths above assume the supplied patches have been copied into `patches/` inside that isolated checkout. Review the patches before applying them.

For the 144 additional corrected-calendar cases, copy `rules/GeroCalendarExpandedContractTest.java` to `fineract-core/src/test/java/org/apache/fineract/portfolio/savings/domain/interest/GeroCalendarExpandedContractTest.java` and add `--tests "*GeroCalendarExpandedContractTest"` to the core test task. The complete extension was verified only on the corrected source.

## Scope and licensing

The tests encode explicit component contracts. Grace tests use the stated fixed month/year bases. Calendar tests use the stated fiscal compounding conventions. Different product conventions require a separate review.

No HTTP integration, database posting, production deployment or customer impact is established. An earlier expanded unpatched run exhausted test-worker heap and produced incomplete results; do not interpret the corrected extension as unpatched coverage.

Test files retain Apache-2.0 headers. Changes for this harness include renamed test classes, unique parameterized display names and separation of the expanded calendar grid. Included license and notice files preserve upstream attribution. No Apache affiliation or endorsement is claimed.
