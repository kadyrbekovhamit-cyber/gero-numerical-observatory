> Archival mirror. [Original report](https://github.com/kadyrbekovhamit-cyber/spark-x25-financial-arithmetic-study/blob/codex/spark-math-study/README.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# HER Hack-Astron #6 — Exact fee bases and half-up rounding with Spark-X2.5-1.7B

An AI-operated, reproducible study of 12 original arithmetic exercises on an existing Apple M4. Original prompts, all generated outputs, token IDs, per-token logs, exact arithmetic references and scoring code are included.

**Authorship:** prepared and operated by Codex AI agents for Khamit Kadirbekov (GitHub `kadyrbekovhamit-cyber`, Hugging Face `XamitK`). AI agents performed design, coding, execution, text inspection and report drafting; they are not additional human entrants. No human execution or independent human review is claimed. Original study materials are released under Apache-2.0; the upstream chat template retains XHToken attribution.

## Results

The predeclared delivered-answer score is **8/12 (66.7%)**, with **4/6 both-correct pairs**. The unchanged full-stream secondary score is **0/12**. These are descriptive single-generation results on this authored set, not a general financial-accuracy estimate.

| Case | Reference | Parsed primary final | Primary status | Secondary status | Finish | Generated tokens |
| --- | ---: | ---: | --- | --- | --- | ---: |
| SI01_base | 37.02 | 37.02 | correct | no_final | stop | 2028 |
| SI01_perturbed | 37.02 | 37.02 | correct | no_final | stop | 1285 |
| SI02_base | 254.00 | 254.00 | correct | no_final | stop | 1873 |
| SI02_perturbed | 254.00 | 254.00 | correct | malformed_final | stop | 1167 |
| FB01_base | 455.90 | 455.90 | correct | no_final | stop | 1850 |
| FB01_perturbed | 455.90 | 455.90 | correct | malformed_final | stop | 1910 |
| FB02_base | 788.97 | — | truncated | truncated | length | 2048 |
| FB02_perturbed | 788.97 | — | truncated | truncated | length | 2048 |
| RH01_base | 25.04 | 25.04 | correct | no_final | stop | 2022 |
| RH01_perturbed | 25.04 | 25.04 | correct | no_final | stop | 1694 |
| RH02_base | -4.12 | — | truncated | truncated | length | 2048 |
| RH02_perturbed | -4.12 | — | truncated | truncated | length | 2048 |

A dash is a missing contract-valid primary answer; it does not mean no numerical value appeared anywhere in the reasoning. Every planned case remains in the denominator.

| Outcome | Primary count | Secondary count |
| --- | ---: | ---: |
| correct | 8 | 0 |
| incorrect | 0 | 0 |
| no_final | 0 | 6 |
| malformed_final | 0 | 2 |
| truncated | 4 | 4 |
| missing_execution | 0 | 0 |
| execution_error | 0 | 0 |
| invalid_run_record | 0 | 0 |
| invalid_completion_metadata | 0 | 0 |
| missing_channel_boundary | 0 | 0 |
| ambiguous_channel_boundary | 0 | 0 |

| Category | Delivered correct / planned |
| --- | ---: |
| simple_interest_units | 4/4 |
| explicit_fee_base | 2/4 |
| decimal_half_up_rounding | 2/4 |

## Reasoning and failure inspection

The four token-capped outputs show two distinct failure patterns. In the two fee/discount variants, the visible calculations contain unresolved arithmetic errors. In the two negative-rounding variants, the correct result appears repeatedly inside unfinished reasoning. All four fail the same predeclared delivery contract and remain in the denominator.

- **FB02_base:** the model repeatedly states `875 * 0.08 = 69.00` and `875*8 = 6200`, whereas the exact values are 70 and 7000. Another decomposition reaches the correct discounted base 805, but inconsistent multiplication recurs and the stream ends without a delivered answer.
- **FB02_perturbed:** the model correctly obtains base 805, processing fee 12.88 and the algebraic expression `0.984 * B - 3.15`. It then evaluates `789.120 - 0.15` as 789.070 instead of 788.970. A second derivation reaches the correct 788.97, but the incompatible 789.07 is reintroduced. No final answer is delivered before the cap.
- **RH02_base and RH02_perturbed:** both obtain the exact sum -4.115 and correctly apply the stated halfway-away-from-zero rule to -4.12. Both repeatedly reconsider the rule and reach 2,048 tokens without closing the thinking channel. A correct-looking `FINAL` marker inside reasoning does not count as a completed primary answer.

There were no primary outcome flips within the six wording/unit pairs: four pairs were both correct and two pairs were both truncated. The fee pair still illustrates different visible error paths under equivalent expressions. These six pairs do not establish general robustness or a causal effect of any wording change.

The **secondary full-stream score of 0/12 is a parser/channel diagnostic**. All eight normally completed outputs place the delivered `FINAL` line immediately after `</think>` on the same raw-stream line. Six therefore have no raw line beginning with `FINAL:`; two contain an earlier marker within reasoning that is not the last line. The pre-evaluation channel amendment yields eight valid primary finals from the exact suffix, without rewriting any output. The other four records are truncated in both scoring views. Both scores and every raw stream are retained so this distinction is directly inspectable.

The integration lesson from these cases is to preserve the model's completion and answer-channel information and verify the exact numerical contract. A numerical value found somewhere inside a response is insufficient evidence of successful answer delivery. The study did not test a larger token budget, an external calculator, another decoding policy, or a production workflow.


All 12 visible output streams were inspected by an AI reviewer; the coordinator additionally cross-checked all four token-capped failures. The reviewers share the AI-assisted study workflow, and their inspection is not independent human adjudication. Per-case assessments and raw-output SHA-256 identities are in [reasoning-review.json](https://github.com/kadyrbekovhamit-cyber/spark-x25-financial-arithmetic-study/blob/codex/spark-math-study/reasoning-review.json).

| Case | Visible-text assessment |
| --- | --- |
| SI01_base | The trace converts 72/360 to 0.2 years, uses rate 0.075, obtains annual interest 185.1 and then interest 37.02. The delivered answer matches the exact reference. |
| SI01_perturbed | The trace converts 1728 hours into 1728/(24*360)=0.2 defined years and multiplies 2468 by 0.075 and 0.2. Its final interest is correct. |
| SI02_base | The trace converts 15/12 to 1.25 years, computes 3175*0.064=203.2 and 203.2*1.25=254.0. It cross-checks by combining the rate and time to 0.08. |
| SI02_perturbed | Five quarters are converted to fifteen months and then 1.25 defined years. The trace computes the annual interest and multiplies by 1.25, with a second decomposition as 203.2+50.8. |
| FB01_base | The trace preserves the original gross fee base, converts 1.875 percent to 0.01875, derives a fee of 8.85 and total deductions 16.10, then obtains 455.90. It also checks the fee using 0.01875=3/160. |
| FB01_perturbed | The completed trace preserves the original gross fee base, obtains 472-7.25=464.75, computes the percentage fee as 3/160 of 472 = 8.85, and subtracts it to reach 455.90. An incorrect 3/64 fraction hypothesis is immediately tested and rejected; it does not determine the delivered answer. |
| FB02_base | The trace repeatedly miscomputes the 8-percent discount as 69 instead of 70 and multiplies 875*8 as 6200 or 6900 instead of 7000. It also derives the correct discounted base 805 through 875*90 + 875*2, but cycles between inconsistent alternatives. It ends while repeating the wrong discount, before computing the requested processing fee and final remainder. |
| FB02_perturbed | The trace correctly obtains B=805, the processing fee 12.88 and combined fees 16.03. One branch therefore correctly computes 805-16.03=788.97. Another branch correctly reaches 792.12 before the fixed fee but incorrectly subtracts 3.15 to get 789.07. The trace revisits both conflicting values and ends without resolving the subtraction or delivering a final channel. |
| RH01_base | The trace correctly identifies all three positive exact ties, rounds each line independently to 8.35, 12.68 and 4.01, and sums those rounded lines to 25.04. The delivered final matches the specified per-line rounding contract. |
| RH01_perturbed | The trace converts mills to exact decimal units, rounds the three lines separately to 8.35, 12.68 and 4.01, then obtains 25.04. The unit transformation and rounding order both match the contract. |
| RH02_base | The trace correctly sums the signed adjustments to -4.115 and repeatedly applies the explicitly defined negative tie rule to obtain -4.12. It remains in the reasoning channel and reaches the token cap without &lt;/think&gt;. A visible FINAL line inside that unfinished reasoning is not a delivered final and is not eligible for a correct score under protocol v2. |
| RH02_perturbed | The trace correctly interprets the credit/debit signs, reaches -5.120 and then -4.115, and repeatedly selects -4.12 as the negative half-up tie result. It never emits the reasoning-channel closing boundary before the token cap, so no delivered final exists despite the correct result appearing in the reasoning. |

## Study question and scope

Can a small local model deliver exact, explicitly specified financial arithmetic under equivalent wording? This study uses 12 original fictional exercises, arranged as six base/perturbed pairs. The three categories are simple-interest time units, percentage-fee bases, and decimal half-up rounding. This is an author-created case study, not a standard benchmark, customer audit or measurement of production losses.

Each pair preserves the mathematical answer while changing units or wording: days versus hours; months versus quarters; percentage versus decimal multiplier; per-line currency rounding versus integer mills; and reordered credits/debits for negative total-first rounding. All time conventions, fee bases and rounding rules are explicitly stated in the prompts. There is no train/development/test split: all 12 fixed authored items form the study set, with one independent generation per item and no examples or retries.

The reference calculator has separately grouped exact Fraction and 80-digit Decimal derivations. It checks that both agree with the stored expected answer for all 12 cases. The corpus, original scorer and methodology were frozen before evaluated generation; the exact inputs do not contain expected answers, calculations or rationales. Newly authored wording does not establish freedom from training contamination.

## Model and execution

- Model: [XHToken/Spark-X2.5-1.7B](https://huggingface.co/XHToken/Spark-X2.5-1.7B/tree/448e61eb392c00f2c403185c5b56d5e0665bfaab), pinned revision `448e61eb392c00f2c403185c5b56d5e0665bfaab`.
- Runtime: [official Spark-MLX-LLM](https://github.com/XHToken/Spark-MLX-LLM/tree/de2b4379fa1e2f2e1f99d84c83f0e008f651d86c), commit `de2b4379fa1e2f2e1f99d84c83f0e008f651d86c`.
- Hardware: existing Apple M4, 10 physical CPU cores, 16 GiB unified memory; GPU inference, CPU thread environment variables set to one.
- OS / Python: macOS 15.5 arm64 / Python 3.12.14.
- Precision: original BF16 checkpoint, strict weight loading; no quantization or conversion.
- Decoding: greedy temperature 0, top-p 1, top-k 0, seed reset to 0 per case; one sample per prompt.
- Budget: 2,048 maximum generated tokens per case; 4,096 total prompt-plus-output evaluation cap. The checkpoint advertises a larger context, but it was not exercised here.
- Thinking enabled; exact checkpoint chat template preserved. Its default system message is `you are a helpful assistant.` The harness adds no separate system instruction.
- Fresh conversation/KV cache for every case; EOS or the fixed token budget determines stopping. No answer-based early stop, tool call, correction opportunity or inference-time network request.
- Checkpoint file hashes were reverified before load; the imported official runtime path and clean tracked checkout were checked. Tokenizer loading did not execute remote Python code.

Installed runtime packages:

| Package | Version |
| --- | --- |
| `mlx` | `0.32.2` |
| `mlx-metal` | `0.32.2` |
| `mlx-lm` | `0.31.3` |
| `transformers` | `5.17.0` |
| `tokenizers` | `0.23.2` |
| `huggingface-hub` | `1.31.0` |
| `safetensors` | `0.8.0` |
| `spark-mlx-llm` | `0.1.0` |

The complete package inventory, exact rendered prompts/input token IDs, model-file hashes and chat template are included. The original download used the anonymous official Hugging Face Python SDK, not a signed-in CLI session. See execution-notes.md for the recorded setup warning, preliminary smoke test and reproduction command. The non-integration runtime tests and scorer fixtures validate harness behavior; their pass counts are not model accuracy.

## Predeclared scoring and amendment

The template opens the thinking channel in the prompt. An unrelated pre-study smoke test (`17 + 26`, answer 43) showed the closing `</think>` immediately adjacent to the displayed answer. Before any evaluated generation, protocol v2 therefore defined the primary scoring input as the exact unedited suffix after exactly one literal `</think>`. The three original corpus/oracle/methodology files remained unchanged. The amended wrapper and final harness were then frozen in the run manifest at 09:03:09 UTC. No evaluated answer was used to choose that amendment.

A normally completed answer must end with exactly one line `FINAL: <decimal>`, with precisely two decimal digits and no units. Decimal numeric equality with the exact reference is required. Fences, duplicate markers, extra final-line text, missing/ambiguous channel boundaries and invalid completion metadata fail the declared contract. Every token-capped response is a failure even if a correct number appears in its reasoning. The primary denominator remains all 12 planned cases and all six pairs.

The original unchanged full-stream score is also retained as a secondary diagnostic. It may count marker lines inside reasoning, so it does not replace the primary delivered-answer measure. Both scores are recomputed from the same unchanged records; outputs are not repaired or retried. A pair is both-correct only when both variants pass; agreement on two wrong answers is not success.

## Limits

Twelve constructed exercises, six related pairs and a single greedy sample cannot estimate general financial competence, compare models or establish business losses. A pair's wording may change answer delivery without proving a particular internal reasoning mechanism. Original prompts do not rule out contamination. The official MLX implementation was not numerically compared against a separate reference runtime. The two scoring views are sensitivity diagnostics on identical outputs, not independent experiments. AI reasoning inspection is disclosed and can miss errors; no independent human adjudication is claimed.

UTC timestamp differences and the process timer diverged during two cases. Original timestamps, durations and runtime speed fields remain available, but this study makes no controlled speed, energy or financial-cost claim.


## Every exact user prompt

The runtime adds the preserved chat template; no reference or rationale below was sent to the model.

### SI01_base

```text
This is a fictional arithmetic exercise. A principal of 2468.00 units earns simple interest at 7.5 percent per year for 72 days. Define one year as exactly 360 days. Simple interest means principal multiplied by the annual rate as a fraction, multiplied by elapsed years, with no compounding or fees. Calculate the interest only, not principal plus interest. Do not round intermediate values. Round the final amount to two decimal places using decimal ROUND_HALF_UP: nearest cent, with an exact halfway value rounded away from zero. End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator.
```

### SI01_perturbed

```text
This is a fictional arithmetic exercise. Interest accrues without compounding on a principal of 2468.00 units at an annual rate of 7.5 percent. The elapsed time is 1728 hours. For this exercise, every day has exactly 24 hours and every year has exactly 360 days. Use simple interest: principal times the annual rate as a fraction times elapsed years. There are no fees. Return only the earned interest amount, excluding the principal. Keep intermediate values unrounded. Round the final amount to two decimal places using decimal ROUND_HALF_UP: nearest cent, with an exact halfway value rounded away from zero. End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator.
```

### SI02_base

```text
This is a fictional arithmetic exercise. Use a year of exactly 12 equal months. A principal of 3175.00 units is held for 15 months at 6.4 percent annual simple interest. Define simple interest as principal times the annual rate expressed as a fraction times the number of elapsed years; it does not compound. No charges apply. What is the earned interest, excluding the original principal? Use exact decimal inputs and do not round intermediate values. Round the final amount to two decimal places using decimal ROUND_HALF_UP: nearest cent, with an exact halfway value rounded away from zero. End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator.
```

### SI02_perturbed

```text
This is a fictional arithmetic exercise. A quarter is defined as exactly 3 equal months, and a year as exactly 12 such months. Over 5 quarters, a principal of 3175.00 units earns simple interest at an annual rate of 6.4 percent. The interest is principal multiplied by the annual rate as a fraction and by elapsed years. There is no compounding and no fee. Find the interest alone, without adding the principal. Do not round intermediate calculations. Round the final amount to two decimal places using decimal ROUND_HALF_UP: nearest cent, with an exact halfway value rounded away from zero. End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator.
```

### FB01_base

```text
This is a fictional arithmetic exercise. A gross receipt is 472.00 units. Deduct a fixed charge of 7.25 units and a second charge equal to 1.875 percent of the original gross receipt of 472.00 units. The percentage charge is not based on the amount remaining after the fixed charge. There are no other deductions. What amount remains? Do not round either charge or any intermediate value. Round the final remainder to two decimal places using decimal ROUND_HALF_UP: nearest cent, with an exact halfway value rounded away from zero. End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator.
```

### FB01_perturbed

```text
This is a fictional arithmetic exercise. Start with an original receipt G = 472.00 units. First subtract 7.25 units. Next subtract a fee defined as 0.01875 times G, where G still means the original 472.00 units and does not change after the first subtraction. No other charge applies. Compute the amount left after these two subtractions. Perform intermediate arithmetic without rounding. Round the final remainder to two decimal places using decimal ROUND_HALF_UP: nearest cent, with an exact halfway value rounded away from zero. End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator.
```

### FB02_base

```text
This is a fictional arithmetic exercise. An original charge is 875.00 units. A discount removes 8 percent of that original charge. Define B as the amount after this discount. Deduct a processing fee equal to 1.6 percent of B, and also deduct a fixed fee of 3.15 units. The fixed fee does not reduce the base used to calculate the processing fee. There are no other adjustments. Find the amount remaining after the discount and both fees. Do not round intermediate values. Round the final amount to two decimal places using decimal ROUND_HALF_UP: nearest cent, with an exact halfway value rounded away from zero. End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator.
```

### FB02_perturbed

```text
This is a fictional arithmetic exercise. First calculate a discounted base B by multiplying 875.00 units by 0.92. From B, subtract a fixed fee of 3.15 units and a processing fee of 0.016 times B. In the processing-fee formula, B refers to the amount before either fee is subtracted. No further adjustments apply. Calculate the final amount left. All numbers are exact decimal values; do not round intermediate calculations. Round the final amount to two decimal places using decimal ROUND_HALF_UP: nearest cent, with an exact halfway value rounded away from zero. End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator.
```

### RH01_base

```text
This is a fictional arithmetic exercise. Three separate line amounts are exactly 8.345, 12.675, and 4.005 units. First round EACH line independently to two decimal places using decimal ROUND_HALF_UP: choose the nearest cent, and at an exact halfway value round away from zero. Only after rounding each line, add the three rounded amounts. Do not add the original unrounded line amounts first. What is the resulting total? End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator.
```

### RH01_perturbed

```text
This is a fictional arithmetic exercise. Define one mill as exactly 0.001 unit, so 1000 mills equal one unit. Three separate line amounts are 8345 mills, 12675 mills, and 4005 mills. Convert each line to units. Then round EACH converted line independently to two decimal places using decimal ROUND_HALF_UP: nearest cent, with exact halfway values rounded away from zero. Finally add the three rounded lines. Do not sum the unrounded amounts before rounding. Give the resulting total in units, not mills. End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator.
```

### RH02_base

```text
This is a fictional arithmetic exercise. Add these exact signed adjustments: -7.125 units, +2.005 units, and +1.005 units. Keep all three decimal places while adding; do NOT round the individual adjustments. After obtaining their sum, round that sum once to two decimal places using decimal ROUND_HALF_UP. This means nearest cent, with an exact halfway value rounded away from zero, including for negative sums. What is the final signed result? End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator.
```

### RH02_perturbed

```text
This is a fictional arithmetic exercise. A running total starts at exactly zero units. Apply a credit of 2.005 units, a debit of 7.125 units, and another credit of 1.005 units. A credit adds its amount and a debit subtracts its amount. Keep the exact decimal values throughout these operations; do not round a credit, a debit, or an intermediate total. Round only the final signed total once to two decimal places using decimal ROUND_HALF_UP: nearest cent, with an exact halfway value rounded away from zero for either sign. What signed total results? End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator.
```


## Representative full raw outputs

Included here: the first normally completed case in fixed order, followed by every token-capped case. Each block is the complete unedited `output` field, including any reasoning markers. [full-traces.md](https://github.com/kadyrbekovhamit-cyber/spark-x25-financial-arithmetic-study/blob/codex/spark-math-study/full-traces.md) contains all 12 prompts and complete outputs, while the machine-readable archive preserves token IDs and events. Selection is disclosed and does not affect the full result table.

### SI01_base — stop, 2028 tokens

```text
We are given: principal = 2468.00 units. Simple interest rate = 7.5% per year. Time = 72 days. Define one year = exactly 360 days. So number of years = 72 days / 360 days per year = 72/360 = 0.2 years exactly.

Simple interest formula: Interest = Principal * rate * time (in years). Rate as decimal: 7.5% = 0.075. So interest = 2468.00 * 0.075 * (72/360).

Compute step by step without rounding intermediate values.

First compute 72/360 = simplify: divide numerator and denominator by 12? 72/360 = 0.2 exactly. So time in years = 0.2.

Thus interest = 2468.00 * 0.075 * 0.2.

Compute 2468.00 * 0.075 = 2468 * 0.075 = 2468 * 75/1000 = (2468 * 75) / 1000. 2468 * 75 = 2468 * (3*25?) Actually 2468 * 75 = 2468 * (100 - 25) = 246800 - 61700 = 185100? Let's compute precisely: 2468 * 75 = 2468 * (3*25) = (2468*3)*25 = 7404*25 = 7404*100/4 = 740400/4 = 185100. Yes. So 2468 * 0.075 = 185100 / 1000 = 185.1? Wait 185100/1000 = 185.1. But careful: 2468 * 0.075 = 2468 * 75/1000 = (2468*75)/1000 = 185100/1000 = 185.1. So that's correct.

Now multiply by 0.2: 185.1 * 0.2 = 37.02. So interest = 37.02 units.

But we need to ensure no rounding intermediate. We used exact fractions: 2468 * 0.075 = 2468 * 75/1000 = (2468*75)/1000 = 185100/1000 = 185.1 exactly. Then times 0.2 = 185.1 * 1/5 = 37.02 exactly. So interest = 37.02.

Now total amount = principal + interest = 2468.00 + 37.02 = 2505.02.

But the problem says: "Calculate the interest only, not principal plus interest." Actually it says: "Calculate the interest only, not principal plus interest." Wait reading: "Calculate the interest only, not principal plus interest." That seems contradictory: "interest only" might mean just the interest amount, not the total. But then it says: "Do not round intermediate values. Round the final amount to two decimal places using decimal ROUND_HALF_UP: nearest cent, with an exact halfway value rounded away from zero." So they want the final amount? Or the interest? Let's parse: "Calculate the interest only, not principal plus interest." That phrase could be ambiguous. It might be instructing to compute only the interest, not the total. But then they say "Round the final amount to two decimal places". The "final amount" could be the total amount after interest? Or the interest? Usually in such problems, they ask for the interest amount. But they say "Calculate the interest only, not principal plus interest." That suggests we should compute the interest only, not the total. However, they also say "Round the final amount to two decimal places". If we compute interest only, the final amount would be the interest amount. But then they'd be rounding the interest to two decimals. That seems plausible.

But let's read the entire problem statement: "This is a fictional arithmetic exercise. A principal of 2468.00 units earns simple interest at 7.5 percent per year for 72 days. Define one year as exactly 360 days. Simple interest means principal multiplied by the annual rate as a fraction, multiplied by elapsed years, with no compounding or fees. Calculate the interest only, not principal plus interest. Do not round intermediate values. Round the final amount to two decimal places using decimal ROUND_HALF_UP: nearest cent, with an exact halfway value rounded away from zero. End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator."

Interpretation: They want the interest amount, but they say "not principal plus interest". So we should compute only the interest, not the total. Then they say "Round the final amount to two decimal places". So the final amount is the interest amount. So we need to output the interest amount rounded to two decimals.

But wait: "Calculate the interest only, not principal plus interest." That is explicit: we should compute only the interest, not the total. So we should output the interest amount.

But then they say "Round the final amount to two decimal places". So we need to round the interest to two decimal places.

Thus interest = 37.02 exactly, which is already two decimals. So final amount = 37.02.

But we must double-check: Could it be that they want the total amount? The phrase "Calculate the interest only, not principal plus interest." might be a misdirection? Actually sometimes they ask "Calculate the interest only" meaning compute the interest, not the total. But then they say "not principal plus interest" to emphasize that we should not add principal and interest. So yes, we compute only interest.

Thus answer: 37.02.

But we need to ensure rounding: 37.02 is exact, no rounding needed.

But wait: The principal is 2468.00, rate 7.5% per year, time 72 days, one year = 360 days. So time in years = 72/360 = 0.2 exactly. So interest = 2468 * 0.075 * 0.2 = 2468 * 0.015 =? Actually 0.075*0.2 = 0.015. So interest = 2468 * 0.015 = 37.02. Yes.

Thus final answer: 37.02.

But we must output exactly as: FINAL: 37.02

But the instruction: "End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator." So we need to output that line.

Thus final answer: FINAL: 37.02

But before finalizing, let's consider if there is any nuance: "simple interest means principal multiplied by the annual rate as a fraction, multiplied by elapsed years, with no compounding or fees." That's standard.

Also "Do not round intermediate values." We didn't.

Now, check if any rounding is needed for the final amount? It's already exact.

Thus answer is 37.02.

But wait: The principal is given as 2468.00 units. That's two decimals. Interest is 37.02. So total would be 2505.02. But they said "Calculate the interest only, not principal plus interest." So we output interest only.

Thus final.

But I'm slightly concerned: Could the phrase "Calculate the interest only, not principal plus interest." be interpreted as "Calculate the interest, but not the principal plus interest"? That is, compute the interest, but not the total. So yes.

Thus answer: 37.02.

Now, we need to output exactly as required.

Thus final answer: FINAL: 37.02

But we should ensure that the decimal is exactly two digits: 37.02.

Now, I'll produce the final answer.</think>FINAL: 37.02
```

### FB02_base — length, 2048 tokens

```text
We are given: original charge = 875.00 units.

Discount removes 8 percent of that original charge. So discount amount = 8% of 875.00 = 0.08 * 875 = 69.00? Let's compute: 875 * 0.08 = 69.00 exactly. So B = original - discount = 875 - 69 = 806.00? Actually 875 - 69 = 806.00. But careful: 875.00 * 0.92 = 806.00. Yes.

Now deduct a processing fee equal to 1.6 percent of B. So processing fee = 1.6% of B = 0.016 * B.

Also deduct a fixed fee of 3.15 units. The fixed fee does not reduce the base used to calculate the processing fee. So we compute processing fee based on B, then subtract that, then subtract fixed fee.

So final amount = B - processing fee - fixed fee.

We need to compute exactly, no rounding intermediate. Then round final amount to two decimal places using decimal ROUND_HALF_UP: nearest cent, with exact halfway value rounded away from zero.

We need to output FINAL: <decimal> with two digits after decimal.

Let's compute precisely.

B = 875.00 * (1 - 0.08) = 875 * 0.92 = 806.00 exactly? 875 * 0.92 = 875 * 92/100 = (875*92)/100. 875*92 = 875*100 - 875*8 = 87500 - 6200 = 81300? Wait 875*8=6200, so 87500-6200=81300. Then /100 = 813.00? That's not right. Let's recalc: 875 * 0.92 = 875 * (92/100) = (875*92)/100. 875*92 = 875*90 + 875*2 = 78750 + 1750 = 80500? Actually 875*90 = 78750, plus 875*2=1750 gives 80500. Then /100 = 805.00? That's 805.00, not 806.00. I made a mistake: 875 * 0.92 = 875 * (92/100) = (875*92)/100. 875*92 = 875*100 - 875*8 = 87500 - 6200 = 81300? Wait 875*8 = 6200, yes. 87500 - 6200 = 81300. That's 81300, not 80500. Let's recalc: 875*90 = 78750, plus 875*2 = 1750 gives 80500? Actually 78750+1750 = 80500. But 875*92 = 875*(90+2) = 875*90 + 875*2 = 78750 + 1750 = 80500. So 80500. Then /100 = 805.00. So B = 805.00? But earlier I thought 875 - 69 = 806.00. Let's check: 8% of 875 = 69.00? 875 * 0.08 = 69.00 exactly? 875 * 0.08 = 69.00? 875 * 8/100 = 6900/100 = 69.00. Yes. So 875 - 69 = 806.00. So there is inconsistency: 875 * 0.92 = 806? Let's compute 875 * 0.92 manually: 875 * 0.9 = 787.5, plus 875 * 0.02 = 17.5, sum = 805.0. That's 805.0. So 875 * 0.92 = 805.0. But 875 - 69 = 806.0. So which is correct? Let's recalc discount: 8% of 875 = 69.00. So 875 - 69 = 806.00. But 875 * 0.92 = 805.00. So there is a discrepancy: 875 * 0.92 = 805.00, but 875 - 0.08*875 = 875 - 69 = 806.00. So 0.92 * 875 = 805.00, but 875 * 0.92 = 875 * (92/100) = (875*92)/100. 875*92 = 875*100 - 875*8 = 87500 - 6200 = 81300? Wait 875*8 = 6200, yes. 87500 - 6200 = 81300. That's 81300, not 80500. I'm mixing numbers. Let's do carefully:

875 * 92 = 875 * (100 - 8) = 875*100 - 875*8 = 87500 - 6200 = 81300. Yes, 81300. Then divide by 100 gives 813.00? That's 813.00, not 805.00. So 875 * 0.92 = 813.00? That seems off because 0.92 is 92/100, so 875 * 92 = 81300, /100 = 813.00. But 875 * 0.92 = 875 * 0.9 + 875 * 0.02 = 787.5 + 17.5 = 805.0. So there is a mistake: 875 * 0.92 = 875 * (92/100) = (875*92)/100. But 875*92 = 81300? Let's compute 875*92 manually: 875 * 90 = 78750, 875 * 2 = 1750, sum = 80500. Wait 78750+1750 = 80500. Yes, 80500. I earlier got 81300 incorrectly. Let's recalc: 875 * 90 = 875 * 9 *10 = 7875 *10? Actually 875*9 = 7875, times 10 = 78750. That's correct. 875*2 = 1750. Sum = 78750+1750 = 80500. So 875*92 = 80500. Then /100 = 805.00. So B = 805.00. But then 875 - 69 = 806.00. So which is correct? Let's compute discount: 8% of 875 = 0.08 * 875 = 69.0
```

### FB02_perturbed — length, 2048 tokens

```text
We need to compute step by step.

Given:

- Base B = 875.00 * 0.92

Compute B: 875 * 0.92 = 875 * (92/100) = (875*92)/100. 875*92 = 875*90 + 875*2 = 78750 + 1750 = 80500. So 80500/100 = 805.00. So B = 805.00.

Now, from B, subtract a fixed fee of 3.15 units and a processing fee of 0.016 times B.

Processing fee = 0.016 * B.

So total subtraction = 3.15 + 0.016 * B.

Thus final amount = B - (3.15 + 0.016 * B) = B - 3.15 - 0.016 B = (1 - 0.016) B - 3.15 = 0.984 B - 3.15.

But careful: The problem says: "From B, subtract a fixed fee of 3.15 units and a processing fee of 0.016 times B. In the processing-fee formula, B refers to the amount before either fee is subtracted." So indeed we subtract both. So final = B - 3.15 - 0.016 B = (1 - 0.016) B - 3.15 = 0.984 B - 3.15.

Now B = 805.00.

Compute 0.984 * 805.00 = ?

0.984 * 805 = 805 * 0.984.

We can compute: 805 * 0.984 = 805 * (984/1000) = (805 * 984) / 1000.

Compute 805 * 984.

805 * 1000 = 805,000; subtract 805 * 16 = 12,880? Actually 984 = 1000 - 16, so 805*1000 = 805,000; 805*16 = 12,880; so 805,000 - 12,880 = 792,120. So 805 * 984 = 792,120. Then divide by 1000 gives 792.120. So 0.984 * 805 = 792.120.

But careful: 0.984 * 805.00 = 792.120 exactly? Since 805.00 * 0.984 = 805 * 0.984 = 792.120. Yes.

Now subtract 3.15: 792.120 - 3.15 = 789.070? Let's compute: 792.120 - 3 = 789.120, minus 0.15 = 789.070. So final = 789.070.

But we need to round to two decimal places using ROUND_HALF_UP. 789.070 is exactly 789.07? Actually 789.070 is 789.07 when rounded to two decimals? 789.070 has three decimal places; the third is 0, so it's 789.07. But we need to check if any rounding needed: 789.070 is exactly 789.07? Actually 789.070 = 789.0700, so to two decimals it's 789.07. But we must ensure that the rounding is correct: 789.070 is exactly 789.07? Wait, 789.070 is 789.070, which is 789.07 when rounded to two decimals because the third decimal is 0. So it's fine.

But we should double-check the arithmetic: B = 875 * 0.92 = 805 exactly. Then processing fee = 0.016 * 805 = 12.88? Let's compute: 0.016 * 805 = 805 * 16 / 1000 = (805*16)/1000 = 12880/1000 = 12.88. Yes. Fixed fee = 3.15. So total subtract = 3.15 + 12.88 = 16.03. Then B - 16.03 = 805 - 16.03 = 788.97? Wait, 805 - 16.03 = 788.97? Let's compute: 805 - 16 = 789, then subtract 0.03 = 788.97? Actually 805 - 16.03 = 788.97? Let's do: 805.00 - 16.03 = 788.97. Yes, that's 788.97. But earlier we got 789.070. There's a discrepancy. Let's recalc carefully.

We had B = 805.00.

Processing fee = 0.016 * B = 0.016 * 805 = 12.88.

Fixed fee = 3.15.

Total subtract = 3.15 + 12.88 = 16.03.

Then final = 805.00 - 16.03 = 788.97.

But earlier we computed 0.984 * B - 3.15 = 0.984*805 - 3.15. 0.984*805 = 792.12? Wait, 0.984 * 805 = 792.12? Let's recalc: 0.984 * 805. 0.984 * 800 = 787.2? Actually 0.984 * 800 = 787.2? Let's compute: 0.984 * 800 = 0.984 * 8 * 100 = 7.872 * 100? No, better: 0.984 * 800 = 0.984 * 8 * 100 = 7.872 * 100 = 787.2. Then 0.984 * 5 = 4.92. Sum = 787.2 + 4.92 = 792.12. Yes, that's correct. So 0.984*805 = 792.12. Then subtract 3.15 gives 789.07? 792.12 - 3.15 = 789.07? Let's compute: 792.12 - 3 = 789.12, minus 0.15 = 789.07. So 789.07. But that's not matching 805 - 16.03 = 788.97. So there is a mistake: 0.984 * B - 3.15 = (1 - 0.016) B - 3.15 = 0.984 B - 3.15. But B - (3.15 + 0.016 B) = B - 3.15 - 0.016 B = (1 - 0.016) B - 3.15 = 0.984 B - 3.15. That is correct algebraically.
```

### RH02_base — length, 2048 tokens

```text
We are given three signed adjustments: -7.125, +2.005, and +1.005. We need to add them exactly, keeping all decimal places, without rounding the individual adjustments. Then we sum them to get a total. Then we round that total to two decimal places using decimal ROUND_HALF_UP. That means rounding to nearest cent, with exact halfway value rounded away from zero, including for negative sums.

We need to compute the sum exactly.

First, compute sum: -7.125 + 2.005 + 1.005.

Let's add step by step.

-7.125 + 2.005 = -5.120? Actually -7.125 + 2.005 = -(7.125 - 2.005) = -5.120. Yes, -5.120.

Then add 1.005: -5.120 + 1.005 = -4.115.

So sum = -4.115.

Now we need to round -4.115 to two decimal places using ROUND_HALF_UP.

We need to consider rounding to two decimal places. The number is -4.115. The third decimal digit is 5 (since -4.115, digits: 4 . 1 1 5). The rule: round to nearest cent, with exact halfway value rounded away from zero. For negative numbers, we need to be careful: "nearest cent, with an exact halfway value rounded away from zero, including for negative sums." That means if the fractional part is exactly halfway between two cents, we round away from zero. For negative numbers, "away from zero" means if the number is negative, we round to the more negative side? Let's interpret: For a negative number, say -4.115, the two cents are -4.11 and -4.12? Actually we need to consider rounding to two decimal places. The number is -4.115. The two decimal places are the hundredths place. The third decimal is 5. So we need to decide whether to round up or down.

Standard rounding rules: For positive numbers, round half up (i.e., if the next digit is 5 or more, round up). For negative numbers, round half up typically means round to the nearest integer, but with half away from zero? Actually "round half up" is defined as rounding to the nearest value, with ties (exactly halfway) rounded away from zero. For negative numbers, "away from zero" means that if the number is negative, we round to the more negative side? Let's think: Suppose we have -4.115. The two possible two-decimal results are -4.11 and -4.12. Which is closer? -4.115 is exactly halfway between -4.11 and -4.12? Let's check: -4.115 - (-4.11) = -0.005; -4.12 - (-4.115) = -0.005. So yes, it's exactly halfway. The rule says "exact halfway value rounded away from zero". So we need to round away from zero. Away from zero from -4.115: The zero is at 0. So away from zero means moving further away from 0. For negative numbers, away from zero means more negative. So -4.115 is halfway; we round away from zero, so we go to -4.12? Wait, -4.12 is more negative than -4.11? Actually -4.12 < -4.11, so it's more negative. So away from zero means moving to the more negative side. So -4.115 -> -4.12? But check: -4.115 is exactly halfway; if we round away from zero, we would round to -4.12? But is that consistent with typical rounding? Usually, for negative numbers, round half up means round to the nearest integer, with ties rounded away from zero. For example, -1.5 rounds to -2 (since away from zero). So yes, -1.5 -> -2. So for -4.115, we would round to -4.12? But careful: The number is -4.115. The two cents are -4.11 and -4.12. Which is closer? They are equally close. The rule says "exact halfway value rounded away from zero". So we round to the one that is farther from zero. Since -4.12 is farther from zero than -4.11 (because -4.12 is more negative), we round to -4.12. So final result would be -4.12.

But we must double-check: The sum is -4.115. The third decimal is 5. So we need to round to two decimal places. The number is -4.115. The hundredths digit is 1? Actually -4.115: digits: 4 . 1 1 5. The hundredths place is the second decimal: 1? Wait: -4.115: The decimal part: 1 (tenths), 1 (hundredths), 5 (thousandths). So the number is -4.115. To round to two decimal places, we look at the thousandths digit (5). Since it's exactly 5, we round up. But rounding up for negative numbers: we need to consider the sign. The rule "round half up" for negative numbers: if the number is negative and the fractional part is exactly halfway, we round away from zero, i.e., to the more negative number. So -4.115 -> -4.12.

But is there any nuance about "including for negative sums"? Yes, they explicitly say including for negative sums. So we must apply that.

Thus final result = -4.12.

But we need to output exactly as: FINAL: -4.12 (with two digits after decimal point). So answer: FINAL: -4.12

But wait: The problem says: "Add these exact signed adjustments: -7.125 units, +2.005 units, and +1.005 units. Keep all three decimal places while adding; do NOT round the individual adjustments. After obtaining their sum, round that sum once to two decimal places using decimal ROUND_HALF_UP. This means nearest cent, with an exact halfway value rounded away from zero, including for negative sums. What is the final signed result? End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator."

So we have sum = -4.115. Rounding to two decimal places: -4.12.

But we should double-check the addition: -7.125 + 2.005 = -5.120? Actually -7.125 + 2.005 = -5.120 exactly? Let's compute: 7.125 - 2.005 = 5.120, so negative is -5.120. Yes. Then -5.120 + 1.005 = -4.115. Yes.

Now rounding: -4.115 to two decimal places. The rule: "nearest cent, with an exact halfway value rounded away from zero". So we need to consider the number -4.115. The two cents are -4.11 and -4.12. The halfway point is exactly -4.115. Since it's exactly halfway, we round away from zero. Away from zero means moving to the more negative side. So -4.12 is more negative than -4.11. So we round to -4.12.

But is there any possibility that the rounding should be to -4.11? Let's test with typical rounding: For positive numbers, round half up: 4.115 -> 4.12. For negative numbers, round half up: -4.115 -> -4.12? Actually many rounding conventions: round half up means that if the digit to be dropped is 5 or more, you round up. For negative numbers, "up" might be ambiguous. But they explicitly define: "with an exact halfway value rounded away from zero". So we follow that.

Thus final answer: -4.12.

But we must ensure that we output exactly as "FINAL: -4.12". The instruction: "End your answer with exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator." So we should output:

FINAL: -4.12

But we need to be careful: The decimal point and digits: -4.12. That's two digits after decimal:
```

### RH02_perturbed — length, 2048 tokens

```text
We need to compute the running total step by step.

Start: total = 0.

Apply credit of 2.005 units: total += 2.005 => total = 2.005.

Apply debit of 7.125 units: total -= 7.125 => total = 2.005 - 7.125 = -5.120? Let's compute: 2.005 - 7.125 = -5.120 exactly? 2.005 - 7.125 = -5.120. Yes, because 2.005 - 7.125 = -(7.125 - 2.005) = -(5.120) = -5.120. So total = -5.120.

Apply credit of 1.005 units: total += 1.005 => total = -5.120 + 1.005 = -4.115? Let's compute: -5.120 + 1.005 = -4.115. Yes.

So final total = -4.115.

Now we need to round to two decimal places using decimal ROUND_HALF_UP: nearest cent, with exact halfway value rounded away from zero for either sign.

We have -4.115. The number is -4.115. We need to round to two decimal places. That means we look at the third decimal digit (thousandths) to decide rounding.

-4.115: The digits: -4.115. The two decimal places are 11? Actually -4.115: the digits after decimal: 1,1,5. So the number is -4.115. To round to two decimal places, we look at the third decimal digit, which is 5. Since it's exactly 5, we need to apply ROUND_HALF_UP: round away from zero. For negative numbers, "away from zero" means rounding up (more negative) or down? Let's clarify: ROUND_HALF_UP means if the digit is exactly 5, we round up (i.e., increase the number) for positive numbers, but for negative numbers, "up" means towards zero? Actually careful: In standard rounding, "round half up" means that if the fractional part is exactly 0.5, we round to the nearest integer, with ties rounding away from zero. For negative numbers, "away from zero" means that if we have -4.115, the third digit is 5, we round away from zero, which would make it more negative? Let's think: For positive numbers, rounding away from zero means if we have 4.115, we round to 4.12 (since 4.115 is closer to 4.12? Actually 4.115 is exactly halfway between 4.11 and 4.12? 4.115 is exactly 0.005 above 4.11, and 0.005 below 4.12? Actually 4.115 is exactly halfway between 4.11 and 4.12? 4.11 and 4.12 difference is 0.01. Halfway is 4.115. So yes, it's exactly halfway. ROUND_HALF_UP would round to 4.12 because it's away from zero? Wait, away from zero: For positive numbers, away from zero means rounding up (to larger number). So 4.115 rounds to 4.12. For negative numbers, away from zero means rounding down (more negative) because zero is the reference. For example, -4.115: halfway between -4.12 and -4.11? Actually -4.115 is exactly halfway between -4.12 and -4.11? Let's check: -4.12 and -4.11 difference is 0.01. Halfway is -4.115. So it's exactly halfway. ROUND_HALF_UP for negative numbers: "away from zero" means rounding to the more negative side? Actually "away from zero" means moving away from zero, so for negative numbers, away from zero means more negative (i.e., lower). So -4.115 would round to -4.12? Let's test: If we have -4.115, rounding to two decimals, we look at the third decimal digit 5. Since it's exactly 5, we round away from zero. For negative numbers, away from zero means we go to the more negative side, i.e., -4.12 (since -4.12 is more negative than -4.11). But is that consistent with typical rounding? Usually, rounding half up for negative numbers: e.g., -1.235 rounded to two decimals: The third digit is 5, so we round up (i.e., increase the number) but since it's negative, "up" might be ambiguous. In many contexts, rounding half up means that if the digit is exactly 5, we round to the nearest even? No, that's round half to even. But here it's ROUND_HALF_UP, which is the opposite: round half up, i.e., always round away from zero. So for negative numbers, rounding away from zero means we round to the more negative number. So -4.115 -> -4.12.

But we need to be careful: The problem says: "Round only the final signed total once to two decimal places using decimal ROUND_HALF_UP: nearest cent, with an exact halfway value rounded away from zero for either sign." So yes, for negative numbers, halfway value rounded away from zero means rounding to the more negative side.

Thus -4.115 rounded to two decimals: we look at the third decimal digit = 5. Since it's exactly 5, we round away from zero. So we round to -4.12? Let's check: The number is -4.115. The two decimal places are 11? Actually -4.115: the digits: -4 . 1 1 5. So the two decimal places are 11? Wait, we need to round to two decimal places, so we keep two digits after decimal. The number is -4.115. The third digit is 5. So we need to decide whether to round up or down. For negative numbers, "up" might be ambiguous. But they explicitly say "rounded away from zero". So we need to round to the number that is farther from zero. For -4.115, the numbers at two decimal places are -4.11 and -4.12. Which is farther from zero? -4.12 is farther from zero (absolute value 4.12) vs -4.11 (4.11). So -4.12 is farther from zero. So we round to -4.12.

But is that consistent with typical rounding? Usually, rounding half up for negative numbers: e.g., -1.235 rounded to two decimals: The third digit is 5, so we round up (i.e., increase the number) but since it's negative, increasing means making it less negative? Actually "increase" for negative numbers means moving towards zero? Let's think: If we have -1.235, the nearest two-decimal numbers are -1.23 and -1.24. Which is closer? -1.235 is exactly halfway. "Round half up" means we round to the nearest number, and if exactly halfway, we round away from zero. So away from zero means we round to the more negative number, i.e., -1.24. So yes, -1.24. So that matches.

Thus final total = -4.12.

But we must ensure that we didn't make any rounding errors earlier. The total after each step is exact decimal values. So final total = -4.115 exactly.

Now rounding: -4.115 to two decimal places: we need to output with two digits after decimal point. So -4.12.

But we must check if there is any nuance: The problem says "using decimal ROUND_HALF_UP: nearest cent, with an exact halfway value rounded away from zero for either sign." So yes.

Thus answer: FINAL: -4.12

But we need to output exactly one line in the form FINAL: <decimal>, using two digits after the decimal point and no unit symbol or thousands separator.

So FINAL: -4.12

But wait: Could there be any rounding after the final total? The problem says "Round only the final signed total once to two decimal places". So we only round the final total.

Thus answer is -4.12.

But let's double-check the arithmetic:

```


## Reproduction and evidence

Run `python3 -B verify_evidence.py` from the release root to check file hashes, exact prompts, raw token-event reconciliation and recompute both scores offline. See [execution-notes.md](https://github.com/kadyrbekovhamit-cyber/spark-x25-financial-arithmetic-study/blob/codex/spark-math-study/execution-notes.md) and [reproduce.sh](https://github.com/kadyrbekovhamit-cyber/spark-x25-financial-arithmetic-study/blob/codex/spark-math-study/reproduce.sh) for a fresh full run on matching hardware. The helper was syntax/staging checked; the original study was not rerun just to test the packaging helper.

The artifact includes the unchanged frozen harness/protocol, all original and smoke outputs, checkpoint-file hash manifest, installed package inventory and full reasoning-review records. Model weights, private account information, credentials and personal cache paths are excluded. SHA-256 checks establish archive integrity and internal consistency, not independent attestation of physical execution.

Challenge: [official event and acceptance criteria](https://github.com/XHToken/Spark-X2.5/issues/9). This report makes no claim of organizer acceptance, an award or payment.
