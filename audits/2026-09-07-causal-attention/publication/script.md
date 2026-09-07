# Spoken script

## Hook — S1 / S2 · NATIVE REPRODUCTION

Causal attention must not read future values.

This N N trainer test shows that it does.

## Example — S2 · ZERO LOGITS · UPPER-LEFT MASK

With zero logits and values two, ten, fifty,

the outputs should be two and six.

Both are about twenty point six seven instead.

## Invariant — S2 · VALUE PERTURBATION TEST

Changing the final value changes earlier outputs.

A causal mask must prevent that.

## Mechanisms — S1 / S2 · ATTENTIONLAYER

A square mask fails on rectangular scores.

A finite penalty and later chunks also leak future values.

## Repair — S2 · PROPOSED LOCAL PATCH

The patch sets forbidden scores to negative infinity

and respects the query offset.

## Validation — S2 · 28 NEW + 14 EXISTING TESTS

Twenty-one tests fail before the patch. All forty-two pass after.

Native C P U, float thirty-two. Key cases repeat in three fresh processes.

## Scope — S3 · NNTRAINER ISSUE #4333

Reported upstream. Maintainer review is pending.

Attention engineers, reproduce it at gero dot U Z.
