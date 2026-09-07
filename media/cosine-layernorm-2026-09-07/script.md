# Spoken script

## Hook — S1 / S3

Can a vector fail to match itself?

Two reproducible defects in M L X and N N trainer.

## Cosine — S1 / S3

M L X cosine similarity returns not a number for this finite F P sixteen vector.

Compared with itself, the result should be one.

## LayerNorm — S2 / S4

N N trainer layer norm returns a zero input gradient.

Unequal gamma weights require a nonzero answer.

Finite differences confirm it.

## Repairs — S3 / S4

For cosine, scale before squaring and guard zero norm derivatives.

For layer norm, apply gamma before averaging gradients.

## Validation — S3 / S4 · RECORDED LOCAL RUNS

After local patches, one hundred one cosine tests pass on C P U and Metal.

All forty-two N N trainer C P U tests pass.

## Evidence — S3 / S4

Three fresh processes reproduce the key examples.

Recorded duplicate searches found no exact match.

## Scope — S1–S4 · LINKS IN DESCRIPTION

Maintainer confirmation and model level impact remain unproven.

Inspect the tests and patches at gero dot U Z.
