> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/chain-of-thought-monitorability.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# Monitoring reasoning is a stability problem

Xamit Kadirbekov · Originally published 2026-09-02

OpenAI's monitorability evaluations raise a systems question: will a useful safety signal remain informative after training, scaling and distribution shift?

[Original GERO article](https://www.gero.uz/research/articles/chain-of-thought-monitorability.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/chain-of-thought-monitorability) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22729114)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `46e98e320358fd49ea3cf7e55ab5d0206e50460ce891802078dec4a09d014b76`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
Monitoring reasoning is a stability problem
https://www.gero.uz/research/articles/chain-of-thought-monitorability.html

← Research index

FRONTIER LAB WATCH

Reviewed 2 September 2026

Monitoring reasoning is a stability problem

OpenAI's monitorability evaluations raise a systems question: will a useful safety signal remain informative after training, scaling and distribution shift?

Xamit Kadirbekov

Independent analysis · Source: OpenAI

Chain-of-thought

Monitoring

Stability

STATUS · SOURCE REPORT + GERO ANALYSIS

This brief has not independently reproduced the laboratory's experiment.

PRIMARY SOURCE · 18 December 2025

Evaluating chain-of-thought monitorability ↗

What the laboratory reports

OpenAI introduces a suite of evaluations for how well monitors can detect properties of model reasoning.

The report describes current frontier reasoning as fairly but not perfectly monitorable.

It warns that monitorability could be fragile under future changes in training and scaling and frames it as one layer in defense in depth.

The mathematical problem

A monitor is a detector coupled to a changing generator. We need sensitivity, specificity and calibration across a family of model and environment perturbations, plus a stability margin before monitor outputs can control release decisions.

GERO's proposed response

Define monitorability as a versioned evidence contract, not a permanent model property.

Stress the monitor under paraphrase, tool changes, optimization pressure and shifted task distributions.

Fuse monitor evidence with mechanistic and deterministic checks while preserving disagreement.

A falsifiable experiment

Freeze a hidden hazard set and evaluate successive model checkpoints.

Estimate worst-group detection and calibration rather than only the mean score.

Trigger abstention when the measured stability margin falls below the deployment threshold.

Boundary

No chain-of-thought monitor can certify hazards that are absent from its evaluation distribution or unobservable in its input channel.

Read the primary source

This article is an original analytical summary, not a republication. Read

Evaluating chain-of-thought monitorability

for the laboratory's complete claims, methods and context.

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
