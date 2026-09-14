> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/self-correction-decomposition.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# Self-correction is two problems, not one

Xamit Kadirbekov · Originally published 2026-09-02

Google Research separates finding a mistake from repairing it; GERO turns that distinction into separate evidence contracts.

[Original GERO article](https://www.gero.uz/research/articles/self-correction-decomposition.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/self-correction-decomposition) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22729135)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `88c3b6df49c3387f955811351691cd6edfeaacad97821266a4f9551f74bd42f7`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
Self-correction is two problems, not one
https://www.gero.uz/research/articles/self-correction-decomposition.html

← Research index

FRONTIER LAB WATCH

Reviewed 2 September 2026

Self-correction is two problems, not one

Google Research separates finding a mistake from repairing it; GERO turns that distinction into separate evidence contracts.

Xamit Kadirbekov

Independent analysis · Source: Google Research

Self-correction

Error localization

Evaluation

STATUS · SOURCE REPORT + GERO ANALYSIS

This brief has not independently reproduced the laboratory's experiment.

PRIMARY SOURCE · 11 January 2024

Can large language models identify and correct their mistakes? ↗

What the laboratory reports

Google Research studies mistake finding separately from mistake correction.

The work treats feedback quality and repair ability as distinct capabilities rather than one self-correction score.

This decomposition exposes where a system fails: detecting an error, localizing it or producing a valid repair.

The mathematical problem

Detection, localization and repair have different loss functions. Collapsing them into a single accuracy number hides conditional failure rates, especially the probability that a wrong diagnosis leads to a persuasive but invalid repair.

GERO's proposed response

Create separate graph events for contradiction detection, faulty-node localization and dependency-safe repair.

Require a new certificate after repair; never inherit the original verdict automatically.

Track transition metrics such as invalid-to-valid, valid-to-invalid and unresolved-to-falsely-verified.

A falsifiable experiment

Inject labelled defects at different depths of a claim graph.

Score detection, localization and repair independently.

Test whether the repaired conclusion survives the same attacks as a clean reference solution.

Boundary

A model's ability to critique text does not demonstrate that its proposed correction is valid. The repaired artifact needs fresh verification.

Read the primary source

This article is an original analytical summary, not a republication. Read

Can large language models identify and correct their mistakes?

for the laboratory's complete claims, methods and context.

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
