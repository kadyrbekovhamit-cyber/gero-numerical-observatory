> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/formal-proof-statement-contract.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# A machine-checked proof still needs a statement contract

[{"@type": "Person", "name": "Xamit Kadirbekov"}] · Originally published 2026-09-07

Anthropic's Fermat formalization shows the scale now possible in Lean; the remaining assurance problem is preserving meaning, provenance and dependency scope from source theorem to checked root.

[Original GERO article](https://www.gero.uz/research/articles/formal-proof-statement-contract.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/formal-proof-statement-contract) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22728950)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `05f09d70fece610443f2748565bb5e8425b6b81fcd0f9555dc8ef6557d0ce3fe`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
A machine-checked proof still needs a statement contract
https://www.gero.uz/research/articles/formal-proof-statement-contract.html

← Research index

FRONTIER LAB WATCH

Reviewed 7 September 2026

A machine-checked proof still needs a statement contract

Anthropic's Fermat formalization shows the scale now possible in Lean; the remaining assurance problem is preserving meaning, provenance and dependency scope from source theorem to checked root.

Xamit Kadirbekov

Independent analysis · Source: Anthropic

Formal verification

Semantic equivalence

Proof provenance

STATUS · SOURCE REPORT + GERO ANALYSIS

This brief has not independently reproduced the laboratory's experiment.

PRIMARY SOURCE · 4 September 2026

Formalizing Fermat's Last Theorem ↗

What the laboratory reports

Anthropic reports the first complete computer-checked proof of Fermat's Last Theorem, produced largely autonomously by Claude in Lean over 11 days.

The final proof uses 29,500 intermediate theorems; Anthropic reports 13 million generated lines and a collaboration of dozens of agents.

Early attempts lost track of project state. The successful workflow used Prove2Me to maintain a theorem dependency DAG, support search and reuse, and coordinate parallel work.

Anthropic reports that Lean checked the finished proof, a comparator matched its root statement to Mathlib's FLT statement, and Kevin Buzzard reviewed the artifact.

The mathematical problem

Kernel acceptance establishes that a formal term has the encoded type under a particular axiom, library and toolchain environment. End-to-end assurance additionally requires a semantic commuting diagram: the intended natural-language theorem, the chosen formal root, imported definitions and the claimed conclusion must denote the same proposition. At this scale, provenance and change impact across the dependency DAG become part of correctness.

GERO's proposed response

Package the formalization as a certificate-bearing claim graph with the root statement, axiom ledger, dependency DAG, library commits and toolchain hashes.

Keep kernel verification, statement equivalence, source-to-formal mapping and provenance as separate verdicts rather than collapsing them into one 'proved' label.

Prioritize independent review of high-centrality semantic bridges and require a clean replay from pinned artifacts before downstream reuse.

A falsifiable experiment

Create controlled variants with one perturbation at a time: a changed root quantifier, a mismatched definition, an altered import and a stale dependency hash.

Compare Lean compilation alone with the proposed statement-and-provenance contract on defect detection, false acceptance and expert review time.

Repeat the clean build in an isolated pinned environment and record which assurance layers survive without access to the original agent workspace.

Boundary

GERO has not downloaded, rebuilt or independently reviewed Anthropic's proof artifact. This brief accepts the laboratory's report as a primary source and proposes an assurance protocol around it; it does not allege a defect in the formalization.

Read the primary source

This article is an original analytical summary, not a republication. Read

Formalizing Fermat's Last Theorem

for the laboratory's complete claims, methods and context.

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
