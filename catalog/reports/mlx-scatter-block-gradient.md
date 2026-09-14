> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/mlx-scatter-block-gradient.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# A Missing Gradient in MLX Block Updates

Xamit Kadirbekov · Originally published 2026-09-09

A strict C++ scatter_max/min block update loses a winning gradient. Independent finite differences confirm the reference; a minimal extent correction passes 26 targeted checks.

[Original GERO article](https://www.gero.uz/research/articles/mlx-scatter-block-gradient.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/mlx-scatter-block-gradient) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22694499)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `8f39d6dd628d090bf17cf7b12c2603b4a3ab0ef26934039851d869278579cc99`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
A Missing Gradient in MLX Block Updates
https://www.gero.uz/research/articles/mlx-scatter-block-gradient.html

← Research index

NUMERICAL CASE STUDY

9 September 2026

A Missing Gradient in MLX Block Updates

The forward maximum is correct. A winning block element loses its gradient.

Xamit Kadirbekov

Reproducible numerical experiments · GERO Research

MLX C++ API

Block updates

CPU float32

STATUS · LOCALLY REPRODUCED

Strict, in-bounds blocks: baseline fails 12 of 26 targeted checks. Minimal extent correction: all 26 pass. Historical behavior at equal values is treated separately.

A four-element example exposes a gradient-routing error in

scatter_max

and

scatter_min

. The updates overlap, but all competing winners are strictly distinct. The result is locally linear, so no convention about a derivative at a tie is needed.

The

public evidence package

includes exact inputs, a C++ harness, two separate patches and before/after logs. The original C++ translation unit and both patch variants were rebuilt and rerun for publication.

The 52-second explanation

Your browser does not support embedded video.

Download the video

.

Watch the English Short on YouTube →

Original diagrams and synthetic narration by the fictional Alex Vector using the macOS Daniel voice. AI-assisted preparation.

Remotion project, narration and source ledger →

Two blocks, one missing contribution

source = [0, 0, 0, 0]
indices = [0, 1], axis = 0
updates = [[3, 4], [5, 6]]
cotangent = [2, -3, 4, 2]

scatter_max = [3, 5, 6, 0]
source gradient = [0, 0, 0, 2]  # correct

updates gradient:
expected = [[2, 0], [-3, 4]]
actual   = [[2, 0], [-3, 0]]

The first block updates positions 0 and 1, the second positions 1 and 2. Five beats four at the overlap. Every block lies inside the source. Locally, the weighted loss is

L = 2u₀₀ − 3u₁₀ + 4u₁₁ + 2s₃

. Its update gradient must therefore be

[2,0,−3,4]

.

Coordinate central differences of the actual C++ forward operation, with step

1/256

, confirm that vector. Negating the updates and using

scatter_min

gives the same missing derivative. A common shift of every input should change the loss by

5t

; the original backward instead gives a derivative of 1.

Baseline evidence →

The backward gather assumes the wrong extent

The

C++ scatter contract

supports slice updates. The max/min branch in

Scatter::vjp

nevertheless forces gather length one on indexed axes:

auto slice_sizes = cotangents[0].shape();
for (auto ax : axes_) {
  slice_sizes[ax] = 1;
}

The example needs an indexed extent of two. Gathering only block starts and broadcasting those values across the update tensor loses the final winning contribution. The minimal repair reads the trailing update dimensions corresponding to the source rank:

auto slice_sizes = Shape(
    updates.shape().end() - values.ndim(), updates.shape().end());

The minimal patch

adds no new array operations and leaves equality handling unchanged. Performance and full-suite compatibility were not measured.

What passed, and which patch was tested

Native variant

Scenarios passed

Checks

Failed checks

Original, strict blocks

0/2

26

12

Minimal extent patch, strict blocks

2/2

26

0

Original, extended selection

10/40

388

176

Extended extent-and-tie patch

40/40

388

0

These are targeted checks, not the complete MLX suite. One vector comparison counts as one check. Some extended checks require a proposed new policy at ties. The 40/40 result belongs to the extended patch; it must not be attributed to the minimal correction. The 176 failures are not 176 independent defects.

The harness checks forward values, gradients for each argument and both together, cotangent linearity, shared shifts, selected second derivatives and finite differences. Independent coordinate differences are used only for strict block cases.

Inspect the tests →

Equal values have a separate history

On MLX 0.32.2, a scalar copied into the source and four identical scatter updates produces a smooth identity

f(x)=x

, but the computed derivative is 5. A squared-loss composition similarly duplicates first and second derivatives. Ordinary binary max/min controls return the expected derivative.

This behavior is

not claimed as a new unknown finding

.

PR #431

, merged in January 2024, already tested propagation to both a tied source and update. Its author

explicitly asked about that choice

.

The separate

experimental patch

prioritizes winning updates over an equal source and divides the cotangent among tied winning updates. This is one valid policy, not the only possible convention. It changes historical behavior and adds winner-count arrays and scatter/gather work. It needs maintainer compatibility review and performance measurement.

Version pins and reproducibility

Inspected source:

24c699ecee2f7c8b2040de8da1c8382c8bcf31c7

, rechecked before publication.

Compatible native base:

ce916dbbcaa88e433b6fd1e60a17f766d49c27fe

. Both audited VJP methods are byte-identical to the inspected source.

Apple clang 17.0.0, C++20, CPU float32. Original and patched translation units link ahead of a reused CPU-only archive.

All 38 input artifact hashes matched. The C++ builds and runs were repeated. The official wheel equality probe is retained from the original audit and was not rerun for publication.

This is a partial native rebuild on a compatible base, not a clean build of current main.

The strict block example is demonstrated through the C++ API; that exact layout was not independently reproduced through Python

.at

.

export MLX_SOURCE_ROOT=/absolute/path/to/mlx-at-ce916db
export MLX_CPU_BUILD=/absolute/path/to/cpu-build
python3 build_and_test.py
./native-before --strict-blocks       # expected failure
./native-block-only --strict-blocks   # expected pass

Build setup and its validation boundary

·

Actual commands

·

Source and archive hashes

.

Limits, disclosure and evidence

GPU, other dtypes, non-finite inputs, every compiled/vectorized path, very large arrays and full-model effects were not validated. No device harm, security impact, maintainer acceptance or reward eligibility is asserted. The limited public search found no exact strict-block duplicate; novelty remains unestablished.

Prepared with AI assistance. Numerical results come from actual local execution, explicit algebra and finite differences. This is independent GERO Research work, not an Apple-endorsed audit. Included MLX source retains its MIT license.

Full English report, tests and patches →

Download the evidence ZIP

SHA-256: 8afaafb0f726b3818b0ea5af8e37685f68c7091fb78fb83551f16df975b53773

Source ledger

·

File checksums

#MLX #Autodiff #NumericalComputing #SoftwareTesting #OpenSource

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
