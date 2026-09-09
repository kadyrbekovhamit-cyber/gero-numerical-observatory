# Public-history review, 9 September 2026

The public `main` SHA returned by the GitHub API at the recorded check
was `a7ea056e79ab8e14447ea305c1b634e233343258`, matching the tested
checkout. See [provenance](provenance.json) and the saved
[API search results](duplicate-search.json).

Five queries were completed, all with `incomplete_results=false`:

| Query within nntrainer/nntrainer | Results |
|---|---:|
| pooling padding | 15 |
| pooling backward | 10 |
| pooling gradient | 6 |
| "average" "padding" | 6 |
| "height_stride_end" | 0 |

Result sets overlap. These counts are not five independent guarantees
that no report exists. Search matches metadata/body text, not every
code diff, comment or private report.

| Public item inspected | Relation to this case |
|---|---|
| [Issue #1045](https://github.com/nntrainer/nntrainer/issues/1045) | Explicitly asks for missing pooling-with-padding tests. It does not provide this current asymmetric-padding backward reproducer. This is a known historical testing gap. |
| [PR #1051](https://github.com/nntrainer/nntrainer/pull/1051) | Reworks pooling under an older symmetric-padding representation. Using one padding value per axis there does not itself establish the present four-sided mismatch. |
| [PR #1360](https://github.com/nntrainer/nntrainer/pull/1360) | Introduces the four-sided property and **already uses bottom/right limits in backward**. Its forward changes have a different limit expression. This is clear prior implementation of the proposed backward formula; we do not claim first discovery of that formula or prove a historically correct full version. |
| [PR #1577](https://github.com/nntrainer/nntrainer/pull/1577) | Contains padding-related validation/forward corrections. The inspected pooling patch does not change the average-backward limits. |
| [Open PR #4085](https://github.com/nntrainer/nntrainer/pull/4085) | Adds NHWC forward paths and rejects NHWC backward. Its inspected 176-line pooling diff leaves the NCHW average-backward limit expressions unchanged. This audit concerns NCHW. The PR was inspected, not built. |

The first four complete reviews are saved in
[reviewed-public-reports.json](reviewed-public-reports.json).
Fetching PR #4085 metadata timed out; a separate successful request saved
all 63 changed-file entries in [pr4085-files.json](pr4085-files.json),
including its pooling patch. An earlier search request also timed out;
it was resumed successfully. No test success is inferred from those reads.

Conclusion: the current source defect is reproduced and a small repair
is validated locally. No exact report of this current reproducer was
identified in the completed searches and inspected items. **Novelty,
absence of duplicates and reward eligibility are not established.**
The history contains a prior correct backward expression, so the report
must disclose that fact. The introducing commit has not been traced;
the local repository is shallow and its blame boundary is not an origin date.
