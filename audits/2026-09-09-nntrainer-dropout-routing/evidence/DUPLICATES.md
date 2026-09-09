# Public history and duplicate check

Checked on 9 September 2026. This is a bounded public search, not proof
of original discovery or absence of reports.

GitHub API query: `repo:nntrainer/nntrainer dropout in:title,body`.
The first request failed with a TCP connection timeout. A bounded retry
succeeded: **18 results**, `incomplete_results=false`, all returned on
one page. The response is saved in [duplicate-search.json](duplicate-search.json).
Web searches for Dropout with gradient, backward, derivative and
`SINGLE_INOUT_IDX` did not identify an exact current report.

Relevant history:

- [PR 1640](https://github.com/nntrainer/nntrainer/pull/1640), merged in
  October 2021, fixed training/inference copying and added the original
  golden tests. Its actual source patch already contains the constant-zero
  outgoing-gradient index inside the input loop, before and after that
  patch. Therefore the present line is longstanding, not evidence of a
  newly introduced 2026 regression. See `pr1640-files.json`.
- [PR 1359](https://github.com/nntrainer/nntrainer/pull/1359) introduced
  an older in-place Dropout implementation. It predates the present context
  API. Its diff was inspected, not compiled; see `pr1359-files.json`.
  We have not identified the commit that introduced the wrong destination.
- [PR 1660](https://github.com/nntrainer/nntrainer/pull/1660) describes
  mask allocation/use of preallocated memory. Its body describes a
  different issue; its complete patch was not audited here.
- [PR 2663](https://github.com/nntrainer/nntrainer/pull/2663) explicitly
  describes reusing masks when restoring an iteration for mixed precision.
  This supports interpreting `reStoreData(true)` as an existing runtime
  path. Our reproduction itself is CPU FP32, not a mixed-precision test.
- [Issue 1673](https://github.com/nntrainer/nntrainer/issues/1673) concerns
  tensor sharing by specification. No equivalence to the present
  gradient-destination failure has been established.
- [PR 3662](https://github.com/nntrainer/nntrainer/pull/3662) discusses
  RNG/mask preservation for checkpoint recomputation, another failure mode.

No exact fix or equivalent report was identified in the inspected titles,
relevant bodies and two historical file diffs. Full comment threads,
every historical revision, alternate terminology, private reports and
unindexed reports were not exhaustively checked. No novelty, payout or
security-impact conclusion follows from this search.

