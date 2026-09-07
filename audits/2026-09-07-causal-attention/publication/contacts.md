# Relevant contacts for the attention report

The report is already filed in the project's public issue tracker:
https://github.com/nntrainer/nntrainer/issues/4333

Two relevant people listed in nntrainer's `.github/CODEOWNERS` at audited commit
`a7ea056e79ab8e14447ea305c1b634e233343258`:

- MyungJoo Ham — https://github.com/myungjoo
- Jijoong Moon — https://github.com/jijoongmoon

Names and profile URLs were checked against GitHub's public profile API before
publication. CODEOWNERS establishes project involvement; this list does not claim
that either person is assigned to this issue or endorses the report.

Suggested introduction if a personal conversation is useful:

> Hello, I filed nntrainer #4333 with a native CPU reproduction and a tested local
> patch for causal-mask behavior in AttentionLayer. The report covers rectangular
> scores, finite masking penalties and later incremental blocks. I would value
> your feedback on the intended rectangular alignment and incremental contract,
> or an introduction to the engineer responsible for this layer. All inputs,
> source pins, test logs and patches are linked in the issue.

The LinkedIn and GERO publications invite attention-kernel engineers, ML runtime
maintainers and numerical-computing researchers to reproduce and compare results
in the upstream issue. No unsolicited private message or connection request was
sent as part of this publication.
