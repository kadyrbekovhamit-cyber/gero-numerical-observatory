# DropOutLayer routes multiple incoming gradients to input zero

Developer-facing case study. Tested revision:
`a7ea056e79ab8e14447ea305c1b634e233343258`, CPU FP32, macOS 15.5 arm64.

`finalize` accepts multiple input dimensions and requests one mask and
output per input; `forwarding` iterates over those inputs. In
`calcDerivative`, the same loop always requests outgoing gradient zero.

With two tensors of shape `[1,1,1,3]`, rate zero and separately allocated
input/output gradient buffers:

| | Incoming gradient | Actual input gradient | Expected |
|---|---|---|---|
| Input 0 | `[1,2,3]` | `[10,20,30]` | `[1,2,3]` |
| Input 1 | `[10,20,30]` | `[0,0,0]` | `[10,20,30]` |

Input gradients are initialized to zero. Forward is identity for both
inputs; input values and incoming gradients are not modified.

Suggested repair: use `getOutgoingDerivative(i)` inside the loop and
remove the unused `SINGLE_INOUT_IDX` constant.

Four regression cases cover a single-input control, the two-input identity
case, central differences of the actual forward, and deterministic
three-input mask replay. The control and 15 existing tests pass before
and after the repair. The other three new cases fail before and pass
after: original 16/19 pass; fixed 19/19 pass.

[Reproduction instructions](BUILD.md), [source/test patches](evidence/patches/),
[logs and XML results](evidence/), and [publication-day verification](prepublication-check/) are included.

The build reuses existing objects and unrelated earlier local repairs.
The baseline Dropout source exactly matched the pinned revision; only
Dropout files belong to the supplied patches. No fresh platform matrix,
FP16/GPU, in-place full graph or end-to-end training test is claimed.
The same index occurs in PR #1640 from 2021. No exact equivalent report
was identified in a bounded public search; originality is not established.

Prepared with AI assistance. No security impact or bounty claim is made.


Tags: #MachineLearning #SoftwareTesting #Autodiff #OpenSource #nntrainer
