# Samsung ONE: two tensor-selection errors in the last public source

Xamit Kadirbekov · 22 October 2026 · AI-assisted investigation and writing.

Two small synthetic graphs expose incorrect rewrites in the public ONE model compiler. Native C++ tests fail on the pinned original pass implementations and pass after the supplied patches. This report concerns source revision `de7f4736dc4c4f5e47f72a4022a9aa9ac6d1ad1a`; it does not establish behavior in current Samsung devices or private code.

## 1. Gather must copy the whole selected slice

For integer input `[[0,1,2],[3,4,5]]`, indices `[1]`, axis `0`, the folded constant should be `[[3,4,5]]`. The actual FoldGather pass produces `[[3,0,0]]`. Its flattened copy writes one element per selected index even when each selected inner slice has several elements. A loop over the inner slice restores the Gather definition. This is a constant-folding error, not a rounding difference.

The existing last-axis tests have inner slice size one. The new axis-zero regression fails before the patch. With the patch, all six tests in the selected module pass.

## 2. Consecutive means must respect retained dimensions

With input shape `[1,2,3,4]`, `keep_dims=true`, and successive reductions along axes 1 and 2, the fused reduction must use axes `{1,2}`. The actual FuseMeanWithMean pass selects `{1,3}` because it remaps the second axis as though the first reduction had removed a dimension. Using second axis 3 instead throws an out-of-range exception.

A related failure affects negative second axes when dimensions are removed. On shape `[1,64,20,32]`, first axis 1 followed by axis -2 should map to original axes `{1,2}`; the pass produces `{1,3}`. Second axis -1 throws. The patch normalizes negative axes against the intermediate rank and only remaps axes when the first reduction removes dimensions.

Four regression cases fail before the change; all seven selected tests pass afterward. These are pass-level native tests. The complete model-conversion pipeline and device runtime were not tested.

## Reproduction and limits

The accompanying archive contains the two minimal patches, added regression tests, small graph witnesses, build helpers and captured before/after logs. The original pass source is fetched from the pinned public repository. Compilation is sequential. The focused test binaries were rerun independently in the main review session on 22 October. No production model, real-user data or measured customer loss is asserted.

Independent references are the definitions of slice selection and dimensional reduction, not expected numbers copied from the pass. The Gather witness selects the second row directly; the Mean witnesses explicitly track which original axes survive each reduction.

## Upstream status and prior-art search

The Samsung repository removed its source in July 2026. In public issue 16479 the maintainers explain the planned move/private transition and ask contributors not to merge more PRs because synchronization with the internal system has stopped. The public `one-project/ONE` fork retains the pinned source but has no issue tracker enabled. Consequently no upstream PR or acceptance is claimed for these patches. Project contacts were notified on 22 October and asked for the correct receiving team.

The bounded same-day search covered open and closed FoldGather/FuseMeanWithMean issues and PRs and the relevant file histories. The introducing PRs are 8720 and 7354. No exact correction was identified in the reviewed public scope. Private trackers and unpublished successors are not searchable; priority is not claimed beyond the published evidence.

## Sources

- [Pinned public source](https://github.com/one-project/ONE/tree/de7f4736dc4c4f5e47f72a4022a9aa9ac6d1ad1a)
- [FoldGather implementation](https://github.com/one-project/ONE/blob/de7f4736dc4c4f5e47f72a4022a9aa9ac6d1ad1a/compiler/luci/pass/src/FoldGatherPass.cpp)
- [FuseMeanWithMean implementation](https://github.com/one-project/ONE/blob/de7f4736dc4c4f5e47f72a4022a9aa9ac6d1ad1a/compiler/luci/pass/src/FuseMeanWithMeanPass.cpp)
- [Public repository transition](https://github.com/Samsung/ONE/issues/16479)
- [ONNX Gather definition](https://onnx.ai/onnx/operators/onnx__Gather.html)
- [ONNX ReduceMean definition](https://onnx.ai/onnx/operators/onnx__ReduceMean.html)

Report: CC BY 4.0. Upstream source and derived patches retain Apache-2.0 notices. AI assistance was used in research, code and prose; independent upstream review remains pending.
