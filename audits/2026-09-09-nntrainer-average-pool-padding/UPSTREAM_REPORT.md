# Average pooling backward skips windows for asymmetric padding, including SAME

At `a7ea056e79ab8e14447ea305c1b634e233343258`, CPU FP32
`Pooling2DLayer::calcDerivative` does not traverse the same output
windows as forward when bottom/top or right/left padding differ.
This silently produces the wrong input gradient for ordinary finite data.

Minimal example, NCHW input `[1,1,2,2]`:

```text
pooling=average
pool_size=2,2
stride=1,1
padding=same
x  = [[1,2],[3,4]]
dy = [[1,1],[1,1]]
forward       = [[2.5,3.0],[3.5,4.0]]  (correct)
actual dx     = [[0.25,0.25],[0.25,0.25]]
expected dx   = [[0.25,0.75],[0.75,2.25]]
sum(actual dx)=1; sum(dy)=4
```

The expected gradient uses the library's own forward convention:
average over the valid input elements of each window, excluding padding.
It is not a comparison against a different padding-divisor convention.
A separate test differentiates `sum(dy * forward(x))` using actual C++
forward calls, with mixed-sign dy and central differences. It agrees
with the reference derivative and disagrees with original backward.

The configured padding is `(top,bottom,left,right)=(0,1,0,1)`.
Forward's last row start is `height - pool_height + bottom`; backward
instead uses `height - pool_height + top`. Width has the same swap.
In the minimal example, backward therefore processes only one of four
outputs. Multiple channels/batches can also consume the saved per-window
counters out of alignment after the first skipped outputs.

The attached minimal source patch uses bottom/right padding for the
backward limits and casts to signed integers before subtraction.
The mathematical forward and division by each valid window count stay
unchanged. The regression patch includes six affected matrix cases,
four negative controls, the hand example and a real-forward finite
difference test. It also runs 46 existing pooling semantics/property tests.

```text
Original source: 58 tests, 8 failures, exit 1.
Patched source:  58 tests, 0 failures, exit 0.
```

The supplied source/tests patches apply to clean pinned copies of their
files. Recorded native runs reused a configured macOS arm64 CPU FP32
build with unrelated earlier repairs; a completely fresh build and the
upstream platform/FP16 matrix were not run. See `BUILD.md`, the XML/logs,
and `provenance.json` for reproduction and precise boundaries.

This is a numerical training-correctness report, not a demonstrated
security exploit or a claim of bounty eligibility. See `DUPLICATES.md`
for the public-history review and its limits. This draft has not been sent.
