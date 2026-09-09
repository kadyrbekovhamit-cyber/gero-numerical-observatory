# Episode brief

Technology Product / GERO. Audience: ML engineers and numerical-software
developers. English, portrait 1080×1920, target runtime 35–45 seconds.

Thesis: a correct average-pooling forward pass can still have a wrong transpose
Jacobian in backward. The example is reproduced in native nntrainer C++, checked
against the layer's own forward with central finite differences, and repaired by
using the bottom/right padding in the backward loop bounds.

Editorial boundary: the tested current-source defect is established. First
discovery, release impact, security impact, FP16/NHWC behavior and upstream
acceptance are not established. PR #1360 already contained the correct boundary
expressions and is disclosed on screen and in the description.

Visuals are original typographic diagrams. Alex Vector is a fictional narrator
using a system synthetic voice; no real-person voice clone, stock footage or music.
