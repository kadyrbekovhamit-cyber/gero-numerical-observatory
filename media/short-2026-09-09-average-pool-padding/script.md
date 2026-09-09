# Narration

Can average pooling be right in forward and wrong in backward? In Samsung's N N trainer, one asymmetric SAME-padding case did exactly that.

For input one, two, three, four, the forward result was correct: two point five, three, three point five, four.

But with unit output gradients, nntrainer returned four quarters. The Jacobian and finite differences require zero point two five, zero point seven five, zero point seven five, two point two five.

The backward loop used top and left padding in its end bounds. Replacing them with bottom and right made all fifty-eight focused tests pass.

A 2021 pull request already contained the correct boundaries, so I am not claiming first discovery. The complete reproducer and limits are linked.
