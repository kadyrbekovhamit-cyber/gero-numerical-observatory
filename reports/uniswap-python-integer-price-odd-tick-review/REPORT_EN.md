# uniswap-python: integer price flooring and odd-tick liquidity, with a public patch

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 22 October 2026

**Status:** Reproduced in main; public contributor PR 482 fixes the illustrated cases; not merged

Current main `51286c425fb88cc0a315c7b2db81e1ca863c218d` still truncates fractional spot prices before applying token decimals. A synthetic price of 0.00000049 becomes zero; reversing the quote raises division by zero. This affects the tested v3 and v4 Python methods, not the Uniswap smart contracts.

The v3 liquidity routine also uses integer division when turning a tick into a square-root price exponent. For liquidity 10^18 in the narrow interval [0,1), below the interval, it returns zero token0 instead of approximately 49,996,250,312,500 raw units. The even-tick controls retain their expected numerical behaviour.

We reran 14 scenarios on main and on public PR 482 at `3cd69062ea720fb838bcda5a290f7f9b139b09de`. The patch removes the illustrated large errors and zero-divisions; it is not an exact-arithmetic implementation. Our harness extracts unchanged method bodies, removes decorators, mocks RPC responses and compares with Fraction and 50-digit arithmetic. It does not import the full web3 classes, run the entire upstream suite, query a chain or test transactions. The equations follow the [Uniswap v3 whitepaper](https://app.uniswap.org/whitepaper-v3.pdf).

The earlier price issue 269 is prior work by YANGXIAO-LI. PR 482 is by TimeToBuildBob and expressly credits Xamit and Daniyal Kadirbekov (GERO) for additional cases and the odd-tick issue. We do not claim sole discovery or authorship of that PR. Our 5 October private report received a technical response; this is **not** an unanswered-report case. The public patch remains open as checked on 22 October.

## Evidence and limits

The attached archive contains the selected source, reproduction and observed output. AI-assisted investigation and writing; the stated executions were actually performed locally. Numerical jobs ran sequentially with one numerical worker and no GPU. No production account, real funds or actual customer losses were tested. Bounded public-source and issue review does not establish global priority. Private correspondence and contact details are excluded.

Report: CC BY 4.0. Original reproduction: MIT. Bundled upstream source retains its own licence; see the archived notices.
