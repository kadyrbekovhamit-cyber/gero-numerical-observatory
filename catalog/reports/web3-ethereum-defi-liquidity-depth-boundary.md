# web3-ethereum-defi: zero depth inside an active liquidity interval

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 22 October 2026

**Status:** Reproduced in isolated function with mocked pool state; no response found in reviewed channel

At `073d4ca855e1c0db34809df646dda481db8189e7`, the tested `estimate_liquidity_depth_at_block` function returns zero for both -0.5% and +0.5% moves in a synthetic pool with liquidity 10^18, current tick 0 and a single position [-60,60]. Both targets remain inside that position. Integrating the [Uniswap v3 equations](https://app.uniswap.org/whitepaper-v3.pdf) gives approximately 2.50313283699983×10^15 and 2.49066389236710×10^15 raw units, respectively.

The harness executes the unchanged function and its unchanged arithmetic helpers extracted with AST. Pool state is supplied by a mock, and the independent integral is evaluated with mpmath. The source also calls its real state provider without a required `api_key`; the mock bypasses that separate wiring problem. Thus this is not a successful live invocation, a deployed quote, or evidence of losses in a trading application.

The candidate repair direction is to integrate the active interval before crossing its next initialised tick, updating liquidity only at a crossed boundary. A full patch must additionally resolve the provider signature, direction/units, tick-boundary conventions and RPC integration. None is claimed complete here.

Our private 5 October report permitted publication no earlier than 20 October. Current source and a targeted mailbox review were checked on 22 October; no relevant response was found. Repository: [web3-ethereum-defi](https://github.com/tradingstrategy-ai/web3-ethereum-defi).

## Evidence and limits

The attached archive contains the selected source, reproduction and observed output. AI-assisted investigation and writing; the stated executions were actually performed locally. Numerical jobs ran sequentially with one numerical worker and no GPU. No production account, real funds or actual customer losses were tested. Bounded public-source and issue review does not establish global priority. Private correspondence and contact details are excluded.

Report: CC BY 4.0. Original reproduction: MIT. Bundled upstream source retains its own licence; see the archived notices.
