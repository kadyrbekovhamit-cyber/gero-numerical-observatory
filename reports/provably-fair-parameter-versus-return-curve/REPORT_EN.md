# Provably fair does not mean a constant expected return

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 22 October 2026

**Status:** Qualified educational analysis; no live-operator defect or monetary loss claimed

A reproducible mapping from a seed-derived number to an outcome answers one question. Expected return also depends on the input grid, rounding, target and settlement rule. A parameter called `houseEdge` need not equal the total edge at every target.

**Galabet reference implementation.** At `3fe46bf37d2ccabfa30b352abee27e60f3f619d9` and published npm package 0.1.0, the mapper/rules at target 1,000,000 have 4,210 winning inputs among 2^32 possibilities. Under a uniform-grid assumption, expected gross return is exactly 32,890,625 / 33,554,432 ≈ 98.0217009783%, despite a 0.01 multiplier parameter. Seven target-boundary checks were rerun against actual package/source mappers and demo settlement rules. The public demo explicitly uses virtual chips with no monetary value. The specification describes a multiplier factor; we do not claim it promised a constant 99% return. No real operator, RNG entropy, HMAC preimage, complete API or funds were tested. The target bound was checked statically.

**Archived vero library.** At `9808897452707f13de28786d87596fdfc295b144`, algebra and a Python binary64 translation show that a 5% multiplier reduction does not imply a constant 5% total edge. With inclusive settlement at target 2, the continuous reference gives gross return 1254/1327; at a zero parameter it gives 198/199. The archive includes eight finite-grid boundary checks rerun on 22 October. **Go was not executed**; the settlement rule and uniform 52-bit input are explicit assumptions. The author acknowledged the parameter interpretation in our earlier exchange; no maintenance or deployed fix is claimed.

Both repository heads are unchanged at the fresh review. Private reports were sent on 5 October; Galabet had no response in the reviewed channel and vero did respond. The minimum publication date was 20 October. These are educational formula/contract clarifications, not accusations of fraud, game-playing advice, newly invented probability theory or a claim that any live gambling service was audited.

Public sources: [Galabet fair](https://github.com/galabetorg/fair), [vero](https://github.com/tahirmurata/vero).

## Evidence and limits

The attached archive contains the selected source, reproduction and observed output. AI-assisted investigation and writing; the stated executions were actually performed locally. Numerical jobs ran sequentially with one numerical worker and no GPU. No production account, real funds or actual customer losses were tested. Bounded public-source and issue review does not establish global priority. Private correspondence and contact details are excluded.

Report: CC BY 4.0. Original reproduction: MIT. Bundled upstream source retains its own licence; see the archived notices.
