# Render QA — 9 September 2026

- Final duration: 43.088 seconds; portrait 1080×1920 at 30 fps.
- Video/audio: H.264 + mono AAC 48 kHz; mean volume −19.8 dB, peak −1.4 dB on the checked render.
- All five full-resolution frames and the contact sheet were visually inspected.
- Unsupported Unicode arrows found in the first render were replaced with visible `TO`/`VS` labels and the final render was rechecked.
- Narration uses macOS Daniel at 188 words-per-minute setting. `nntrainer` is spoken as “N N trainer” to avoid ambiguous synthesis.
- Captions cover every spoken sentence and end before the final frame.
- No music, stock footage, third-party images, cloned voice or real-person avatar is present.
- Numerical text matches `results.json`: forward `[2.5, 3, 3.5, 4]`; observed derivative `[0.25, 0.25, 0.25, 0.25]`; reference `[0.25, 0.75, 0.75, 2.25]`; 8/58 failures before and 0/58 after.
- The history boundary is both spoken and shown: PR #1360 already contained the correct boundaries; first discovery is not claimed.
- Thumbnail remains legible at contact-sheet scale and makes only the supported claim: “Forward right. Backward wrong.”
