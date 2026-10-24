# pvlib: clock time and elapsed time diverge on DST-transition days

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO). 24 October 2026.

**Status:** [PR #2923](https://github.com/pvlib/pvlib-python/pull/2923) is OPEN; the proposed correction is not merged or released.

## Reproduced discrepancy

On a daylight-saving transition day, elapsed hours since local midnight differ from local clock hours. The hour_angle calculation subtracts the UTC offset of each timestamp, so feeding it elapsed hours introduces a 15-degree discrepancy for a one-hour transition. The inverse conversion used by the geometric sunrise/sunset function can shift a predicted event by one hour.

The proposed change calculates local clock hours from timezone-naive wall-clock timestamps, and constructs inverse instants in UTC using the timestamp-specific offset. This also avoids constructing an ambiguous or nonexistent local midnight. It does not change the function signatures or return types.

The regression checks compare hour angle with the direct UTC expression 15*(UTC_hours-12)+longitude+equation_of_time/4, modulo a full rotation. They cover New York spring/fall transitions, Berlin, Santiago and Havana's ambiguous midnight, plus geometric event conversion under equivalent fixed-offset labels.

## Verification

Base: pvlib-python main 0c6ab770d2537b06b3c82067c33f4a2dbb87e9fa. Six transition cases fail before and pass after. The complete affected test_solarposition.py module changes from 59 passes / six failures / 21 skips to 65 passes / 21 skips. Optional-dependency skips are preserved.

The fresh solarposition.py and test module ran over an existing pvlib Python installation. This was not a fresh full installation or a successful run of every project test. The archive contains the exact changes, upstream license and local before/after metadata. To reproduce, check out the stated base, apply the patch and run tests/test_solarposition.py with the project's declared dependencies.

## Impact and prior history

This concerns these hour-angle/geometric-event functions, not every solar-position algorithm in pvlib. No universal annual-yield error or measured revenue loss is claimed. The earlier investigation's small annual-yield estimate is scenario-dependent and is not used as a business-loss estimate here.

[Issue #2132](https://github.com/pvlib/pvlib-python/issues/2132) and [PR #2133](https://github.com/pvlib/pvlib-python/pull/2133) already discuss DST/midnight behavior. The current proposal addresses the remaining elapsed-hours discrepancy; it is not a claim to have discovered DST problems in general. Maintainer review and broader integration tests remain pending.

## Attribution

The report authors used Codex/ChatGPT for code investigation, proposed changes, tests and writing. Upstream pvlib remains credited to its developers. Report CC BY 4.0; original reproduction MIT; upstream patch under the pvlib license.
