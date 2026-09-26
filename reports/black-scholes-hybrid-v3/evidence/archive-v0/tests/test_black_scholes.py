from __future__ import annotations

import math
import random
import unittest

from lab.black_scholes import naive_prices, stable_detailed, stable_prices


class BlackScholesTests(unittest.TestCase):
    def test_textbook_case_matches_direct_formula(self) -> None:
        direct = naive_prices(100.0, 100.0, 1.0, 0.05, 0.02, 0.2)
        stable = stable_prices(100.0, 100.0, 1.0, 0.05, 0.02, 0.2)
        self.assertAlmostEqual(stable.call, direct.call, places=13)
        self.assertAlmostEqual(stable.put, direct.put, places=13)

    def test_put_call_parity(self) -> None:
        for spot, strike, time, rate, dividend, vol in (
            (100.0, 80.0, 2.0, -0.01, 0.03, 0.4),
            (1e-40, 2e-40, 0.25, 0.12, -0.02, 1.2),
            (1e40, 0.9e40, 10.0, 0.02, 0.04, 0.05),
        ):
            pair = stable_prices(spot, strike, time, rate, dividend, vol)
            self.assertAlmostEqual(pair.call - pair.put, pair.parity, delta=max(math.ulp(pair.call), math.ulp(pair.put), math.ulp(pair.parity)))

    def test_atm_small_total_volatility_keeps_time_value(self) -> None:
        direct = naive_prices(1.0, 1.0, 1.0, 0.0, 0.0, 1e-10)
        stable = stable_prices(1.0, 1.0, 1.0, 0.0, 0.0, 1e-10)
        expected = 1e-10 / math.sqrt(2.0 * math.pi)
        self.assertGreater(stable.call, 0.0)
        self.assertLess(abs(stable.call - expected), abs(direct.call - expected))

    def test_zero_volatility_is_discounted_deterministic_payoff(self) -> None:
        pair = stable_prices(100.0, 110.0, 2.0, 0.03, 0.01, 0.0)
        parity = 100.0 * math.exp(-0.02) - 110.0 * math.exp(-0.06)
        self.assertAlmostEqual(pair.call, max(parity, 0.0), places=13)
        self.assertAlmostEqual(pair.put, max(-parity, 0.0), places=13)

    def test_invalid_inputs_fail_explicitly(self) -> None:
        with self.assertRaises(ValueError):
            stable_prices(0.0, 100.0, 1.0, 0.0, 0.0, 0.2)
        with self.assertRaises(ValueError):
            stable_prices(100.0, 100.0, -1.0, 0.0, 0.0, 0.2)

    def test_frozen_property_sample(self) -> None:
        rng = random.Random(20260925)
        for _ in range(100):
            spot = math.exp(rng.uniform(-5.0, 5.0))
            strike = math.exp(rng.uniform(-5.0, 5.0))
            time = 10.0 ** rng.uniform(-5.0, 1.0)
            rate = rng.uniform(-0.1, 0.2)
            dividend = rng.uniform(-0.1, 0.2)
            vol = 10.0 ** rng.uniform(-4.0, 0.5)
            pair = stable_prices(spot, strike, time, rate, dividend, vol)
            stock_leg = spot * math.exp(-dividend * time)
            strike_leg = strike * math.exp(-rate * time)
            self.assertGreaterEqual(pair.call, 0.0)
            self.assertGreaterEqual(pair.put, 0.0)
            self.assertLessEqual(pair.call, stock_leg * (1.0 + 1e-14))
            self.assertLessEqual(pair.put, strike_leg * (1.0 + 1e-14))
            higher_vol = stable_prices(
                spot, strike, time, rate, dividend, vol * 1.0001
            )
            self.assertGreaterEqual(higher_vol.call + 1e-14 * stock_leg, pair.call)
            self.assertGreaterEqual(higher_vol.put + 1e-14 * strike_leg, pair.put)

    def test_log_contract_preserves_unrepresentable_tail(self) -> None:
        detailed = stable_detailed(
            math.exp(-100.0), 1.0, 1.0, 0.0, 0.0, 1e-5
        )
        self.assertEqual(detailed.otm_option, "call")
        self.assertEqual(detailed.otm_value, 0.0)
        self.assertTrue(detailed.underflow)
        self.assertTrue(math.isfinite(detailed.log_otm_value))
        self.assertLess(detailed.log_otm_value, math.log(math.ulp(0.0)))
        self.assertEqual(detailed.time_value, detailed.otm_value)
        self.assertEqual(detailed.log_time_value, detailed.log_otm_value)
        self.assertEqual(detailed.call_intrinsic, 0.0)
        self.assertGreater(detailed.put_intrinsic, 0.0)


if __name__ == "__main__":
    unittest.main()
