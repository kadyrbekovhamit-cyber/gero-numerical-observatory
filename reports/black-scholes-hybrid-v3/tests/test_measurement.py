import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

import mpmath as mp

from lab import reference
from lab.black_scholes import stable_prices
from lab.corpus import arguments
from lab.metrics import price_error, round_binary64, rounding_class


ROOT = Path(__file__).resolve().parents[1]


class MeasurementTests(unittest.TestCase):
    def test_exact_binary_input_atm_identity(self):
        with mp.workdps(180):
            expected = mp.erf(mp.mpf(0.2) / (2 * mp.sqrt(2)))
            actual = reference.evaluate(1., 1., 1., 0., 0., 0.2).call
            decimal = mp.erf(mp.mpf("0.2") / (2 * mp.sqrt(2)))
            self.assertLess(abs(actual - expected), mp.mpf("1e-175"))
            self.assertGreater(abs(actual - decimal), mp.mpf("1e-18"))

    def test_relative_error_keeps_small_difference(self):
        with mp.workdps(180):
            ref = mp.mpf(1) + mp.power(2, -200)
        with mp.workdps(5):
            error = price_error(1., ref)
        with mp.workdps(180):
            self.assertGreater(mp.mpf(error["relative"]), 0)
            self.assertLess(abs(mp.mpf(error["absolute"]) / mp.power(2, -200) - 1), mp.mpf("1e-33"))

    def test_half_subnormal_ties_round_even(self):
        with mp.workdps(180):
            tiny = mp.power(2, -1074)
            self.assertEqual(round_binary64(tiny/2), 0.)
            self.assertEqual(round_binary64(tiny*(mp.mpf("0.5")+mp.mpf("1e-100"))), math.ulp(0.))
            self.assertEqual(round_binary64(tiny*mp.mpf("1.5")), 2*math.ulp(0.))
            self.assertEqual(rounding_class(tiny*mp.mpf("0.75")), "subnormal")
            self.assertEqual(rounding_class(tiny*mp.mpf("0.25")), "rounds_to_zero")

    def test_positive_integral_agrees_in_central_and_far_tail(self):
        for args in ((1., 1., 1., 0., 0., 0.2),
                     (math.exp(-20.), 1., 1., 0., 0., 1.),
                     (math.exp(-100.), 1., 1., 0., 0., 1e-10)):
            oracle, convergence = reference.checked(*args)
            integral = reference.integral_log_otm(*args, dps=90)
            with mp.workdps(180):
                self.assertTrue(convergence["passed"])
                self.assertLess(abs(oracle.log_otm-integral), mp.mpf("1e-50"))

    def test_instrumentation_does_not_change_frozen_candidate(self):
        path = ROOT / "evidence/archive-v0/lab/black_scholes.py"
        spec = importlib.util.spec_from_file_location("frozen_bsm_v0", path)
        archived = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = archived
        spec.loader.exec_module(archived)
        corpus = json.loads((ROOT / "evidence/corpus-v1.json").read_text())
        for case in corpus["cases"]:
            args = arguments(case)
            before = archived.stable_prices(*args)
            diagnostic = {}
            after = stable_prices(*args, diagnostics=diagnostic)
            self.assertEqual((before.call, before.put, before.parity),
                             (after.call, after.put, after.parity), case["id"])
            self.assertIn("raw_otm", diagnostic)


if __name__ == "__main__":
    unittest.main()
