import pytest

from tools.zero_failure_bounds import (
    exact_one_sided_upper_zero,
    minimum_zero_failure_sample,
    wilson_two_sided_upper_zero,
)


def test_two_clean_cases_do_not_establish_zero_risk():
    assert exact_one_sided_upper_zero(2) == pytest.approx(0.77639320225)
    assert wilson_two_sided_upper_zero(2) == pytest.approx(0.65761977249)


def test_required_clean_samples_for_named_upper_bounds():
    assert minimum_zero_failure_sample(0.05) == 59
    assert minimum_zero_failure_sample(0.01) == 299


def test_more_clean_cases_tighten_both_bounds():
    assert exact_one_sided_upper_zero(30) < exact_one_sided_upper_zero(10)
    assert wilson_two_sided_upper_zero(30) < wilson_two_sided_upper_zero(10)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        exact_one_sided_upper_zero(0)
    with pytest.raises(ValueError):
        minimum_zero_failure_sample(1.0)
