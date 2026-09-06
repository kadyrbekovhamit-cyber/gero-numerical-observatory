import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from onnx import helper as h, TensorProto as T
from gero_stability.backends import evaluator
from gero_stability.cases import make_case, generate_cases, assert_coverage
from gero_stability.invariants import output_checks, metamorphic_checks
from gero_stability.compare import compare


@pytest.mark.parametrize("backend", ["reference", "ort"])
@pytest.mark.parametrize("optimization", ["disabled", "all"])
@pytest.mark.parametrize("operation", ["Softmax", "LogSoftmax", "MaskedSoftmax", "LayerNormalization", "LpNormalization", "MSE", "CosineSimilarity", "DequantizeLinear"])
def test_real_backend_contracts(operation, backend, optimization):
    rng = np.random.default_rng(132)
    feeds = {"x": rng.normal(size=(4, 13)).astype(np.float32)}
    if operation in ("MSE", "CosineSimilarity"): feeds["target"] = rng.normal(size=(4, 13)).astype(np.float32)
    if operation == "MaskedSoftmax":
        feeds["mask"] = rng.random((4, 13)) > .5; feeds["mask"][0] = False
    if operation == "DequantizeLinear": feeds["x"] = rng.integers(-128, 128, (4, 13), dtype=np.int8)
    case = make_case(operation, feeds)
    y = evaluator(case.model, backend, optimization)(feeds)
    checks = output_checks(case, y) + metamorphic_checks(case, y, backend, optimization)
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]


@given(st.lists(st.floats(-80, 80, allow_nan=False, allow_infinity=False, width=32), min_size=2, max_size=64))
@settings(max_examples=30, deadline=None, derandomize=True)
def test_probability_and_shift_invariance(values):
    x = np.array([values], dtype=np.float32)
    case = make_case("Softmax", {"x": x})
    for backend in ("reference", "ort"):
        y = evaluator(case.model, backend)({"x": x})
        assert all(c["passed"] for c in output_checks(case, y))
        # Exactly representable moderate shift; tolerance covers rounded logits.
        shifted = evaluator(case.model, backend)({"x": x + np.float32(1)})
        assert compare(shifted, y)["passed"]


def test_mask_contract_rejects_uniform_fully_masked_rows():
    case = make_case("MaskedSoftmax", {"x": np.ones((2, 4), np.float32), "mask": np.zeros((2, 4), bool)})
    broken = np.full((2, 4), .25, np.float32)
    checks = output_checks(case, broken)
    assert not next(c for c in checks if c["name"] == "masked_entries_exact_zero")["passed"]


def test_perturbed_probability_is_detected():
    case = make_case("Softmax", {"x": np.zeros((2, 4), np.float32)})
    broken = np.full((2, 4), .25, np.float32); broken[0, 0] = -.01
    assert any(not c["passed"] for c in output_checks(case, broken))


def test_catalogue_is_seeded_valid_and_excludes_quantizelinear():
    a, b = generate_cases(7, 2), generate_cases(7, 2)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert_coverage(x.model)
        assert x.model.SerializeToString() == y.model.SerializeToString()
        for key in x.feeds: np.testing.assert_array_equal(x.feeds[key], y.feeds[key])


def test_exclusion_applies_to_nested_subgraphs_and_functions():
    forbidden = h.make_node("QuantizeLinear", ["x", "s"], ["y"])
    subgraph = h.make_graph([forbidden], "branch", [], [])
    model = h.make_model(h.make_graph([h.make_node("If", ["c"], ["y"], then_branch=subgraph, else_branch=subgraph)], "outer", [], []))
    with pytest.raises(ValueError, match="Excluded"): assert_coverage(model)
    fn = h.make_function("test", "hidden", ["x", "s"], ["y"], [forbidden], [h.make_opsetid("", 18)])
    model = h.make_model(h.make_graph([], "empty", [], []), functions=[fn])
    with pytest.raises(ValueError, match="Excluded"): assert_coverage(model)
