"""Small, named ONNX graphs. All reductions are along the feature axis."""
from dataclasses import dataclass
import hashlib
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

EXCLUDED = {"QuantizeLinear", "DynamicQuantizeLinear"}
OPSET = 19
IR_VERSION = 10


def assert_coverage(model):
    """Reject excluded operators recursively, including subgraphs and functions."""
    def visit(nodes):
        for node in nodes:
            if node.op_type in EXCLUDED:
                raise ValueError(f"Excluded from coverage: {node.op_type}")
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    visit(attr.g.node)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for graph in attr.graphs:
                        visit(graph.node)
    visit(model.graph.node)
    for fn in model.functions:
        visit(fn.node)


@dataclass
class Case:
    name: str
    category: str
    operation: str
    variant: str
    seed: int
    feeds: dict
    model: onnx.ModelProto
    equivalent: onnx.ModelProto | None
    contract: str
    parameters: dict

    def with_feeds(self, feeds):
        return make_case(self.operation, feeds, self.variant, self.seed, self.parameters)

    def metadata(self):
        return {"name": self.name, "category": self.category, "operation": self.operation,
                "variant": self.variant, "seed": self.seed, "contract": self.contract,
                "parameters": self.parameters, "opset": OPSET, "ir_version": IR_VERSION,
                "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in self.feeds.items()},
                "operators": sorted({n.op_type for n in self.model.graph.node})}


def make_case(operation, feeds, variant="ordinary", seed=0, parameters=None):
    parameters = dict(parameters or {})
    x = feeds["x"]
    dtype = np.dtype("float32") if operation == "DequantizeLinear" else x.dtype
    width = x.shape[-1]
    category = {"Softmax": "softmax", "LogSoftmax": "softmax", "MaskedSoftmax": "masking",
                "LayerNormalization": "normalization", "LpNormalization": "normalization",
                "MSE": "metrics", "CosineSimilarity": "metrics", "DequantizeLinear": "quantization"}[operation]
    contract = {"Softmax": "probability", "LogSoftmax": "log_probability", "MaskedSoftmax": "masked_probability",
                "LayerNormalization": "layer_norm", "LpNormalization": "unit_norm", "MSE": "mse",
                "CosineSimilarity": "cosine", "DequantizeLinear": "dequantize"}[operation]
    scalar = lambda value: np.asarray(value, dtype=dtype)
    nodes, inits = [], []
    def init(name, value):
        inits.append(nh.from_array(np.asarray(value), name)); return name
    def node(op, ins, out, **attrs):
        nodes.append(h.make_node(op, ins, [out], **attrs)); return out
    def reduce(op, inp, out):
        return node(op, [inp, "axis"], out, keepdims=1)
    init("axis", np.array([-1], dtype=np.int64))
    output_shape = list(x.shape)
    equivalent_nodes = None
    if operation in ("Softmax", "LogSoftmax", "MaskedSoftmax"):
        src = "x"
        if operation == "MaskedSoftmax":
            # Empty row policy: zero output, no invalid -inf - -inf intermediates.
            init("floor", scalar(np.finfo(dtype).min)); init("zero", scalar(0)); init("one", scalar(1))
            src = node("Where", ["mask", "x", "floor"], "masked")
        if operation != "MaskedSoftmax":
            node(operation, [src], "y", axis=-1)
            primary = list(nodes); nodes.clear()
        reduce("ReduceMax", src, "mx")
        node("Sub", [src, "mx"], "shift")
        node("Exp", ["shift"], "exp")
        if operation == "MaskedSoftmax":
            node("Where", ["mask", "exp", "zero"], "kept")
            reduce("ReduceSum", "kept", "total")
            node("Equal", ["total", "zero"], "empty")
            node("Where", ["empty", "one", "total"], "denominator")
            node("Div", ["kept", "denominator"], "y")
        else:
            reduce("ReduceSum", "exp", "total")
            if operation == "Softmax":
                node("Div", ["exp", "total"], "y")
            else:
                node("Log", ["total"], "log_total")
                node("Sub", ["shift", "log_total"], "y")
            equivalent_nodes, nodes = list(nodes), primary
    elif operation == "LayerNormalization":
        eps = parameters.setdefault("epsilon", 1e-5)
        init("scale", np.ones(width, dtype=dtype)); init("bias", np.zeros(width, dtype=dtype))
        init("epsilon", scalar(eps))
        node(operation, ["x", "scale", "bias"], "y", axis=-1, epsilon=eps, stash_type=T.FLOAT)
        primary = list(nodes); nodes.clear()
        reduce("ReduceMean", "x", "mean"); node("Sub", ["x", "mean"], "centered")
        node("Mul", ["centered", "centered"], "sq"); reduce("ReduceMean", "sq", "variance")
        node("Add", ["variance", "epsilon"], "regularized"); node("Sqrt", ["regularized"], "std")
        node("Div", ["centered", "std"], "y")
        equivalent_nodes, nodes = list(nodes), primary
    elif operation == "LpNormalization":
        node(operation, ["x"], "y", axis=-1, p=2)
        primary = list(nodes); nodes.clear()
        init("zero", scalar(0)); init("one", scalar(1))
        node("Mul", ["x", "x"], "sq"); reduce("ReduceSum", "sq", "sum")
        node("Sqrt", ["sum"], "norm"); node("Equal", ["norm", "zero"], "empty")
        node("Where", ["empty", "one", "norm"], "denom"); node("Div", ["x", "denom"], "y")
        equivalent_nodes, nodes = list(nodes), primary
    elif operation == "MSE":
        node("Sub", ["x", "target"], "difference")
        node("Mul", ["difference", "difference"], "sq"); reduce("ReduceMean", "sq", "y")
        init("two", scalar(2))
        equivalent_nodes = [nodes[0], h.make_node("Pow", ["difference", "two"], ["sq"]), nodes[2]]
        output_shape[-1] = 1
    elif operation == "CosineSimilarity":
        init("epsilon", scalar(parameters.setdefault("epsilon", 1e-6)))
        for name in ("x", "target"):
            node("Mul", [name, name], name + "_sq"); reduce("ReduceSum", name + "_sq", name + "_sum")
            node("Sqrt", [name + "_sum"], name + "_norm")
        node("Mul", ["x", "target"], "product"); reduce("ReduceSum", "product", "dot")
        node("Mul", ["x_norm", "target_norm"], "norm_product")
        node("Max", ["norm_product", "epsilon"], "denominator")
        node("Div", ["dot", "denominator"], "y")
        output_shape[-1] = 1
    elif operation == "DequantizeLinear":
        per_axis = parameters.setdefault("per_axis", False)
        scale = np.linspace(0.01, 0.2, width, dtype=dtype) if per_axis else scalar(0.03125)
        zero = np.full(width if per_axis else (), 3, dtype=x.dtype)
        init("scale", scale); init("zero_point", zero)
        node(operation, ["x", "scale", "zero_point"], "y", axis=-1)
        equivalent_nodes = [h.make_node("Cast", ["x"], ["xf"], to=T.FLOAT),
                            h.make_node("Cast", ["zero_point"], ["zf"], to=T.FLOAT),
                            h.make_node("Sub", ["xf", "zf"], ["difference"]),
                            h.make_node("Mul", ["difference", "scale"], ["y"])]
    def build(graph_nodes, suffix):
        # Strip unused constants to keep runtime diagnostics meaningful.
        used = {name for n in graph_nodes for name in n.input}
        inputs = [h.make_tensor_value_info(k, h.np_dtype_to_tensor_dtype(v.dtype), list(v.shape)) for k, v in feeds.items()]
        output = h.make_tensor_value_info("y", h.np_dtype_to_tensor_dtype(dtype), output_shape)
        model = h.make_model(h.make_graph(graph_nodes, operation + suffix, inputs, [output],
                                         [i for i in inits if i.name in used]),
                             opset_imports=[h.make_opsetid("", OPSET)], producer_name="gero-stability", ir_version=IR_VERSION)
        assert_coverage(model); onnx.checker.check_model(model, full_check=True)
        return model
    model = build(nodes, "")
    equivalent = build(equivalent_nodes, "_equivalent") if equivalent_nodes else None
    name = f"{operation}/{variant}/{x.dtype}/{'x'.join(map(str, x.shape))}/seed-{seed}"
    return Case(name, category, operation, variant, seed, feeds, model, equivalent, contract, parameters)


def generate_cases(seed=20260906, random_cases=12):
    cases = []
    operations = ["Softmax", "LogSoftmax", "MaskedSoftmax", "LayerNormalization", "LpNormalization", "MSE", "CosineSimilarity"]
    for operation in operations:
        for dtype in ([np.float32] if operation == "LayerNormalization" else [np.float16, np.float32, np.float64]):
            for variant in ("ordinary", "constant", "large_offset", "tiny", "wide"):
                # A local seed makes each case independent of catalogue iteration order.
                key = f"{seed}:{operation}:{np.dtype(dtype)}:{variant}"
                local_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
                rng = np.random.default_rng(local_seed)
                shape = (4, 257 if variant == "wide" else 17)
                x = rng.normal(size=shape)
                if variant == "constant": x.fill(3)
                if variant == "large_offset": x = x * .125 + (100 if dtype == np.float16 else 10000)
                if variant == "tiny": x *= 1e-4
                feeds = {"x": x.astype(dtype)}
                if operation in ("MSE", "CosineSimilarity"):
                    feeds["target"] = (x + rng.normal(0, .01, shape)).astype(dtype)
                if operation == "MaskedSoftmax":
                    mask = rng.random(shape) > .4; mask[0] = False; mask[1] = True
                    feeds["mask"] = mask
                cases.append(make_case(operation, feeds, variant, seed))
    for dtype in (np.int8, np.uint8):
        for per_axis in (False, True):
            lim = np.iinfo(dtype)
            x = np.linspace(lim.min, lim.max, 4 * 17).astype(dtype).reshape(4, 17)
            cases.append(make_case("DequantizeLinear", {"x": x}, "per_axis" if per_axis else "per_tensor", seed,
                                   {"per_axis": per_axis}))
    rng = np.random.default_rng(seed)
    for i in range(random_cases):
        operation = operations[i % len(operations)]
        shape = (4, int(rng.integers(2, 129)))
        x = (rng.normal(size=shape) * 10 ** rng.uniform(-3, 3)).astype(np.float32)
        feeds = {"x": x}
        if operation in ("MSE", "CosineSimilarity"): feeds["target"] = rng.normal(size=shape).astype(np.float32)
        if operation == "MaskedSoftmax": feeds["mask"] = rng.random(shape) > .5
        cases.append(make_case(operation, feeds, f"random-{i:03d}", seed))
    return cases
