from pathlib import Path
ROOT=Path(__file__).resolve().parent
def method(source, cls, name):
    begin = source.index("std::vector<array> " + cls + "::" + name + "(")
    return source[begin:source.index("\n}", begin)+2]

def patch(source):
    for cls, flag in (("ArcSinh", "false"), ("ArcCosh", "true")):
        old = method(source, cls, "jvp")
        guard = (
            "  if (!issubdtype(primals[0].dtype(), complexfloating)) {\n"
            "    return {multiply(\n"
            "        tangents[0],\n"
            f"        real_inverse_hyperbolic_slope(primals[0], {flag}, stream()),\n"
            "        stream())};\n"
            "  }\n")
        assert old.count("  array one") == 1
        source = source.replace(old, old.replace("  array one", guard + "  array one"), 1)
    needle = "std::vector<array> ArcCosh::vjp("
    helper = (ROOT / "stable_slope.cpp.inc").read_text()
    return source.replace(needle, helper + "\n" + needle, 1)


