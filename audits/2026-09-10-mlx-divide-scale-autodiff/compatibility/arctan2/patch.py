from pathlib import Path
ROOT=Path(__file__).resolve().parent
def method(source, name):
    begin = source.index("std::vector<array> ArcTan2::" + name + "(")
    return source[begin:source.index("\n}", begin)+2]
def patch(source):
    vjp = """std::vector<array> ArcTan2::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 2);
  assert(cotangents.size() == 1);
  const auto& s = stream();
  auto [scale, norm] = arctan2_scale_norm(primals[0], primals[1], s);
  std::vector<array> grads;
  for (auto arg : argnums) {
    grads.push_back(arctan2_weighted_partial(
        cotangents[0],
        arg == 0 ? primals[1] : negative(primals[0], s),
        scale, norm, s));
  }
  return grads;
}"""
    jvp = """std::vector<array> ArcTan2::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 2);
  assert(tangents.size() == argnums.size());
  assert(!argnums.empty());
  const auto& s = stream();
  auto [scale, norm] = arctan2_scale_norm(primals[0], primals[1], s);
  auto term = [&](int i) {
    return arctan2_weighted_partial(
        tangents[i],
        argnums[i] == 0 ? primals[1] : negative(primals[0], s),
        scale, norm, s);
  };
  auto out = term(0);
  for (int i = 1; i < argnums.size(); ++i) {
    out = add(out, term(i), s);
  }
  return {out};
}"""
    source = source.replace(method(source, "vjp"), vjp, 1)
    source = source.replace(method(source, "jvp"), jvp, 1)
    return source.replace("std::vector<array> ArcTan2::vjp(",
        (ROOT/"scaled_partials.cpp.inc").read_text()+"\nstd::vector<array> ArcTan2::vjp(", 1)

