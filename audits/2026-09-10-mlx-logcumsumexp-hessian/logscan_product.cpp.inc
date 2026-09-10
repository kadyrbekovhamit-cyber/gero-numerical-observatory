namespace {

// Apply the log-scan Jacobian or its transpose.
array log_scan_product(
    const array& x,
    const array& v,
    int axis,
    bool reverse,
    bool inclusive,
    bool transposed,
    Stream s) {
  if (x.size() == 0) {
    return zeros_like(x, s);
  }
  auto forward = [=](const std::vector<array>& p) {
    const auto& a = p[0];
    const auto& b = p[1];
    auto y = logcumsumexp(a, axis, reverse, inclusive, s);
    auto neg_inf = array(-std::numeric_limits<float>::infinity(), b.dtype());
    auto z = array(0, b.dtype());
    auto log_abs = log(abs(b, s), s);
    auto lp = where(greater(b, z, s), log_abs, neg_inf, s);
    auto ln = where(less(b, z, s), log_abs, neg_inf, s);
    auto part = [&](const array& log_b) {
      if (transposed) {
        auto q = where(isfinite(y, s), subtract(log_b, y, s), neg_inf, s);
        return exp(add(logcumsumexp(q, axis, !reverse, inclusive, s), a, s), s);
      }
      auto q = logcumsumexp(add(a, log_b, s), axis, reverse, inclusive, s);
      return where(isfinite(y, s), exp(subtract(q, y, s), s), z, s);
    };
    return std::vector<array>{subtract(part(lp), part(ln), s)};
  };
  auto backward = [=](
      const std::vector<array>& p,
      const std::vector<array>& cot,
      const std::vector<array>& out) {
    const auto& a = p[0];
    const auto& b = p[1];
    const auto& h = cot[0];
    if (transposed) {
      auto jh = log_scan_product(a, h, axis, reverse, inclusive, false, s);
      auto correction = log_scan_product(
          a, multiply(b, jh, s), axis, reverse, inclusive, true, s);
      return std::vector<array>{
          subtract(multiply(h, out[0], s), correction, s), jh};
    }
    auto jth = log_scan_product(a, h, axis, reverse, inclusive, true, s);
    auto correction = log_scan_product(
        a, multiply(h, out[0], s), axis, reverse, inclusive, true, s);
    return std::vector<array>{
        subtract(multiply(b, jth, s), correction, s), jth};
  };
  return custom_vjp(forward, backward)({x, v})[0];
}

} // namespace
