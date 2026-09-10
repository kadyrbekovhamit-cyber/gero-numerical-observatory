namespace {

array real_inverse_hyperbolic_slope(
    const array& x, bool acosh, Stream s) {
  auto one = array(1., x.dtype());
  auto scale = stop_gradient(
      where(isfinite(x, s), maximum(abs(x, s), one, s), one, s), s);
  if (acosh) {
    auto lower = divide(subtract(x, one, s), scale, s);
    auto upper = divide(add(x, one, s), scale, s);
    return divide(rsqrt(multiply(lower, upper, s), s), scale, s);
  }
  auto value = divide(x, scale, s);
  auto unit = divide(one, scale, s);
  auto norm = add(square(value, s), square(unit, s), s);
  return divide(rsqrt(norm, s), scale, s);
}

} // namespace
