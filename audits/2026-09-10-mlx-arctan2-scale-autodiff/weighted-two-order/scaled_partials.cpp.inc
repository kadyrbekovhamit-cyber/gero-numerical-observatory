namespace {

std::pair<array, array> arctan2_scale_norm(
    const array& y, const array& x, Stream s) {
  auto one = array(1., y.dtype());
  auto magnitude = maximum(abs(y, s), abs(x, s), s);
  auto valid = logical_and(
      isfinite(magnitude, s), greater(magnitude, array(0., y.dtype()), s), s);
  auto scale = stop_gradient(where(valid, magnitude, one, s), s);
  auto u = divide(y, scale, s);
  auto v = divide(x, scale, s);
  auto norm = add(square(u, s), square(v, s), s);
  return {scale, norm};
}

array arctan2_weighted_partial(
    const array& weight,
    const array& coordinate,
    const array& scale,
    const array& norm,
    Stream s) {
  auto one = array(1., weight.dtype());
  auto divide_first = isfinite(divide(stop_gradient(weight, s), scale, s), s);
  auto first_divisor = where(divide_first, scale, one, s);
  auto last_divisor = where(divide_first, one, scale, s);
  auto numerator = multiply(divide(weight, first_divisor, s), coordinate, s);
  auto term = divide(numerator, norm, s);
  return divide(divide(term, scale, s), last_divisor, s);
}

} // namespace
