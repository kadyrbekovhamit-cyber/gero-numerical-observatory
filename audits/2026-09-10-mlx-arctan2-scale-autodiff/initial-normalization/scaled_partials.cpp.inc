namespace {

std::tuple<array, array, array> arctan2_scaled_partials(
    const array& y, const array& x, Stream s) {
  auto one = array(1., y.dtype());
  auto magnitude = maximum(abs(y, s), abs(x, s), s);
  auto valid = logical_and(
      isfinite(magnitude, s), greater(magnitude, array(0., y.dtype()), s), s);
  auto scale = stop_gradient(where(valid, magnitude, one, s), s);
  auto u = divide(y, scale, s);
  auto v = divide(x, scale, s);
  auto norm = add(square(u, s), square(v, s), s);
  return {divide(v, norm, s), negative(divide(u, norm, s), s), scale};
}

} // namespace
