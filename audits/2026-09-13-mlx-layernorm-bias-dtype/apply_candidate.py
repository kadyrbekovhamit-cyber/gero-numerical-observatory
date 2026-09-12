"""Apply one reviewed change to a disposable, exact source snapshot."""
import argparse
import hashlib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("mlx_source", type=Path)
a = p.parse_args()
path = a.mlx_source / "mlx/fast.cpp"
data = path.read_bytes()
expected = "29b31f96bdb02afdadbde3f2bcf42cf458fc69385a908bbde9fc3a14d23de077"
if hashlib.sha256(data).hexdigest() != expected:
    raise SystemExit("Source hash differs from the tested baseline; nothing changed")
source = data.decode()
start = source.index("array layer_norm(")
end = source.index("\nstd::vector<array> LayerNorm::vjp(", start)
section = source[start:end]
needle = "      : x.dtype();"
assert section.count(needle) == 1
updated = section.replace(needle, "      : ((has_bias) ? result_type(x, *bias) : x.dtype());")
path.write_text(source[:start] + updated + source[end:])
print("Applied the candidate bias-promotion branch to", path)
