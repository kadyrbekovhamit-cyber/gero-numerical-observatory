"""Compile prerequisites against an existing CPU MLX build."""
import json, os, shlex, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BUILD=Path(os.environ["MLX_CPU_BUILD"]).resolve()
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
entry=next(x for x in json.loads((BUILD/"compile_commands.json").read_text()) if x["file"].endswith("/mlx/primitives.cpp"))
args=shlex.split(entry["command"])
base=["-O0" if x=="-O3" else x for x in args[:args.index("-o")]]
for src,dest in (("baseline-primitives.cpp","before.o"),("native_regression.cpp","native_regression.o")):
    subprocess.run(base+["-o",str(ROOT/"higher-order"/dest),"-c",str(ROOT/"higher-order"/src)],cwd=BUILD,check=True)
