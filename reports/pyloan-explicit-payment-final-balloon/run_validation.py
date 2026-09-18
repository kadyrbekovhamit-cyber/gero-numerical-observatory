from pathlib import Path
import subprocess,sys
b=Path(__file__).parent
for name in ["run_grid.py","run_tests.py"]:
    subprocess.run([sys.executable,"-B",str(b/name)],check=True)
