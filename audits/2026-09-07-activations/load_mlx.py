import importlib.util
import os
from pathlib import Path
import subprocess
import types

SOURCE_ROOT = Path(os.getenv('AUDIT_SOURCE_ROOT', Path(__file__).parents[2] / 'audit-targets/current-stack-2026-09-07'))

def load_activations():
    root = SOURCE_ROOT / 'mlx'
    source = root / 'python/mlx/nn/layers/activations.py'
    code = (subprocess.check_output(['git', '-C', str(root), 'show', 'HEAD:python/mlx/nn/layers/activations.py'], text=True)
            if os.getenv('AUDIT_ORIGINAL') == '1' else source.read_text())
    module = types.ModuleType('audited_mlx_activations')
    module.__file__ = str(source)
    exec(compile(code, str(source), 'exec'), module.__dict__)
    return module
