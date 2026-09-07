import argparse
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--original', action='store_true')
parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
args = parser.parse_args()
os.environ['AUDIT_ORIGINAL'] = '1' if args.original else '0'
os.environ['DEVICE'] = args.device
import mlx.core as mx
import mlx.nn as nn
import pytest
from load_mlx import load_activations, SOURCE_ROOT

mx.set_default_device(getattr(mx, args.device))
module = load_activations()
for name in ['elu', 'selu', 'gelu', 'gelu_approx', 'ELU', 'SELU', 'GELU']:
    setattr(nn, name, getattr(module, name))
sys.path.insert(0, str(SOURCE_ROOT / 'mlx/python/tests'))
print('source:', 'HEAD' if args.original else 'patched', 'device:', args.device, flush=True)
raise SystemExit(pytest.main([
    str(Path(__file__).with_name('test_mlx_activations.py')),
    str(SOURCE_ROOT / 'mlx/python/tests/test_nn.py') + '::TestLayers::test_elu',
    str(SOURCE_ROOT / 'mlx/python/tests/test_nn.py') + '::TestLayers::test_gelu',
    '-q',
]))
