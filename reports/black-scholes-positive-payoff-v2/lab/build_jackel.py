"""Build one translation unit at a time; no package installers or fast math."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['erf_cody.cpp', 'normaldistribution.cpp', 'rationalcubic.cpp',
           'LetsBeRational.cpp']


def build():
    out = ROOT / 'build'
    out.mkdir(exist_ok=True)
    compiler = os.environ.get('CXX', 'clang++')
    flags = ['-std=c++11', '-O2', '-DNDEBUG', '-ffp-contract=off']
    commands, objects = [], []
    for source in [ROOT/'vendor/jackel'/s for s in SOURCES] + [ROOT/'lab/jackel_cli.cpp']:
        obj = out / (source.stem + '.o')
        command = [compiler, *flags, '-c', str(source), '-o', str(obj)]
        subprocess.run(command, check=True)
        commands.append(command)
        objects.append(str(obj))
    binary = out/'jackel_cli'
    command = [compiler, *objects, '-o', str(binary)]
    subprocess.run(command, check=True)
    commands.append(command)
    manifest = {
        'compiler': subprocess.check_output([compiler, '--version'], text=True),
        'platform': platform.platform(), 'flags': flags, 'commands': commands,
        'upstream_commit': '83ae882df8e19323798c7ebfb8898f94d2d92ade',
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT/'vendor/jackel').iterdir())
                          if p.suffix in ('.cpp', '.h')},
        'wrapper_sha256': hashlib.sha256((ROOT/'lab/jackel_cli.cpp').read_bytes()).hexdigest(),
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'floating_environment': json.loads(subprocess.check_output([str(binary), '--environment'], text=True)),
    }
    (ROOT/'evidence/jackel-build-v1.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ('platform','flags','floating_environment')}, indent=2))


if __name__ == '__main__':
    build()
