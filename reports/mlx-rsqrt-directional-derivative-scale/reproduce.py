"""Replay the frozen rsqrt audit with CMake/Ninja and a C++20 compiler on CPU."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

PACKAGE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', required=True, type=Path)
    ap.add_argument('--cmake', default=shutil.which('cmake'))
    ap.add_argument('--ninja', default=shutil.which('ninja'))
    args = ap.parse_args()
    if sys.version_info < (3, 12) or not args.cmake or not args.ninja:
        ap.error('Use Python >=3.12, mpmath 1.3.0, CMake >=3.25 and Ninja.')
    import mpmath
    assert mpmath.__version__ == '1.3.0'
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    assert not (work/'REPLAY_RECEIPT.json').exists(), 'Choose a new work directory.'
    manifest = json.loads((PACKAGE/'SOURCE_MANIFEST.json').read_text())
    paths = {}
    verified = {}
    for key, item in manifest.items():
        archive = PACKAGE/'sources'/item['archive']
        assert sha(archive) == item['sha256'], key
        with tarfile.open(archive) as stream:
            stream.extractall(work/'sources', filter='data')
        source = work/'sources'/item['directory']
        tree = json.loads((PACKAGE/'evidence'/item['git_tree']).read_text())
        count = 0
        for obj in tree['tree']:
            if obj['type'] != 'blob':
                continue
            p = source/obj['path']
            content = os.readlink(p).encode() if p.is_symlink() else p.read_bytes()
            actual = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            assert actual == obj['sha'], (key,obj['path'])
            count += 1
        paths[key] = source
        verified[key] = count
    driver = work/'driver'
    driver.mkdir(exist_ok=True)
    for name in ['check_rsqrt.py','rsqrt_probe.cpp','rsqrt_controls.cpp','rsqrt_compat.cpp','inputs.txt','oracle.json']:
        shutil.copyfile(PACKAGE/name, driver/name)
    (driver/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.25)\nproject(gero_rsqrt LANGUAGES C CXX)\nset(CMAKE_CXX_STANDARD 20)\nadd_subdirectory("'+str(paths['mlx'])+'" mlx-build)\n'+''.join('add_executable('+n+' '+n+'.cpp)\ntarget_link_libraries('+n+' PRIVATE mlx)\n' for n in ['rsqrt_probe','rsqrt_controls','rsqrt_compat']))
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',CMAKE_BUILD_PARALLEL_LEVEL='1')
    commands = []
    def call(cmd, label):
        commands.append({'label':label,'command':list(map(str,cmd))})
        with (work/(label+'.log')).open('w') as log:
            subprocess.run(list(map(str,cmd)),env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    command = [args.cmake,'-S',driver,'-B',work/'build','-G','Ninja','-DCMAKE_MAKE_PROGRAM='+args.ninja,'-DCMAKE_BUILD_TYPE=Release','-DFETCHCONTENT_SOURCE_DIR_FMT='+str(paths['fmt']),'-DFETCHCONTENT_SOURCE_DIR_JSON='+str(paths['json'])]
    command += ['-DMLX_BUILD_CPU=ON']+['-D'+k+'=OFF' for k in ['MLX_BUILD_METAL','MLX_BUILD_CUDA','MLX_BUILD_TESTS','MLX_BUILD_EXAMPLES','MLX_BUILD_GGUF','MLX_BUILD_SAFETENSORS','MLX_BUILD_PYTHON_BINDINGS','MLX_USE_CCACHE']]
    call(command,'configure')
    call([sys.executable,driver/'check_rsqrt.py','prepare'],'oracle')
    assert (driver/'oracle.json').read_bytes() == (PACKAGE/'oracle.json').read_bytes()
    assert (driver/'inputs.txt').read_bytes() == (PACKAGE/'inputs.txt').read_bytes()
    source_file = paths['mlx']/'mlx/primitives.cpp'
    original = source_file.read_bytes()
    try:
        for variant in ['original','candidate','restored']:
            if variant == 'candidate':
                call(['patch','-p1','-d',paths['mlx'],'-i',PACKAGE/'candidate.patch'],'apply-candidate')
            if variant == 'restored':
                source_file.write_bytes(original)
            call([args.cmake,'--build',work/'build','--target','rsqrt_probe','rsqrt_controls','rsqrt_compat','--parallel','1'],'build-'+variant)
            call([sys.executable,driver/'check_rsqrt.py','run',work/'build/rsqrt_probe',variant],'grid-'+variant)
            for target in ['rsqrt_controls','rsqrt_compat']:
                call([work/'build'/target],variant+'-'+target)
    finally:
        source_file.write_bytes(original)
    results = driver/'results'
    counts = []
    for variant in ['original','candidate','restored']:
        data = json.loads((results/(variant+'-summary.json')).read_text())
        counts.append(data['layouts']['flat']['failed_rows'])
        for layout in ['flat','row','column']:
            assert data['layouts'][layout]['same_as_flat']
            assert data['layouts'][layout]['forward_failures'] == 0
            assert (results/(variant+'-'+layout+'.csv')).read_bytes() == (PACKAGE/'observed'/(variant+'-'+layout+'.csv')).read_bytes()
    assert counts == [443,0,443]
    assert (work/'original-rsqrt_compat.log').read_bytes() == (work/'candidate-rsqrt_compat.log').read_bytes()
    assert 'checks=39 failures=0' in (work/'candidate-rsqrt_controls.log').read_text()
    receipt = {'completed_utc':datetime.now(timezone.utc).isoformat(),'source_git_blobs_verified':verified,'cases':2136,'failures_original_candidate_restored':counts,'csvs_byte_reproduced':9,'compatibility_scalar_outputs_unchanged':192,'ordinary_derivative_controls_passed':39,'source_restored':source_file.read_bytes()==original,'oracle_regenerated_exactly':True,'device':'CPU','gpu':False,'audio_playback':False,'commands':commands}
    (work/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
