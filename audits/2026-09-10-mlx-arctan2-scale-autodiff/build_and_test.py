"""Rebuild three translation units sequentially; never edit the MLX source checkout.
Set MLX_SOURCE_ROOT and MLX_CPU_BUILD. Generated files stay in rerun/.
"""
import hashlib,json,os,resource,shlex,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'rerun'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
resource.setrlimit(resource.RLIMIT_CPU,(120,120))
def method(source, name):
    begin = source.index("std::vector<array> ArcTan2::" + name + "(")
    return source[begin:source.index("\n}", begin)+2]
def patch(source):
    vjp = """std::vector<array> ArcTan2::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 2);
  assert(cotangents.size() == 1);
  const auto& s = stream();
  auto [scale, norm] = arctan2_scale_norm(primals[0], primals[1], s);
  std::vector<array> grads;
  for (auto arg : argnums) {
    grads.push_back(arctan2_weighted_partial(
        cotangents[0],
        arg == 0 ? primals[1] : negative(primals[0], s),
        scale, norm, s));
  }
  return grads;
}"""
    jvp = """std::vector<array> ArcTan2::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 2);
  assert(tangents.size() == argnums.size());
  assert(!argnums.empty());
  const auto& s = stream();
  auto [scale, norm] = arctan2_scale_norm(primals[0], primals[1], s);
  auto term = [&](int i) {
    return arctan2_weighted_partial(
        tangents[i],
        argnums[i] == 0 ? primals[1] : negative(primals[0], s),
        scale, norm, s);
  };
  auto out = term(0);
  for (int i = 1; i < argnums.size(); ++i) {
    out = add(out, term(i), s);
  }
  return {out};
}"""
    source = source.replace(method(source, "vjp"), vjp, 1)
    source = source.replace(method(source, "jvp"), jvp, 1)
    return source.replace("std::vector<array> ArcTan2::vjp(",
        (ROOT/"scaled_partials.cpp.inc").read_text()+"\nstd::vector<array> ArcTan2::vjp(", 1)

def main():
    repo=Path(os.environ['MLX_SOURCE_ROOT']).resolve()
    build=Path(os.environ['MLX_CPU_BUILD']).resolve()
    archive=build/'mlx-build/libmlx.a'
    baseline=(ROOT/'baseline-primitives.cpp').read_text()
    actual=subprocess.check_output(['git','show','HEAD:mlx/primitives.cpp'],cwd=repo,text=True)
    assert actual==baseline, 'Source checkout HEAD differs from recorded baseline'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()=='ce916dbbcaa88e433b6fd1e60a17f766d49c27fe'
    OUT.mkdir(exist_ok=True)
    assert patch(baseline)==(ROOT/'patched-baseline-primitives.cpp').read_text()
    db=json.loads((build/'compile_commands.json').read_text())
    entry=next(x for x in db if x['file'].endswith('/mlx/primitives.cpp'))
    args=shlex.split(entry['command'])
    base=['-O0' if x=='-O3' else x for x in args[:args.index('-o')]]
    records=[]
    def run(cmd,label,expected=0):
        before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
        r=subprocess.run(cmd,cwd=entry['directory'],text=True,capture_output=True,timeout=60)
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        output=r.stdout+r.stderr
        # Do not expose machine-specific home directories in published logs.
        replacements={str(ROOT):'${PACKAGE}',str(repo):'${MLX_SOURCE_ROOT}',str(build):'${MLX_CPU_BUILD}'}
        for old,new in replacements.items():output=output.replace(old,new)
        (OUT/(label+'.log')).write_text(output)
        cmdtext=json.dumps(cmd)
        for old,new in replacements.items():cmdtext=cmdtext.replace(old,new)
        records.append(dict(label=label,command=json.loads(cmdtext),returncode=r.returncode,wall_seconds=time.monotonic()-start,cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
        (OUT/'build-results.json').write_text(json.dumps(records,indent=2)+'\n')
        rows=[json.loads(x[5:]) for x in r.stdout.splitlines() if x.startswith('CASE ')]
        if rows:(OUT/(label+'.json')).write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        print(label,r.returncode,*[x for x in r.stdout.splitlines() if x.startswith('SUMMARY')],flush=True)
        assert r.returncode==expected, output[-2000:]
    for name,src in [('test','native_regression.cpp'),('before','baseline-primitives.cpp'),('after','patched-baseline-primitives.cpp')]:
        run(base+['-o',str(OUT/(name+'.o')),'-c',str(ROOT/src)],'compile-'+name)
    for label,expected in [('before',1),('after',0)]:
        binary=OUT/('native-'+label)
        run(['/usr/bin/c++',str(OUT/'test.o'),str(OUT/(label+'.o')),str(archive),'-framework','Accelerate','-o',str(binary)],'link-'+label)
        run([str(binary)],'run-'+label,expected)
    comparison={label:json.loads((OUT/('run-'+label+'.json')).read_text())==json.loads((ROOT/('run-'+label+'.json')).read_text()) for label in ('before','after')}
    assert all(comparison.values()),'Native rerun differs from recorded rows'
    meta=dict(scope='Isolated translation-unit rebuild before existing CPU archive; not a full current-main build.',archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),rows_identical_to_original=comparison,baseline_revision='ce916dbbcaa88e433b6fd1e60a17f766d49c27fe',checks_per_variant=1368,before_failures=772,after_failures=0,numerical_threads=1,gpu_used=False,child_cpu_seconds=sum(x['cpu_seconds'] for x in records))
    (OUT/'publication-rerun.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
