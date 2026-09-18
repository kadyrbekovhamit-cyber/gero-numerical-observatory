#!/usr/bin/env python3
"""Replay official Strata release and three source states; Python standard library only."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,os,platform,shutil,subprocess,urllib.request,zipfile
ROOT=Path(__file__).resolve().parent
URL='https://github.com/OpenGamma/Strata/releases/download/v2.12.74/strata-report-tool-2.12.74.zip'
ZIP_SHA='42be9278993782e7b44c616a13c9bd1f34892ed05af32e854f107e28de3db4f3'
JAR_SHA='0fb4c6c778f25b7c6fe13c001f47a4e404868df55e4dd9841c752128bb864125'
SOURCE_SHA='a81577e70386f538ae7f1884c928ea7c43c4c765be829eef8e7ce7bd2825fba1'
FLAGS=['-XX:ActiveProcessorCount=1','-XX:+UseSerialGC','-Xint']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--release-zip',type=Path,help='Existing official ZIP; otherwise download to output/cache');ap.add_argument('--java-home',type=Path,help='Existing JDK directory with bin/java and bin/javac');ap.add_argument('--output',type=Path,default=ROOT/'replay-output');args=ap.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);cache=out/'cache';cache.mkdir(exist_ok=True)
 java=str(args.java_home/'bin/java') if args.java_home else shutil.which('java');javac=str(args.java_home/'bin/javac') if args.java_home else shutil.which('javac');assert java and javac,'Provide an installed JDK; this script does not install one.'
 z=args.release_zip
 if z is None:
  z=cache/'strata-report-tool-2.12.74.zip'
  if not z.exists():
   with urllib.request.urlopen(URL,timeout=60) as r:z.write_bytes(r.read())
 assert sha(z)==ZIP_SHA,'Release ZIP checksum mismatch'
 jar=cache/'strata-report-tool.jar'
 with zipfile.ZipFile(z) as archive:jar.write_bytes(archive.read('strata-report-tool-2.12.74/strata-report-tool.jar'))
 assert sha(jar)==JAR_SHA,'Dependency jar checksum mismatch'
 original=ROOT/'source/NormalFormulaRepository.java';assert sha(original)==SOURCE_SHA,'Original source changed'
 source=original.read_text();old='Math.min(Math.abs(initialNormalVol), 1e-10)';new='Math.max(Math.abs(initialNormalVol), 1e-10)';assert source.count(old)==1
 result={};env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
 version=subprocess.run([java,*FLAGS,'-version'],capture_output=True,text=True,env=env,check=True)
 for state in ['release','current','candidate','restored']:
  classes=out/state/'classes';classes.mkdir(parents=True,exist_ok=True);sources=[str(ROOT/'StrataGrid.java'),str(ROOT/'Repro.java')]
  if state!='release':
   f=out/state/'src/NormalFormulaRepository.java';f.parent.mkdir(parents=True,exist_ok=True);f.write_text(source.replace(old,new) if state=='candidate' else source);sources.insert(0,str(f))
  compile_run=subprocess.run([javac,*['-J'+x for x in FLAGS],'-cp',str(jar),'-d',str(classes),*sources],capture_output=True,text=True,env=env)
  (out/(state+'-compile.log')).write_text(compile_run.stdout+compile_run.stderr);assert compile_run.returncode==0,state+' compilation failed'
  minimal=subprocess.run([java,*FLAGS,'-cp',str(classes)+os.pathsep+str(jar),'Repro'],capture_output=True,text=True,env=env,check=True)
  (out/(state+'-minimal.txt')).write_text(minimal.stdout)
  run=subprocess.run([java,*FLAGS,'-cp',str(classes)+os.pathsep+str(jar),'StrataGrid'],capture_output=True,text=True,env=env)
  (out/(state+'-grid.stderr')).write_text(run.stderr);assert run.returncode==0,state+' runtime failed';f=out/(state+'-grid.csv');f.write_text(run.stdout)
  rows=list(csv.DictReader(run.stdout.splitlines()));assert len(rows)==1750;fail=sum(x['pass']=='false' for x in rows);assert fail==(0 if state=='candidate' else 864),(state,fail)
  result[state]={'minimal_example_stdout':minimal.stdout,'observations':len(rows),'failures':fail,'sha256':sha(f),'matches_archived_csv':f.read_bytes()==(ROOT/'observations'/f.name).read_bytes(),'groups':{g:{'observations':sum(x['group']==g for x in rows),'failures':sum(x['group']==g and x['pass']=='false' for x in rows)} for g in sorted({x['group'] for x in rows})}}
 assert (out/'release-grid.csv').read_bytes()==(out/'current-grid.csv').read_bytes()==(out/'restored-grid.csv').read_bytes()
 receipt={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'java_version':(version.stdout+version.stderr).strip(),'platform':platform.system(),'machine':platform.machine(),'java_flags':FLAGS,'source_pin':'987932ee95bf53e2baaff9a6b8e738a00f558b10','release':'2.12.74','release_zip_sha256':ZIP_SHA,'jar_sha256':JAR_SHA,'source_sha256':SOURCE_SHA,'results':result,'full_project_build':False,'full_upstream_test_suite':False,'note':'Official released jar and separately compiled target class with the same released dependencies. CSV exact matching is recorded; validation tolerances are explicit in StrataGrid.java.'}
 (out/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
