"""Bounded independent process parallelism; no changes to previous experiments."""
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
NAMES=['q2_neutral','q2_cvar','q3_neutral','q3_cvar']


def one(name):
    env=os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
    logs=ROOT/'tmp/q4_logs'; logs.mkdir(parents=True,exist_ok=True)
    with (logs/f'{name}.log').open('a',encoding='utf8') as f:
        p=subprocess.run([sys.executable,str(ROOT/'code/run_c_q4_cvar.py'),'--variant',name],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError(f'{name} failed; inspect {logs/name}.log')
    subprocess.run([sys.executable,str(ROOT/'code/verify_c_q4_cvar.py'),str(ROOT/'results/q4_cvar'/name)]+(['--prefix',str(ROOT/'tmp/q4_week_ipm/q3_cvar')] if name=='q3_cvar' else []),cwd=ROOT,env=env,check=True)
    return name


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for task in concurrent.futures.as_completed([pool.submit(one,n) for n in NAMES]): print('COMPLETE',task.result(),flush=True)
