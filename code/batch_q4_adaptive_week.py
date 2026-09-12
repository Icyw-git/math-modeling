"""Run only the four requested seven-day trials, then independent verification."""
import concurrent.futures
import subprocess
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES=('q2_neutral','q2_cvar','q3_neutral','q3_cvar')

def one(name):
    env=os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
    logs=ROOT/'tmp/q4_adaptive_week_logs'; logs.mkdir(exist_ok=True)
    with (logs/f'{name}.log').open('a',encoding='utf8') as stream:
        subprocess.run([sys.executable,str(ROOT/'code/run_q4_adaptive_week.py'),'--variant',name,'--end-day','38'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(ROOT/'code/verify_c_q4_cvar.py'),str(ROOT/'results/q4_adaptive_week'/name)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE',name,flush=True)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for task in concurrent.futures.as_completed([pool.submit(one,name) for name in NAMES]): task.result()
