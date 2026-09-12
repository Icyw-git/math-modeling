"""Four predetermined seasonal weeks by four policy variants."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
SEASONS={'winter':(31,38),'spring':(134,141),'summer':(226,233),'autumn':(318,325)}
NAMES=('q2_neutral','q2_cvar','q3_neutral','q3_cvar')

def initial_soc(name,start):
    date={31:'2025-02-01',134:'2025-05-15',226:'2025-08-15',318:'2025-11-15'}[start]
    day=json.loads((ROOT/'results/q4_cvar'/name/f'day_{date}.json').read_text(encoding='utf8'))
    assert day['day']==start
    return day['ledger'][0]['soc_start_kwh']

def one(season,start,end,name):
    env=os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
    output=ROOT/'results/q4_adaptive_seasons'/season; logs=ROOT/'tmp/q4_season_logs'; logs.mkdir(exist_ok=True)
    with (logs/f'{season}_{name}.log').open('a',encoding='utf8') as stream:
        subprocess.run([sys.executable,str(ROOT/'code/run_q4_adaptive_period.py'),'--variant',name,'--output',str(output),'--start-day',str(start),'--end-day',str(end),'--initial-soc',str(initial_soc(name,start))],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(ROOT/'code/verify_q4_period.py'),str(output/name),'--prefix',str(ROOT/'results/q4_cvar'/name)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    return season,name

if __name__=='__main__':
    jobs=[(season,*bounds,name) for season,bounds in SEASONS.items() for name in NAMES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(one,*job) for job in jobs]):
            print('COMPLETE',*future.result(),flush=True)
