"""Parallel independent past-only terminal curves; not policy parallelism."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import q3_joint_policy as j

def initialize():
    global DATA,PRED
    DATA=j.ref.old.load_data(); PRED,_,_=j.ref.en.forecast_archive(DATA['load'],DATA['pv'])

def task(day):
    values,meta=j.terminal_value(DATA,PRED,day)
    return dict(day=day,values=values.tolist(),meta=meta)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--workers',type=int,default=3); ap.add_argument('--pilot',type=Path); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(j.__file__),Path(j.ref.opt.__file__)]},workers=a.workers,information='each date uses strictly prior historical observations')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    if a.pilot:
        import pandas as pd
        for line in (a.pilot/'terminal_curves.jsonl').read_text().splitlines():
            row=json.loads(line); day=(pd.Timestamp(row.pop('date'))-pd.Timestamp('2025-01-01')).days; values=row.pop('values')
            (a.output/f'{day:03}.json').write_text(json.dumps(dict(day=day,values=values,meta=row)),encoding='utf8')
    remaining=[d for d in range(31,365) if not (a.output/f'{d:03}.json').exists()]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize) as pool:
        futures={pool.submit(task,d):d for d in remaining}
        for i,future in enumerate(as_completed(futures)):
            row=future.result(); (a.output/f'{row["day"]:03}.json').write_text(json.dumps(row),encoding='utf8')
            if i%20==0 or i==len(remaining)-1: print('cached',i+1,'/',len(remaining),flush=True)
