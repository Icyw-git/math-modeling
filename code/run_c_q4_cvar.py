"""Six separately optimized experiments; atomic per-day checkpoints and resume."""
import argparse
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import q4_model as m

VARIANTS={f'{mode}_{kind}':(mode,0. if kind=='neutral' else .2,kind=='known') for mode in ('q2','q3') for kind in ('neutral','cvar','known')}


def dump(path,value):
    temp=path.with_suffix('.pending')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    temp.replace(path)


def run(name,output,end_day=365):
    mode,rho,known=VARIANTS[name]; data=m.load_data(mode); arc=m.archive(data)
    out=Path(output)/name; out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path(m.__file__),Path(m.en.__file__),Path(m.base.__file__),m.ROOT/'code/audit_c_attachments.py']
    hashes={str(p.relative_to(m.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    config=dict(variant=name,mode=mode,rho=rho,alpha=.9,known_price=known,grid_size=97,time_limit=30,scenario_days=28,source_hashes=hashes,input_hashes=data['hashes'],initial_soc=1200,annual_terminal_soc=1200,version=1)
    cp=out/'config.json'
    if cp.exists():
        if json.loads(cp.read_text(encoding='utf8'))!=config: raise ValueError('Resume inputs/config/code differ; use a new output directory')
    else: dump(cp,config)
    e=1200.; begun=time.perf_counter()
    for day in range(31,end_day):
        date=str(data['dates'][day].date()); dest=out/f'day_{date}.json'
        if dest.exists():
            saved=json.loads(dest.read_text(encoding='utf8'))
            assert saved['day']==day and abs(saved['ledger'][0]['soc_start_kwh']-e)<1e-6
            e=saved['ledger'][-1]['soc_end_kwh']; continue
        start_e=e; q=q0=units=None; changes=[]; ledger=[]; windows=[]; realized=0.; terminal=day==364
        rate=float((data['price'][day] if known else arc['price'][day]).min()/.9)
        for t in range(144):
            if t in (0,36,72,108):
                l,v,p,w,meta=m.scenarios(data,arc,day,t,mode,known)
                if t==0 or mode=='q3':
                    new,solver=m.contract(l,v,p,w,e,meta['load_point'],meta['pv_point'],None if q is None else q[t:],None if units is None else units[t:],realized,rho=rho,terminal=terminal,rate=rate)
                    if q is None: q=new.copy(); q0=q.copy(); units=q.copy()
                    else:
                        prev=q[t:].copy(); up=np.maximum(new-prev,0); down=np.maximum(prev-new,0)
                        units[t:]+=1.5*up-.5*down; q[t:]=new
                        changes.append(dict(start=t,previous=prev.tolist(),new=new.tolist(),up=up.tolist(),down=down.tolist()))
                else: solver=dict(status='contract_frozen',fallback=False,gap=None,seconds=0.)
                grid,val=m.values(q[t:],l,v,p,w,rate,terminal)
                windows.append(dict(start=t,**meta,**solver,price_scenario_min=float(p.min()),price_scenario_max=float(p.max()),price_scenario_sha256=hashlib.sha256(p.tobytes()).hexdigest()))
                boundary=t
            price=float(data['price'][day,t]); load=float(data['load'][day,t]); pv=float(data['pv'][day,t])
            a=m.en.dp_dispatch(float(q[t]),load,pv,e,grid,val[t-boundary+1]/price,143-t,terminal)
            row=dict(interval=t,load_kwh=load,pv_kwh=pv,price=price,price_forecast_0=float(arc['price'][day,t]),q0=float(q0[t]),q=float(q[t]),charged_units=float(units[t]),charge_kwh=a['c'],discharge_kwh=a['d'],emergency_kwh=a['r'],unused_kwh=a['s'],soc_start_kwh=e,soc_end_kwh=a['end'])
            ledger.append(row); e=a['end']; realized+=price*(units[t]+5*a['r'])
        up=sum(float(data['price'][day,c['start']:]@np.asarray(c['up'])) for c in changes)
        down=sum(float(data['price'][day,c['start']:]@np.asarray(c['down'])) for c in changes)
        daily=dict(date=date,total_fee=realized,initial_plan_fee=float(data['price'][day]@q0),increase_fee=1.5*up,refund=down,cancel_penalty=.5*down,emergency_fee=sum(r['price']*5*r['emergency_kwh'] for r in ledger),emergency_kwh=sum(r['emergency_kwh'] for r in ledger),unused_kwh=sum(r['unused_kwh'] for r in ledger),start_soc=start_e,end_soc=e,fallbacks=sum(x['fallback'] for x in windows))
        assert abs(realized-(daily['initial_plan_fee']+daily['increase_fee']-daily['refund']+daily['cancel_penalty']+daily['emergency_fee']))<1e-6
        dump(dest,dict(day=day,daily=daily,ledger=ledger,changes=changes,windows=windows))
        dump(out/'progress.json',dict(last_date=date,days=day-30,elapsed_session_seconds=time.perf_counter()-begun))
        if day%5==1 or day==end_day-1: print(f'{name} {date} day_fee={realized:.2f} fallback={daily["fallbacks"]}',flush=True)
    rows=[json.loads((out/f'day_{str(data["dates"][d].date())}.json').read_text(encoding='utf8')) for d in range(31,end_day)]
    daily=pd.DataFrame([r['daily'] for r in rows]); daily.to_csv(out/'daily.csv',index=False)
    pd.DataFrame([dict(date=r['daily']['date'],**x) for r in rows for x in r['ledger']]).to_csv(out/'ledger.csv',index=False)
    stats={c:float(daily[c].sum()) for c in ['total_fee','initial_plan_fee','increase_fee','refund','cancel_penalty','emergency_fee','emergency_kwh','unused_kwh','fallbacks']}
    stats.update(days=len(rows),intervals=len(rows)*144,mean_daily_fee=float(daily.total_fee.mean()),max_daily_fee=float(daily.total_fee.max()),realized_daily_cvar90=m.cvar(daily.total_fee),end_soc=e,complete=end_day==365)
    dump(out/'summary.json',stats); print(name,stats,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--variant',choices=list(VARIANTS),required=True); p.add_argument('--output',type=Path,default=m.ROOT/'results/q4_cvar'); p.add_argument('--end-day',type=int,default=365)
    a=p.parse_args(); run(a.variant,a.output,a.end_day)
