"""Independent joint closed-loop Q3 experiment. No legacy files overwritten."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import q3_joint_policy as joint
ref=joint.ref


def run(data,out,end_day=365):
    out=Path(out)
    if (out/'summary.json').exists(): raise FileExistsError('Choose a new output directory')
    out.mkdir(parents=True,exist_ok=True); begun=time.perf_counter()
    for field in ['load','pv','fixed_price']: ref.old.b.finite_nonnegative(data[field],field)
    if min(data['fixed_price'])<=0: raise ValueError('Nonpositive tariff')
    pred,_,_=ref.en.forecast_archive(data['load'],data['pv']); e=1200.; rows=[]; days=[]; transactions=[]; solves=[]
    sources=[Path(__file__),Path(joint.__file__),Path(joint.compact.__file__),Path(joint.compact.dense.__file__),Path(ref.__file__),Path(ref.opt.__file__),Path(ref.old.__file__),Path(ref.en.__file__),Path(ref.old.b.__file__)]+[ref.old.SOURCE/f'附件{i}.xlsx' for i in (1,2,3)]
    cfg=dict(mode='joint_feedback_rollout',end_day=end_day,update_hours=[6,12,18],eta=.9,scenario_count=7,terminal_grid=joint.GRID.tolist(),terminal_candidate_time_limit=5,tree_time_limit=15,baseline_time_limit=10,search_steps=[50,20],information='current interval observed, future actual hidden',warmup='same January warmup, formal start 1200',hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    with (out/'searches.jsonl').open('w',encoding='utf8') as archive,(out/'terminal_curves.jsonl').open('w',encoding='utf8') as curves:
        for day in range(31,end_day):
            date=pd.Timestamp(data['dates'][day]); key=str(date.date()); price=data['fixed_price']; terminal=day==364; start_e=e; chunk=[]; q=None; changes=[]
            curve,curve_meta=joint.terminal_value(data,pred,day)
            curves.write(json.dumps(dict(date=key,values=curve.tolist(),**curve_meta))+'\n'); curves.flush()
            for t in range(144):
                if t in ref.old.BOUNDARIES:
                    start=time.perf_counter(); tr=joint.compact.build(data,pred,day,t//6); previous=None if q is None else q[t:].copy()
                    baseline,base_meta=ref.opt.solve(price[t:],tr['load_hat'][t:],tr['pv_hat'][t:],[(l[t:],v[t:]) for l,v in zip(tr['loads'],tr['pvs'])],tr['weights'],e,previous,terminal_target=1200 if terminal else None,time_limit=10,terminal_rate=ref.old.LAMBDA)
                    try:
                        solution,meta=joint.compact.optimize(tr,price,e,previous,terminal,15)
                        contracts=[n['q'] for n in solution]
                    except RuntimeError as exc:
                        contracts=[baseline[n['start']-t:].copy() for n in tr['nodes']]; meta=dict(status='tree fallback',fallback=True,reason=str(exc),gap=None)
                    best,candidates=joint.search(tr,contracts,baseline,e,price,curve,previous,terminal)
                    _,chosen,alpha,gain,label=best; new=chosen[0]
                    if q is None: q=new.copy(); q0=q.copy()
                    else:
                        inc=np.maximum(new-previous,0); dec=np.maximum(previous-new,0); q[t:]=new
                        changes.append(dict(start=t,increase=inc,decrease=dec))
                        for j in range(len(new)): transactions.append(dict(date=key,update_hour=t//6,interval=t+j,previous_kwh=previous[j],new_kwh=new[j],increase_kwh=inc[j],decrease_kwh=dec[j],price=price[t+j]))
                    record=dict(date=key,hour=t//6,status=meta['status'],fallback=bool(meta['fallback'] or base_meta['fallback']),gap=meta.get('gap'),seconds=time.perf_counter()-start,alpha=alpha,gain=gain,objective=best[0],selected=label,candidates=len(candidates))
                    solves.append(record)
                    archive.write(json.dumps(dict(**record,history=tr['history'],nodes=tr['nodes'],contracts=[v.tolist() for v in chosen],search=candidates,tree_solver=meta,baseline_solver=base_meta))+'\n'); archive.flush(); boundary=t
                actual=data['load'][day,:t+1]-data['pv'][day,:t+1]
                mean,correction=joint.forecast(tr['loads']-tr['pvs'],tr['weights'],actual,boundary,t)
                action=joint.feedback(q[t],data['load'][day,t],data['pv'][day,t],e,price,t,mean+gain*correction-q,alpha,curve,terminal)
                a={k:float(v) for k,v in action.items()}; ref.old.b.check_step(q[t],data['load'][day,t],data['pv'][day,t],e,a,ref.old.CFG)
                chunk.append(dict(date=key,interval=t,timestamp=date+pd.Timedelta(minutes=10*t),purchase_kwh=q0[t],adjusted_purchase_kwh=q[t],load_kwh=data['load'][day,t],pv_kwh=data['pv'][day,t],price_yuan_per_kwh=price[t],charge_kwh=a['c'],discharge_kwh=a['d'],emergency_kwh=a['r'],unused_kwh=a['s'],soc_start_kwh=e,soc_end_kwh=a['end'],alpha=alpha,gain=gain,forecast_next_net_kwh=float(mean[t+1]+gain*correction[t+1]) if t<143 else 0))
                e=a['end']
            frame=pd.DataFrame(chunk); bill,_=ref.old.compute_day_bill(frame,q0,changes,price)
            days.append(dict(date=key,**bill,start_soc_kwh=start_e,end_soc_kwh=e,unused_kwh=float(frame.unused_kwh.sum()),conversion_loss_kwh=float((.1*frame.charge_kwh+(1/.9-1)*frame.discharge_kwh).sum())))
            rows.extend(chunk)
            if day%15==1 or day==end_day-1:
                pd.DataFrame(days).to_csv(out/'daily_progress.csv',index=False); print(f'joint {key} cost={sum(x["total_fee_yuan"] for x in days):.2f}',flush=True)
    ledger=pd.DataFrame(rows); daily=pd.DataFrame(days); ledger.to_csv(out/'ledger.csv',index=False); daily.to_csv(out/'daily.csv',index=False); pd.DataFrame(transactions).to_csv(out/'transactions.csv',index=False); pd.DataFrame(solves).to_csv(out/'solves.csv',index=False)
    summary={k:float(daily[k].sum()) for k in ['total_fee_yuan','plan_fee_yuan','emergency_fee_yuan','emergency_kwh','adjustment_abs_kwh','unused_kwh','conversion_loss_kwh']}
    summary.update(days=len(days),intervals=len(rows),start_soc=1200,end_soc=e,seconds=time.perf_counter()-begun,fallbacks=sum(s['fallback'] for s in solves),max_gap=max((s['gap'] for s in solves if s['gap'] is not None),default=0))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    from verify_c_q3_refined import verify
    verify(out); print(summary,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--end-day',type=int,default=365); a=ap.parse_args(); run(ref.old.load_data(),a.output,a.end_day)
