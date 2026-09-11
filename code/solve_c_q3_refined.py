"""Checked rolling optimization and historical conditional load errors for Q3."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
# Keep the runtime NumPy already imported; SciPy stays project-local.
sys.path.append(str(ROOT/'tmp/q3_deps'))
import pandas as pd
import solve_c_q3_latest as old
import solve_c_q2_enhanced as en
import q3_refined_contract as opt

def conditioned_paths(data,pred,day,load_hat,pv_hat,hour,count=7,adaptive=True):
    start=hour*6; history=list(range(max(8,day-28),day)); le=[]; ve=[]; prefixes=[]
    for h in history:
        error=data['load'][h]-pred[h,0]; le.append(error)
        ve.append(data['pv'][h]-old.pv_forecast_from_issue(data,h,hour))
        prefixes.append(float(error[max(0,start-12):start].mean()) if start else 0.)
    if not history: return [(load_hat,pv_hat)],np.ones(1),dict(history=[],beta_mean=0.)
    le=np.asarray(le); ve=np.asarray(ve); prior=np.ones(len(history))/len(history); beta=np.zeros(144); correction=np.zeros(144)
    if adaptive and start:
        x=np.asarray(prefixes); observed=float((data['load'][day,max(0,start-12):start]-load_hat[max(0,start-12):start]).mean())
        beta=np.clip(x@le/(x@x+len(x)*25.**2),0,1.5)
        correction=beta*observed; le=le-x[:,None]*beta
        width=max(25.,float(np.std(x))); posterior=np.exp(-.5*((x-observed)/width)**2)
        if posterior.sum()>0: prior=.1*prior+.9*posterior/posterior.sum()
    # Rank only the not-yet-delivered suffix; past PV is not a future error.
    order=np.argsort((le[:,start:]-ve[:,start:]).mean(axis=1),kind='stable'); groups=np.array_split(order,min(count,len(order)))
    paths=[]; weights=[]; selected=[]
    for group in groups:
        mass=prior[group].sum(); middle=np.searchsorted(np.cumsum(prior[group]),mass/2); i=int(group[min(middle,len(group)-1)])
        paths.append((np.maximum(load_hat+correction+le[i],0),np.maximum(pv_hat+ve[i],0))); weights.append(mass); selected.append(history[i])
    return paths,np.asarray(weights),dict(history=selected,beta_mean=float(beta[start:].mean()),correction_mean=float(correction[start:].mean()))

def run(data,out,mode='adaptive',end_day=365,grid_size=97,time_limit=10.,update_hours=(6,12,18)):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); begun=time.perf_counter()
    for field in ['load','pv','fixed_price']: old.b.finite_nonnegative(data[field],field)
    if min(data['fixed_price'])<=0: raise ValueError('Nonpositive tariff')
    pred,_,_=en.forecast_archive(data['load'],data['pv']); e=1200.; records=[]; daily=[]; transactions=[]; solves=[]
    inputs=[Path(__file__),Path(opt.__file__),Path(old.__file__),Path(en.__file__)]+[old.SOURCE/f'附件{i}.xlsx' for i in (1,2,3)]
    (out/'config.json').write_text(json.dumps(dict(mode=mode,end_day=end_day,grid_size=grid_size,time_limit=time_limit,scenario_count=7,update_hours=list(update_hours),ridge_scale_kwh=25,posterior_floor=.1,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),indent=2),encoding='utf8')
    for day in range(31,end_day):
        date=pd.Timestamp(data['dates'][day]); key=str(date.date()); price=data['fixed_price']; q=None; q0=None; changes=[]; chunk=[]; start_e=e; terminal=day==364
        for t in range(144):
            if t in old.BOUNDARIES:
                hour=t//6; issue=max(h for h in (0,*update_hours) if h<=hour)
                pvhat=old.pv_forecast_from_issue(data,day,issue); lh=pred[day,0]
                if mode in ('reference','solver'):
                    paths,w,hist=old.paired_paths(data,pred,day,lh,pvhat,issue,7); diagnostics=dict(history=hist)
                else: paths,w,diagnostics=conditioned_paths(data,pred,day,lh,pvhat,issue,7,mode=='adaptive')
                suffix=[(l[t:],v[t:]) for l,v in paths]
                if t==0 or hour in update_hours:
                    planner=old.contract_milp if mode=='reference' else opt.solve
                    extra={} if mode=='reference' else dict(terminal_rate=old.LAMBDA)
                    qnew,meta=planner(price[t:],lh[t:],pvhat[t:],suffix,w,e,None if q is None else q[t:],terminal_target=1200. if terminal else None,time_limit=.5 if mode=='reference' else time_limit,**extra)
                    if q is None: q=qnew.copy(); q0=q.copy()
                    else:
                        previous=q[t:].copy(); inc=np.maximum(qnew-previous,0); dec=np.maximum(previous-qnew,0); q[t:]=qnew
                        changes.append(dict(start=t,increase=inc,decrease=dec))
                        for j in range(len(qnew)):
                            transactions.append(dict(date=key,update_hour=hour,interval=t+j,previous_kwh=previous[j],new_kwh=qnew[j],increase_kwh=inc[j],decrease_kwh=dec[j],price=price[t+j]))
                    solves.append(dict(date=key,hour=hour,**meta,**diagnostics))
                grid,values=old.value_table(q[t:],price[t:],suffix,w,terminal=terminal,grid_size=grid_size); boundary=t
            # Use corrected continuous candidates and reachable annual boundary.
            a=en.dp_dispatch(float(q[t]),float(data['load'][day,t]),float(data['pv'][day,t]),e,grid,values[t-boundary+1]/price[t],143-t,terminal)
            chunk.append(dict(date=key,interval=t,timestamp=date+pd.Timedelta(minutes=10*t),purchase_kwh=q0[t],adjusted_purchase_kwh=q[t],load_kwh=data['load'][day,t],pv_kwh=data['pv'][day,t],price_yuan_per_kwh=price[t],charge_kwh=a['c'],discharge_kwh=a['d'],emergency_kwh=a['r'],unused_kwh=a['s'],soc_start_kwh=e,soc_end_kwh=a['end']))
            e=a['end']
        frame=pd.DataFrame(chunk); bill,_=old.compute_day_bill(frame,q0,changes,price); check=old.check_frame(frame)
        assert check['balance_max_abs_kwh']<1e-6 and check['soc_transition_max_abs_kwh']<1e-6
        daily.append(dict(date=key,**bill,start_soc_kwh=start_e,end_soc_kwh=e,unused_kwh=float(frame.unused_kwh.sum()),conversion_loss_kwh=float((.1*frame.charge_kwh+(1/.9-1)*frame.discharge_kwh).sum())))
        records.extend(chunk)
        if day%15==1 or day==end_day-1:
            pd.DataFrame(daily).to_csv(out/'daily_progress.csv',index=False)
            print(f'{mode} {key} cost={sum(x["total_fee_yuan"] for x in daily):.2f}',flush=True)
    ledger=pd.DataFrame(records); daily=pd.DataFrame(daily); pd.DataFrame(transactions,columns=['date','update_hour','interval','previous_kwh','new_kwh','increase_kwh','decrease_kwh','price']).to_csv(out/'transactions.csv',index=False)
    ledger.to_csv(out/'ledger.csv',index=False); daily.to_csv(out/'daily.csv',index=False); pd.DataFrame(solves).to_csv(out/'solves.csv',index=False)
    if end_day==365: assert len(ledger)==48096 and abs(e-1200)<1e-6
    summary={key:float(daily[key].sum()) for key in ['total_fee_yuan','plan_fee_yuan','emergency_fee_yuan','emergency_kwh','adjustment_abs_kwh','unused_kwh','conversion_loss_kwh']}
    summary.update(days=len(daily),intervals=len(ledger),start_soc=1200,end_soc=e,seconds=time.perf_counter()-begun,fallbacks=sum(s['fallback'] for s in solves),max_gap=max((s['gap'] for s in solves if s['gap'] is not None and np.isfinite(s['gap'])),default=None))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8'); print(summary,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['reference','solver','adaptive'],default='adaptive'); ap.add_argument('--end-day',type=int,default=365); ap.add_argument('--grid-size',type=int,default=97); ap.add_argument('--time-limit',type=float,default=10.); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    run(old.load_data(),a.output,a.mode,a.end_day,a.grid_size,a.time_limit)
