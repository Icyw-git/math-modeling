"""Fixed-purchase replay; only terminal value differs from the saved reference."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import solve_c_q2_baseline as b
import solve_c_q2_enhanced as en
import q2_terminal_value as tv
from audit_c_attachments import matrix

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--linear',action='store_true'); ap.add_argument('--end-day',type=int,default=365)
    ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    if not 32<=args.end_day<=365: ap.error('end-day out of range')
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    source=b.ROOT/'CUMCM2026Problems'/'C题'/'附件'; refdir=b.ROOT/'results/q2_enhanced_grid193/enhanced'
    ref=pd.read_csv(refdir/'ledger.csv',parse_dates=['timestamp']); rd=pd.read_csv(refdir/'daily.csv')
    dates,load=matrix(source/'附件2.xlsx','小区负载'); _,pv=matrix(source/'附件2.xlsx','光伏发电实际功率'); load=load/6; pv=pv/6
    b.finite_nonnegative(load,'load'); b.finite_nonnegative(pv,'pv')
    prices=pd.read_excel(source/'附件1.xlsx').iloc[:,1].to_numpy(float)
    pl,pvp,_=en.forecast_archive(load,pv); pred=pl-pvp; net=load-pv
    files=[Path(__file__),Path(tv.__file__),Path(en.__file__),Path(b.__file__),source/'附件1.xlsx',source/'附件2.xlsx',refdir/'ledger.csv']
    config=dict(linear=args.linear,end_day=args.end_day,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},terminal_grid_kwh=400,control_grid_kwh=50,lookahead_days=1,scenario_count=7,approximation='Independent scenario recourse; optimistic two-stage proxy; next-day tail linear except annual closure',time_limit=60,gap=.001)
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    rows=[]; days=[]; e=1200.; grid=np.linspace(1200,10800,193); begun=time.perf_counter()
    for day in range(31,args.end_day):
        date=str(pd.Timestamp(dates[day]).date()); pool=en.residual_pool(day,net,pred); paths,prior,indices=en.representative_paths(pool)
        chunk=ref[ref.date==date].copy(); assert len(chunk)==144
        assert np.max(abs(chunk.load_kwh.to_numpy()-load[day]))<1e-6 and np.max(abs(chunk.pv_kwh.to_numpy()-pv[day]))<1e-6
        final=day==364; fallback=False; rec=[]; ts=time.perf_counter()
        if final:
            terminal=np.full(193,1e12); terminal[0]=0
        elif args.linear: terminal=-prices.min()/b.Config().eta*(grid-1200)
        else:
            knots,vals,rec,fallback=tv.curve(prices,pred[day,1][None,:]+paths,prior,day==363)
            terminal=np.interp(grid,knots,vals)
            pd.DataFrame(dict(energy=knots,value=vals)).to_csv(out/f'curve_{date}.csv',index=False)
        (out/f'solve_{date}.json').write_text(json.dumps(dict(records=rec,fallback=fallback),indent=2),encoding='utf8')
        observed=[]; q=chunk.purchase_kwh.to_numpy(); start=e
        for t,idx in enumerate(chunk.index):
            if t%36==0:
                scenarios,weights,_=en.scenario_update(pred[day,0],paths,prior,observed,t,max(10.,float(np.std(pool))))
                _,values=tv.value_table(q,prices,scenarios,weights,terminal,t)
            a=en.dp_dispatch(float(q[t]),float(load[day,t]),float(pv[day,t]),e,grid,values[t+1]/prices[t],143-t,final)
            for col,key in [('charge_kwh','c'),('discharge_kwh','d'),('emergency_kwh','r'),('unused_kwh','s'),('soc_end_kwh','end')]: chunk.loc[idx,col]=a[key]
            chunk.loc[idx,'soc_start_kwh']=e; chunk.loc[idx,'emergency_cost_yuan']=5*prices[t]*a['r']; e=a['end']; observed.append(float(net[day,t]))
        chunk['conversion_loss_kwh']=.1*chunk.charge_kwh+(1/.9-1)*chunk.discharge_kwh
        if args.linear:
            columns=['charge_kwh','discharge_kwh','emergency_kwh','unused_kwh','soc_end_kwh']
            assert np.max(abs(chunk[columns].to_numpy()-ref.loc[chunk.index,columns].to_numpy()))<=1e-6
        chunk.to_csv(out/f'day_{date}.csv',index=False); rows.append(chunk)
        record=dict(date=date,start_soc_kwh=start,end_soc_kwh=e,terminal_fallback=fallback,seconds=time.perf_counter()-ts)
        for key in ['purchase_kwh','charge_kwh','discharge_kwh','emergency_kwh','unused_kwh','plan_cost_yuan','emergency_cost_yuan','conversion_loss_kwh']: record[key]=float(chunk[key].sum())
        record['total_cost_yuan']=record['plan_cost_yuan']+record['emergency_cost_yuan']; days.append(record)
        print(f'{date} fallback={fallback} seconds={record["seconds"]:.1f}',flush=True)
    ledger=pd.concat(rows); daily=pd.DataFrame(days); checks=b.validate_ledger(ledger,b.Config())
    assert np.max(abs(ledger.purchase_kwh.to_numpy()-ref.iloc[:len(ledger)].purchase_kwh.to_numpy()))<1e-9
    assert np.max(abs(ledger.plan_cost_yuan-ledger.purchase_kwh*ledger.price_yuan_per_kwh))<1e-6
    if args.end_day==365: assert len(ledger)==48096 and abs(e-1200)<1e-6
    totals={k:float(daily[k].sum()) for k in ['total_cost_yuan','plan_cost_yuan','emergency_cost_yuan','unused_kwh','emergency_kwh','conversion_loss_kwh']}
    totals.update(start_soc=1200,end_soc=e,days=len(daily),intervals=len(ledger),fallback_days=int(daily.terminal_fallback.sum()),seconds=time.perf_counter()-begun,checks=checks,linear_replay_passed=args.linear)
    totals['reference_cost']=float(rd[rd.date.isin(daily.date)].total_cost_yuan.sum())
    ledger.to_csv(out/'ledger.csv',index=False); daily.to_csv(out/'daily.csv',index=False); b.emergency_events(ledger,b.Config()).to_csv(out/'emergency_events.csv',index=False)
    (out/'summary.json').write_text(json.dumps(totals,indent=2),encoding='utf8'); print(json.dumps(totals),flush=True)

if __name__=='__main__': main()
