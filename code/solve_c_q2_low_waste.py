"""Reduce unused energy through scenario-coupled day-ahead purchasing.

Forecasting, belief updates and actual Bellman controller are imported unchanged
from the verified enhanced model. Scenario recourse is only a planning
approximation; evaluated costs always come from causal actual execution.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import solve_c_q2_baseline as b
import solve_c_q2_enhanced as enhanced

ROOT=enhanced.ROOT
CFG=b.Config()


def check_scenarios(plan, demand, initial, cfg=CFG):
    q=plan['q']; tol=cfg.tolerance
    assert np.isfinite(q).all() and np.min(q)>=-tol
    for j,p in enumerate(plan['recourse']):
        c,d,r,s,e=[p[k] for k in ['c','d','r','s','e']]
        assert np.isfinite(np.r_[c,d,r,s,e]).all()
        assert min(c.min(),d.min(),r.min(),s.min())>=-tol
        assert max(c.max(),d.max())<=cfg.cap+tol
        assert e.min()>=cfg.emin-tol and e.max()<=cfg.emax+tol
        assert abs(e[0]-initial)<tol
        assert abs(q+r+d-demand[j]-c-s).max()<tol
        assert abs(np.diff(e)-cfg.eta*c+d/cfg.eta).max()<tol
        assert not ((c>tol)&((d>tol)|(r>tol))).any()


def scenario_plan(prices,demand,weights,initial,waste_price=None,cfg=CFG):
    prices=b.finite_nonnegative(prices,'prices')
    demand=np.asarray(demand,dtype=float); weights=b.finite_nonnegative(weights,'weights')
    b.validate_state(initial,cfg)
    if demand.ndim!=2 or demand.shape[1]!=len(prices) or not np.isfinite(demand).all():
        raise ValueError('Invalid net-demand scenarios')
    if weights.shape!=(len(demand),) or abs(weights.sum()-1)>1e-9 or np.min(prices)<=0:
        raise ValueError('Invalid prices/scenario probabilities')
    rho=float(.05*prices.min() if waste_price is None else waste_price)
    if not np.isfinite(rho) or rho<0: raise ValueError('Invalid waste price')
    hs=b.load_solver(); h=hs.Highs(); started=time.perf_counter()
    for name,value in [('output_flag',False),('threads',1),('random_seed',0),
                       ('time_limit',cfg.time_limit),('mip_rel_gap',cfg.mip_gap)]:
        h.setOptionValue(name,value)
    k,n=demand.shape; stride=6*n+2
    qmax=np.maximum(demand.max(axis=0),0)+cfg.cap
    for t in range(n):
        h.addVar(0,float(qmax[t])); h.changeColCost(t,float(prices[t]))

    def row(lb,ub,idx,values):
        h.addRow(float(lb),float(ub),len(idx),np.array(idx,dtype=np.int32),np.array(values,dtype=float))

    for j in range(k):
        off=n+j*stride
        # Per scenario r,c,d,s,E,z,terminal absolute deviation.
        rmax=np.maximum(demand[j],0)
        smax=qmax+np.maximum(-demand[j],0)+cfg.cap
        bounds=(list(zip(np.zeros(n),rmax))+[(0,cfg.cap)]*(2*n)
                +list(zip(np.zeros(n),smax))+[(cfg.emin,cfg.emax)]*(n+1)
                +[(0,1)]*n+[(0,hs.kHighsInf)])
        for lo,hi in bounds: h.addVar(float(lo),float(hi))
        h.changeColBounds(off+4*n,initial,initial)
        for t in range(n):
            h.changeColCost(off+t,float(weights[j]*5*prices[t]))
            # Penalize unused PLUS conversion losses, not artificial battery burning.
            h.changeColCost(off+n+t,float(weights[j]*rho*(1-cfg.eta)))
            h.changeColCost(off+2*n+t,float(weights[j]*rho*(1/cfg.eta-1)))
            h.changeColCost(off+3*n+t,float(weights[j]*rho))
            z=off+5*n+1+t
            row(demand[j,t],demand[j,t],[t,off+t,off+n+t,off+2*n+t,off+3*n+t],[1,1,-1,1,-1])
            row(0,0,[off+4*n+t+1,off+4*n+t,off+n+t,off+2*n+t],[1,-1,-cfg.eta,1/cfg.eta])
            row(-hs.kHighsInf,0,[off+n+t,z],[1,-cfg.cap])
            row(-hs.kHighsInf,cfg.cap,[off+2*n+t,z],[1,cfg.cap])
            row(-hs.kHighsInf,rmax[t],[off+t,z],[1,rmax[t]])
        u=off+6*n+1
        h.changeColCost(u,float(weights[j]*prices.min()/cfg.eta))
        row(-hs.kHighsInf,cfg.target,[off+5*n,u],[1,-1])
        row(-hs.kHighsInf,-cfg.target,[off+5*n,u],[-1,-1])
    # Constraint generation: retain a relaxation until physical mutual exclusion
    # is violated, then enforce integrality only at those scenario/time pairs.
    # A physically valid relaxed optimum is also feasible for the full MILP.
    enforced=set(); rounds=0
    while True:
        h.setOptionValue('time_limit',max(.01,cfg.time_limit-(time.perf_counter()-started)))
        h.run(); status=h.modelStatusToString(h.getModelStatus()); info=h.getInfo(); sol=h.getSolution()
        if not sol.value_valid or info.primal_solution_status!=hs.SolutionStatus.kSolutionStatusFeasible:
            break
        x=np.asarray(sol.col_value); violations=[]
        for j in range(k):
            off=n+j*stride
            bad=(x[off+n:off+2*n]>cfg.tolerance)&((x[off:off+n]>cfg.tolerance)|(x[off+2*n:off+3*n]>cfg.tolerance))
            violations.extend(off+5*n+1+int(t) for t in np.flatnonzero(bad))
        if not violations: break
        new=set(violations)-enforced
        if not new or time.perf_counter()-started>=cfg.time_limit:
            status='Physical mutual-exclusion check failed after '+status
            sol.value_valid=False
            break
        rounds+=1
        if rounds>=5:
            new={n+j*stride+5*n+1+t for j in range(k) for t in range(n)}-enforced
        for idx in sorted(new): h.changeColIntegrality(idx,hs.HighsVarType.kInteger)
        enforced.update(new)
    if not sol.value_valid or info.primal_solution_status!=hs.SolutionStatus.kSolutionStatusFeasible:
        # Safe scenario-feasible planning fallback; not a hidden successful solve.
        q=np.maximum(demand.max(axis=0),0)
        recourse=[dict(r=np.zeros(n),c=np.zeros(n),d=np.zeros(n),s=q-demand[j],e=np.full(n+1,initial)) for j in range(k)]
        result=dict(q=q,recourse=recourse,status='fallback:'+status,fallback=True,gap=None,objective=None)
    else:
        x=np.array(sol.col_value); recourse=[]
        for j in range(k):
            off=n+j*stride
            recourse.append(dict(r=x[off:off+n],c=x[off+n:off+2*n],d=x[off+2*n:off+3*n],
                                 s=x[off+3*n:off+4*n],e=x[off+4*n:off+5*n+1]))
        result=dict(q=x[:n],recourse=recourse,status=status,fallback=False,
                    gap=float(info.mip_gap) if np.isfinite(info.mip_gap) else (0.0 if status=='Optimal' else None),objective=float(info.objective_function_value))
    check_scenarios(result,demand,initial,cfg)
    result['q']=np.maximum(result['q'],0)
    result.update(seconds=time.perf_counter()-started,waste_price=rho,integer_pairs=len(enforced),refinement_rounds=rounds)
    return result


def run(dates,load,pv,prices,archives,start_soc=1200.,start_day=31,end_day=365,grid_size=193,waste_price=None):
    pl,pvp,selection=archives; net=load-pv; predicted=pl-pvp
    rows=[]; days=[]; plans=[]; beliefs=[]; scenario_rows=[]; e=start_soc
    for day in range(start_day,min(end_day,len(dates))):
        date=pd.Timestamp(dates[day]); pool=enhanced.residual_pool(day,net,predicted)
        paths,weights,indices=enhanced.representative_paths(pool)
        n=min(288,144*(len(dates)-day))
        demand=predicted[day].reshape(-1)[None,:n]+np.tile(paths,(1,2))[:,:n]
        p=scenario_plan(np.tile(prices,2)[:n],demand,weights,e,waste_price)
        for t in range(n):
            plans.append(dict(issue_date=str(date.date()),target_time=date+pd.Timedelta(minutes=t*10),
                virtual_next_day=t>=144,planned_purchase_kwh=float(p['q'][t]),
                forecast_load_kwh=float(pl[day].reshape(-1)[t]),forecast_pv_kwh=float(pvp[day].reshape(-1)[t])))
        for j,rec in enumerate(p['recourse']):
            for t in range(n):
                scenario_rows.append(dict(issue_date=str(date.date()),scenario=j,slot=t,weight=float(weights[j]),
                    demand_kwh=float(demand[j,t]),q=float(p['q'][t]),r=float(rec['r'][t]),c=float(rec['c'][t]),
                    d=float(rec['d'][t]),s=float(rec['s'][t]),e_start=float(rec['e'][t]),e_end=float(rec['e'][t+1])))
        q=p['q'][:144]; start_e=e; terminal=date==pd.Timestamp('2025-12-31'); observed=[]
        bandwidth=max(10.,float(np.std(pool))); begun=time.perf_counter()
        for t in range(144):
            if t%36==0:
                scenarios,w,correction=enhanced.scenario_update(predicted[day,0],paths,weights,observed,t,bandwidth)
                grid,values=enhanced.value_table(q,prices,scenarios,w,t,grid_size,terminal)
                beliefs.append(dict(date=str(date.date()),slot=t,weights=w.tolist(),residual_indices=indices,correction_kwh=correction))
            a=enhanced.dp_dispatch(float(q[t]),float(load[day,t]),float(pv[day,t]),e,grid,values[t+1]/prices[t],143-t,terminal)
            loss=(1-CFG.eta)*a['c']+(1/CFG.eta-1)*a['d']
            rows.append(dict(timestamp=date+pd.Timedelta(minutes=10*t),date=str(date.date()),
                interval_start=f'{t//6:02d}:{t%6*10:02d}',interval_end=f'{(t+1)//6:02d}:{(t+1)%6*10:02d}',
                evaluation=day>=31,price_yuan_per_kwh=float(prices[t]),predicted_load_kwh=float(pl[day,0,t]),
                predicted_pv_kwh=float(pvp[day,0,t]),load_kwh=float(load[day,t]),pv_kwh=float(pv[day,t]),
                purchase_kwh=float(q[t]),charge_kwh=a['c'],discharge_kwh=a['d'],emergency_kwh=a['r'],
                unused_kwh=a['s'],conversion_loss_kwh=loss,total_energy_waste_kwh=a['s']+loss,
                soc_start_kwh=e,soc_end_kwh=a['end'],plan_cost_yuan=float(prices[t]*q[t]),
                emergency_cost_yuan=float(5*prices[t]*a['r']),solver_status=p['status'],
                planner_fallback=p['fallback'],dispatch_fallback=False))
            e=a['end']; observed.append(float(net[day,t]))
        chunk=rows[-144:]
        days.append(dict(date=str(date.date()),evaluation=day>=31,start_soc_kwh=start_e,end_soc_kwh=e,
            solver_status=p['status'],solver_gap=p['gap'],planner_fallback=p['fallback'],
            solve_seconds=p['seconds'],controller_seconds=time.perf_counter()-begun,
            integer_pairs=p['integer_pairs'],refinement_rounds=p['refinement_rounds'],
            **{key:sum(r[key] for r in chunk) for key in ['purchase_kwh','emergency_kwh','unused_kwh','charge_kwh',
                'discharge_kwh','conversion_loss_kwh','total_energy_waste_kwh','plan_cost_yuan','emergency_cost_yuan']}))
        if day%7==3 or day==start_day or day==min(end_day,len(dates))-1:
            print(f'{date.date()} {p["status"]} solve={p["seconds"]:.2f}s unused={days[-1]["unused_kwh"]:.1f}',flush=True)
    f=pd.DataFrame(rows); daily=pd.DataFrame(days); daily['total_cost_yuan']=daily.plan_cost_yuan+daily.emergency_cost_yuan
    b.validate_ledger(f,CFG)
    if end_day==365: assert abs(e-1200)<1e-6 and len(f)==48096
    return f,daily,pd.DataFrame(plans),pd.DataFrame(scenario_rows),beliefs


def main():
    p=argparse.ArgumentParser(); p.add_argument('--end-day',type=int,default=365)
    p.add_argument('--grid-size',type=int,default=193); p.add_argument('--waste-price',type=float)
    p.add_argument('--output',type=Path,default=ROOT/'results'/'q2_low_waste'); args=p.parse_args()
    if not 32<=args.end_day<=365 or args.grid_size<3: p.error('invalid end day/grid')
    from audit_c_attachments import matrix
    source=ROOT/'CUMCM2026Problems'/'C题'/'附件'
    dates,l=matrix(source/'附件2.xlsx','小区负载'); _,v=matrix(source/'附件2.xlsx','光伏发电实际功率')
    l,v=l/6,v/6; price=pd.read_excel(source/'附件1.xlsx').iloc[:,1].to_numpy(float)
    archives=enhanced.forecast_archive(l,v)
    reference=ROOT/'results'/'q2_enhanced_grid193'/'enhanced'
    rf=pd.read_csv(reference/'ledger.csv'); rd=pd.read_csv(reference/'daily.csv')
    assert rf.soc_start_kwh.iloc[0]==1200
    config=dict(physical=asdict(CFG),grid_size=args.grid_size,end_day=args.end_day,
        waste_price=.05*float(price.min()) if args.waste_price is None else args.waste_price,
        objective='Plan cost + expected emergency cost + terminal penalty + rho*(unused + conversion loss)',
        planning_limit='Shared purchases; full-path recourse is optimistic approximation, NOT realized control',
        unchanged_control='Original forecast archive, scenario_update, value_table and dp_dispatch functions',
        input_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [source/'附件1.xlsx',source/'附件2.xlsx',reference/'ledger.csv',reference/'daily.csv',
             ROOT/'code'/'solve_c_q2_enhanced.py',ROOT/'code'/'solve_c_q2_baseline.py',Path(__file__)]})
    f,d,plans,scenarios,beliefs=run(dates,l,v,price,archives,end_day=args.end_day,grid_size=args.grid_size,waste_price=args.waste_price)
    summary,events,paired=enhanced.compare_summary(f,d,rd)
    subset=rf[rf.date.isin(d.date)]
    loss=float(f.conversion_loss_kwh.sum()); oldloss=float(.1*subset.charge_kwh.sum()+(1/.9-1)*subset.discharge_kwh.sum())
    summary.update(reference='Previous 50 kWh enhanced model',conversion_loss_kwh=loss,
        reference_conversion_loss_kwh=oldloss,total_energy_waste_kwh=float(f.total_energy_waste_kwh.sum()),
        reference_unused_kwh=float(subset.unused_kwh.sum()),
        unused_reduction_kwh=float(subset.unused_kwh.sum()-f.unused_kwh.sum()),
        unused_reduction_fraction=float(1-f.unused_kwh.sum()/subset.unused_kwh.sum()),
        total_energy_waste_reduction_kwh=float(subset.unused_kwh.sum()+oldloss-f.total_energy_waste_kwh.sum()),
        acceptance_same_boundaries=bool(abs(f.soc_start_kwh.iloc[0]-subset.soc_start_kwh.iloc[0])<1e-6 and
                                      abs(f.soc_end_kwh.iloc[-1]-subset.soc_end_kwh.iloc[-1])<1e-6))
    summary['accepted_cost_and_unused']=bool(summary['acceptance_same_boundaries'] and summary['saving_yuan']>=-1e-6 and summary['unused_reduction_kwh']>0)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,frame in [('ledger',f),('daily',d),('planning',plans),('scenario_recourse',scenarios),('emergency_events',events),('paired_daily',paired)]:
        frame.to_csv(args.output/f'{name}.csv',index=False,encoding='utf-8-sig')
    for name,data in [('config',config),('summary',summary),('belief_updates',beliefs)]:
        (args.output/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['selected_days','checks']},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
