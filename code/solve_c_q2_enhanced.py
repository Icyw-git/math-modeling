"""Q2: prequential forecast selection, quantile planning, causal stochastic DP.

No attachment 3/4 information is used. DP is an approximate fixed-belief
controller, not a full multistage globally optimal policy. All energy is kWh.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd

import solve_c_q2_baseline as base

ROOT = base.ROOT
CFG = base.Config()
VARIANTS = {
    'enhanced': dict(alpha=.8, horizon=288),
    'no_margin': dict(alpha=None, horizon=288),
    'one_day': dict(alpha=.8, horizon=144),
    'alpha65': dict(alpha=.65, horizon=288),
    'alpha90': dict(alpha=.9, horizon=288),
}


def candidate_forecasts(history, kind, ahead=1):
    """Forecast a complete future day; never accepts future-day values."""
    history = base.finite_nonnegative(history, 'history')
    n = len(history)
    if history.shape != (n, 144) or n == 0:
        raise ValueError('Need at least one complete historical day')
    if kind == 'load':
        def lag(k):
            idx = n + ahead - 1 - k
            return history[idx if 0 <= idx < n else n-1].copy()
        weekly = lag(7)
        same = [history[i] for i in range(n+ahead-1-7, max(-1,n-29), -7) if 0 <= i < n]
        bias = np.zeros(144)
        if n > 7:
            residual = np.array([history[i]-history[i-7] for i in range(max(7,n-7),n)])
            bias = np.repeat(np.median(residual.reshape(-1,24,6).mean(axis=2),axis=0),6)
        return np.maximum(np.array([weekly, lag(14), np.median(same,axis=0) if same else weekly,
                                    weekly + .5*bias]), 0)
    if kind == 'pv':
        return np.array([history[-1], history[-min(3,n):].mean(axis=0),
                         history[-min(7,n):].mean(axis=0)])
    raise ValueError(kind)


def forecast_archive(load, pv):
    """Prequential candidate scoring on prior 28 completed days (minimum available).

Entire archive is generated for convenience, but each row depends only on its
prefix. Future-mutation tests verify this property.
"""
    n = len(load)
    pred_l, pred_v = np.zeros((n,2,144)), np.zeros((n,2,144))
    candidate_l, candidate_v, selection = [], [], []
    for day in range(n):
        if day == 0:
            candidate_l.append(np.zeros((4,144)))
            candidate_v.append(np.zeros((3,144)))
            selection.append(dict(load_model=0,pv_model=0))
            continue
        first = max(1,day-28)
        if day > first:
            errors_l = np.abs(np.asarray(candidate_l[first:day])-load[first:day,None,:]).mean(axis=(0,2))
            errors_v = np.abs(np.asarray(candidate_v[first:day])-pv[first:day,None,:]).mean(axis=(0,2))
            il, iv = int(errors_l.argmin()), int(errors_v.argmin())
        else:
            il = iv = 0
        cl, cv = candidate_forecasts(load[:day],'load'), candidate_forecasts(pv[:day],'pv')
        candidate_l.append(cl)
        candidate_v.append(cv)
        pred_l[day,0], pred_v[day,0] = cl[il], cv[iv]
        pred_l[day,1] = candidate_forecasts(load[:day],'load',2)[il]
        pred_v[day,1] = candidate_forecasts(pv[:day],'pv',2)[iv]
        selection.append(dict(load_model=il,pv_model=iv))
    return pred_l, pred_v, selection


def residual_pool(day, actual, predicted):
    # Exclude cold-start/short-history days; all included errors were made before realization.
    first = max(8, day-28)
    pool = actual[first:day]-predicted[first:day,0]
    if len(pool)==0:
        pool = np.zeros((1,144))
    return pool


def representative_paths(pool, count=7):
    """Deterministic whole-day residual medoids by daily mean rank; no slot shuffling."""
    count = min(count,len(pool))
    order = np.argsort(pool.mean(axis=1),kind='stable')
    groups = np.array_split(order,count)
    selected = [int(group[len(group)//2]) for group in groups]
    return pool[selected].copy(), np.array([len(g)/len(pool) for g in groups]), selected


def scenario_update(point, paths, prior, observed_net, start, bandwidth):
    """Only the completed prefix may inform a new belief; current interval excluded."""
    if len(observed_net) != start:
        raise ValueError('Only completed prefix is allowed')
    weights = prior.copy()
    correction = 0.0
    if start:
        a = max(0,start-12)
        observed_error = np.asarray(observed_net[a:start])-point[a:start]
        distance = ((paths[:,a:start]-observed_error)/bandwidth)**2
        logw = np.log(prior)-.5*distance.mean(axis=1)
        logw -= logw.max()
        weights = np.exp(np.maximum(logw,-40))
        weights = .95*weights/weights.sum()+.05*prior
        correction = .5*float((observed_error-(weights@paths[:,a:start])).mean())
    future = point[None,:]+paths
    if start:
        future[:,start:] += correction*np.exp(-np.arange(144-start)/12)[None,:]
    return future, weights, correction


def value_table(q, prices, scenarios, weights, start=0, grid_size=49, terminal=False, cfg=CFG):
    """E[min after observing current demand], approximating fixed-belief future marginals.

    No perfect-path clairvoyance: a single shared continuation value is used.
    Posterior scenario weights are refreshed only at completed 6-hour boundaries.
    """
    grid = np.linspace(cfg.emin,cfg.emax,grid_size)
    n = len(q)
    value = np.empty((n+1,grid_size))
    value[n] = 1e12 if terminal else -float(np.min(prices)/cfg.eta)*(grid-cfg.emin)
    if terminal: value[n,0] = 0.0
    delta = grid[None,:]-grid[:,None]
    c = np.maximum(delta,0)/cfg.eta
    d = np.maximum(-delta,0)*cfg.eta
    physical = (c<=cfg.cap+1e-8)&(d<=cfg.cap+1e-8)
    for t in range(n-1,start-1,-1):
        expected = np.zeros(grid_size)
        for scenario, w in zip(scenarios,weights):
            b = q[t]-scenario[t]
            allowed = physical & ((c<=1e-8)|(c<=b+1e-8))
            r = np.maximum(-b+c-d,0)
            cost = cfg.emergency_multiplier*prices[t]*r+value[t+1][None,:]
            expected += w*np.where(allowed,cost,1e12).min(axis=1)
        value[t] = expected
    return grid,value


def dp_dispatch(q, load, pv, e, grid, continuation, remaining, terminal=False, cfg=CFG):
    """Continuous feasible action candidates, guided by an interpolated grid value."""
    base.finite_nonnegative([q,load,pv],'actual step')
    base.validate_state(e,cfg)
    b = q+pv-load
    low = max(cfg.emin,e-cfg.cap/cfg.eta)
    high = min(cfg.emax,e+cfg.eta*min(cfg.cap,max(b,0)))
    # Annual closure is feasible with free unused energy: cap reachable end states.
    if terminal:
        high = min(high,cfg.emin+remaining*cfg.cap/cfg.eta)
    if high < low-1e-7:
        raise RuntimeError('Annual terminal SOC is unreachable')
    candidates = np.unique(np.r_[grid[(grid>=low)&(grid<=high)],low,high,np.clip(e,low,high),
                                 np.clip(e-cfg.eta**-1*max(-b,0),low,high)])
    delta = candidates-e
    c, d = np.maximum(delta,0)/cfg.eta, np.maximum(-delta,0)*cfg.eta
    r = np.maximum(-b+c-d,0)
    continuation_cost = np.interp(candidates,grid,continuation)
    costs = cfg.emergency_multiplier*r+continuation_cost  # caller passes price-scaled continuation
    tied = np.flatnonzero(costs <= costs.min()+1e-9)
    # Equal economic value: avoid gratuitous cycling/dumping stored energy.
    best = int(tied[np.argmin(np.abs(delta[tied]))])
    end = float(candidates[best])
    a = dict(c=float(c[best]),d=float(d[best]),r=float(r[best]),
             s=float(max(b-c[best]+d[best],0)),end=end,fallback=False)
    base.check_step(q,load,pv,e,a,cfg)
    return a


def run_variant(dates, load, pv, prices, archives, warmup, name, grid_size=49, end_day=365):
    options = VARIANTS[name]
    pl,pvpred,selection = archives
    net = load-pv
    prediction = pl-pvpred
    e = float(warmup.soc_end_kwh.iloc[-1])
    rows, days, beliefs, plans = [], [], [], []
    for day in range(31,min(end_day,len(dates))):
        date = pd.Timestamp(dates[day])
        pool = residual_pool(day,net,prediction)
        paths,prior,indices = representative_paths(pool)
        margin = np.zeros(144) if options['alpha'] is None else np.maximum(np.quantile(pool,options['alpha'],axis=0),0)
        n = min(options['horizon'],144*(len(dates)-day))
        # The second-day plan is virtual lookahead only and never contracted early.
        future_l = pl[day].reshape(-1)[:n]+np.tile(margin,2)[:n]
        future_pv = pvpred[day].reshape(-1)[:n]
        p = base.plan_day(np.tile(prices,2)[:n],future_l,future_pv,e)
        q = p['q'][:144].copy()
        for k in range(n):
            plans.append(dict(issue_date=str(date.date()),target_time=date+pd.Timedelta(minutes=10*k),
                virtual_next_day=k>=144,forecast_load_kwh=float(pl[day].reshape(-1)[k]),
                forecast_pv_kwh=float(future_pv[k]),safety_margin_kwh=float(margin[k%144]),
                planned_purchase_kwh=float(p['q'][k]),planned_charge_kwh=float(p['c'][k]),
                planned_discharge_kwh=float(p['d'][k]),planned_unused_kwh=float(p['s'][k]),
                planned_soc_start_kwh=float(p['e'][k]),planned_soc_end_kwh=float(p['e'][k+1])))
        start_e = e
        terminal = date == pd.Timestamp('2025-12-31')
        observed = []
        bandwidth = max(10.0,float(np.std(pool)))
        begun = time.perf_counter()
        for t in range(144):
            if t%36 == 0:
                scenarios,weights,correction = scenario_update(prediction[day,0],paths,prior,observed,t,bandwidth)
                grid, values = value_table(q,prices,scenarios,weights,t,grid_size,terminal)
                beliefs.append(dict(date=str(date.date()),slot=t,weights=weights.tolist(),
                    residual_indices=indices,correction_kwh=correction,bandwidth_kwh=bandwidth))
            # Scale value to current price for dispatcher; immediate multiplier remains 5.
            a = dp_dispatch(float(q[t]),float(load[day,t]),float(pv[day,t]),e,
                            grid,values[t+1]/prices[t],143-t,terminal)
            rows.append(dict(timestamp=date+pd.Timedelta(minutes=10*t),date=str(date.date()),
                interval_start=f'{t//6:02d}:{t%6*10:02d}',interval_end=f'{(t+1)//6:02d}:{(t+1)%6*10:02d}',
                evaluation=True,price_yuan_per_kwh=float(prices[t]),
                predicted_load_kwh=float(pl[day,0,t]),predicted_pv_kwh=float(pvpred[day,0,t]),
                safety_margin_kwh=float(margin[t]),load_kwh=float(load[day,t]),pv_kwh=float(pv[day,t]),
                purchase_kwh=float(q[t]),charge_kwh=a['c'],discharge_kwh=a['d'],emergency_kwh=a['r'],
                unused_kwh=a['s'],soc_start_kwh=e,soc_end_kwh=a['end'],
                plan_cost_yuan=float(prices[t]*q[t]),emergency_cost_yuan=float(5*prices[t]*a['r']),
                solver_status=p['status'],planner_fallback=p['fallback'],dispatch_fallback=False,
                load_model=selection[day]['load_model'],pv_model=selection[day]['pv_model']))
            e = a['end']
            observed.append(float(net[day,t]))
        chunk = rows[-144:]
        days.append(dict(date=str(date.date()),evaluation=True,start_soc_kwh=start_e,end_soc_kwh=e,
            solver_status=p['status'],planner_fallback=p['fallback'],solver_gap=p['gap'],
            solve_seconds=p['seconds'],controller_seconds=time.perf_counter()-begun,
            **{key:sum(r[key] for r in chunk) for key in ('purchase_kwh','charge_kwh','discharge_kwh',
                'emergency_kwh','unused_kwh','plan_cost_yuan','emergency_cost_yuan','dispatch_fallback')}))
        if day%15==1 or day==min(end_day,len(dates))-1:
            print(f'{name} {date.date()} SOC={e:.2f} emergency={days[-1]["emergency_kwh"]:.2f}',flush=True)
    ledger = pd.DataFrame(rows)
    daily = pd.DataFrame(days)
    daily['total_cost_yuan'] = daily.plan_cost_yuan+daily.emergency_cost_yuan
    base.validate_ledger(ledger,CFG)
    if end_day==365:
        assert len(ledger)==48096 and abs(e-CFG.emin)<1e-6
    return ledger,daily,beliefs,pd.DataFrame(plans)


def compare_summary(ledger,daily,baseline):
    total = float(daily.total_cost_yuan.sum())
    baseline = baseline[baseline.date.isin(daily.date)]
    reference = float(baseline.total_cost_yuan.sum())
    events = base.emergency_events(ledger,CFG)
    result = dict(total_cost_yuan=total,baseline_cost_yuan=reference,saving_yuan=reference-total,
        saving_fraction=1-total/reference,days=len(daily),intervals=len(ledger),
        start_soc_kwh=float(ledger.soc_start_kwh.iloc[0]),end_soc_kwh=float(ledger.soc_end_kwh.iloc[-1]),
        plan_cost_yuan=float(ledger.plan_cost_yuan.sum()),emergency_cost_yuan=float(ledger.emergency_cost_yuan.sum()),
        **{k:float(ledger[k].sum()) for k in ('purchase_kwh','emergency_kwh','charge_kwh','discharge_kwh','unused_kwh')},
        emergency_events=len(events),emergency_intervals=int((ledger.emergency_kwh>1e-6).sum()),
        planner_fallback_days=int(daily.planner_fallback.sum()),
        max_solver_gap=float(daily.solver_gap.max()),checks=base.validate_ledger(ledger,CFG),
        load_mae_kw=float(abs(ledger.load_kwh-ledger.predicted_load_kwh).mean()*6),
        pv_mae_kw=float(abs(ledger.pv_kwh-ledger.predicted_pv_kwh).mean()*6),
        selected_days=daily[daily.date.isin(['2025-03-20','2025-06-21','2025-09-23','2025-12-21'])]
                      .drop(columns=['solve_seconds','controller_seconds']).to_dict('records'))
    paired = daily[['date','total_cost_yuan']].merge(baseline[['date','total_cost_yuan']],on='date',suffixes=('_new','_baseline'))
    paired['saving_yuan'] = paired.total_cost_yuan_baseline-paired.total_cost_yuan_new
    # Paired moving-block bootstrap, descriptive uncertainty, not a causal population guarantee.
    a=paired.saving_yuan.to_numpy(); rng=np.random.default_rng(0)
    draws=[]
    for _ in range(2000):
        starts=rng.integers(0,len(a),size=(len(a)+6)//7)
        idx=((starts[:,None]+np.arange(7))%len(a)).ravel()[:len(a)]
        draws.append(float(a[idx].sum()))
    result['paired_7day_bootstrap_saving_95pct_yuan']=np.quantile(draws,[.025,.975]).tolist()
    return result,events,paired


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=['enhanced','no_margin','one_day'])
    parser.add_argument('--end-day',type=int,default=365)
    parser.add_argument('--grid-size',type=int,default=49)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'q2_enhanced')
    args=parser.parse_args()
    if not 32<=args.end_day<=365 or args.grid_size<3: parser.error('invalid day/grid size')
    from audit_c_attachments import matrix
    source=ROOT/'CUMCM2026Problems'/'C题'/'附件'
    dates,l=matrix(source/'附件2.xlsx','小区负载')
    _,v=matrix(source/'附件2.xlsx','光伏发电实际功率')
    l,v=l/6,v/6
    prices=pd.read_excel(source/'附件1.xlsx').iloc[:,1].to_numpy(float)
    archives=forecast_archive(l,v)
    basepath=ROOT/'results'/'q2_baseline'
    ref=pd.read_csv(basepath/'ledger.csv',parse_dates=['timestamp'])
    warmup=ref[~ref.evaluation].copy()
    base.validate_ledger(ref,CFG)
    assert len(warmup)==31*144 and warmup.soc_end_kwh.iloc[-1]==1200
    bd=pd.read_csv(basepath/'daily.csv')
    manifest=dict(physical=asdict(CFG),variants={x:VARIANTS[x] for x in args.variants},grid_size=args.grid_size,
        scenario_count=7,history_window=28,belief_update_slots=[0,36,72,108],end_day=args.end_day,
        posterior_recent_slots=12,posterior_prior_floor=.05,residual_bias_shrinkage=.5,
        residual_bias_decay_slots=12,bandwidth_floor_kwh=10.,tie_break='Minimum cycling among equal-cost actions',
        forecast_candidates=dict(load=['lag7','lag14','same_weekday_median','lag7_half_hourly_recent_bias'],
                                 pv=['lag1','mean3','mean7']),
        initial_condition='Reuse unchanged baseline January trajectory; February 1 SOC=1200',
        terminal_condition='December 31 actual SOC=1200; forced reachable states, free unused energy allowed',
        dp_limit='Fixed-belief marginal Bellman approximation, refreshed using completed prefixes; not full scenario-tree optimality',
        terminal_value='Nonfinal day: -min(tariff)/eta*(E-Emin); final day hard Emin',
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [source/'附件1.xlsx',source/'附件2.xlsx',basepath/'ledger.csv',basepath/'daily.csv',Path(__file__)]},
        highs_version=base.load_solver().Highs().version())
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    records=[]
    for name in args.variants:
        ledger,daily,beliefs,plans=run_variant(dates,l,v,prices,archives,warmup,name,args.grid_size,args.end_day)
        summary,events,paired=compare_summary(ledger,daily,bd)
        out=args.output/name; out.mkdir(exist_ok=True)
        for filename,frame in [('ledger',ledger),('daily',daily),('emergency_events',events),('paired_daily',paired),('planning',plans)]:
            frame.to_csv(out/f'{filename}.csv',index=False,encoding='utf-8-sig')
        for filename,data in [('summary',summary),('belief_updates',beliefs)]:
            (out/f'{filename}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        records.append(dict(variant=name,**{k:summary[k] for k in ('total_cost_yuan','plan_cost_yuan',
            'emergency_cost_yuan','saving_yuan','saving_fraction','emergency_kwh','unused_kwh','start_soc_kwh','end_soc_kwh')}))
        print(json.dumps(records[-1],ensure_ascii=False),flush=True)
    pd.DataFrame(records).to_csv(args.output/'comparison.csv',index=False,encoding='utf-8-sig')


if __name__=='__main__':
    main()
