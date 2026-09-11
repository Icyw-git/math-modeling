"""Conditional Q1 pilot with a HiGHS/SciPy reproducible solver path.

HiGHS is used when available to enforce binary charge/discharge modes.  The
SciPy fallback is accepted only when its LP solution has no simultaneous
charging and discharging.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
LOCAL_DEPS=ROOT/'tmp'/'c_solver_deps'
if LOCAL_DEPS.exists():
    sys.path.insert(0,str(LOCAL_DEPS))
try:
    import highspy
except ModuleNotFoundError:
    highspy = None
from scipy.optimize import linprog


def _solve_lp(price, load, pv, eta):
    """Solve the continuous relaxation for environments without highspy."""
    n = len(price)
    cap = 5000 / 6
    # q, c, d, s, E[0..n]
    total = 5 * n + 1
    objective = np.r_[price, np.zeros(total - n)]
    a_eq, b_eq = [], []
    for t in range(n):
        row = np.zeros(total)
        row[t], row[n+t], row[2*n+t], row[3*n+t] = 1, -1, 1, -1
        a_eq.append(row); b_eq.append(float(load[t] - pv[t]))
        row = np.zeros(total)
        row[4*n+t+1], row[4*n+t] = 1, -1
        row[n+t], row[2*n+t] = -eta, 1 / eta
        a_eq.append(row); b_eq.append(0.0)
    bounds = ([(0, None)] * n + [(0, cap)] * (2*n)
              + [(0, None)] * n + [(1200, 10800)] * (n+1))
    bounds[4*n] = (6000, 6000)
    bounds[5*n] = (6000, 6000)
    result = linprog(objective, A_eq=np.asarray(a_eq), b_eq=np.asarray(b_eq),
                     bounds=bounds, method='highs')
    if not result.success:
        raise RuntimeError(f'SciPy LP failed: {result.message}')
    values = result.x
    q, c, d, s = values[:n], values[n:2*n], values[2*n:3*n], values[3*n:4*n]
    e = values[4*n:5*n+1]
    simultaneous = int(((c > 1e-6) & (d > 1e-6)).sum())
    if simultaneous:
        raise RuntimeError('SciPy LP produced simultaneous charge/discharge; install highspy for MILP.')
    checks = dict(balance_max_abs_kwh=float(abs(q+pv+d-load-c-s).max()),
        soc_transition_max_abs_kwh=float(abs(np.diff(e)-eta*c+d/eta).max()),
        min_soc_kwh=float(e.min()), max_soc_kwh=float(e.max()),
        start_soc_kwh=float(e[0]), end_soc_kwh=float(e[-1]),
        max_charge_kw=float(c.max()*6), max_discharge_kw=float(d.max()*6),
        simultaneous_intervals=simultaneous,
        minimum_nonnegative_variable=float(np.min(np.r_[q,c,d,s])))
    assert checks['balance_max_abs_kwh'] < 1e-5
    assert checks['soc_transition_max_abs_kwh'] < 1e-5
    assert checks['min_soc_kwh'] >= 1200-1e-5 and checks['max_soc_kwh'] <= 10800+1e-5
    summary = dict(model='LP fallback', eta_charge=eta, eta_discharge=eta,
        status='Optimal', cost_yuan=float(price@q), purchase_kwh=float(q.sum()),
        charge_grid_side_kwh=float(c.sum()), discharge_grid_side_kwh=float(d.sum()),
        unused_kwh=float(s.sum()), checks=checks)
    frame = pd.DataFrame(dict(
        interval_start=[f'{t//6:02d}:{t%6*10:02d}' for t in range(n)],
        interval_end=[f'{(t+1)//6:02d}:{(t+1)%6*10:02d}' for t in range(n)],
        price_yuan_per_kwh=price, load_kwh=load, pv_kwh=pv, purchase_kwh=q,
        charge_kwh=c, discharge_kwh=d, unused_kwh=s,
        soc_start_kwh=e[:-1], soc_end_kwh=e[1:]))
    return summary, frame


def solve(price,load,pv,eta=0.9,integer=True):
    if highspy is None:
        return _solve_lp(price, load, pv, eta)
    n=len(price); cap=5000/6
    h=highspy.Highs()
    h.setOptionValue('output_flag',False)
    h.setOptionValue('mip_rel_gap',1e-9)
    h.setOptionValue('time_limit',120.0)
    h.setOptionValue('threads',1)
    h.setOptionValue('random_seed',0)
    bounds=([(0,highspy.kHighsInf)]*n+[(0,cap)]*(2*n)
            +[(0,highspy.kHighsInf)]*n+[(1200,10800)]*(n+1))
    for i,(lb,ub) in enumerate(bounds):
        h.addVar(lb,ub)
        if i<n:
            h.changeColCost(i,float(price[i]))
    h.changeColBounds(4*n,6000,6000)
    h.changeColBounds(5*n,6000,6000)
    for t in range(n):
        h.addRow(float(load[t]-pv[t]),float(load[t]-pv[t]),4,
                 np.array([t,n+t,2*n+t,3*n+t],dtype=np.int32),np.array([1.,-1.,1.,-1.]))
        h.addRow(0,0,4,np.array([4*n+t+1,4*n+t,n+t,2*n+t],dtype=np.int32),
                 np.array([1.,-1.,-eta,1/eta]))
    if integer:
        for t in range(n):
            idx=5*n+1+t
            h.addVar(0,1)
            h.changeColIntegrality(idx,highspy.HighsVarType.kInteger)
            h.addRow(-highspy.kHighsInf,0,2,np.array([n+t,idx],dtype=np.int32),np.array([1.,-cap]))
            h.addRow(-highspy.kHighsInf,cap,2,np.array([2*n+t,idx],dtype=np.int32),np.array([1.,cap]))
    h.run()
    status=h.modelStatusToString(h.getModelStatus())
    assert status=='Optimal',status
    x=np.array(h.getSolution().col_value)
    q,c,d,s,e=x[:n],x[n:2*n],x[2*n:3*n],x[3*n:4*n],x[4*n:5*n+1]
    checks=dict(balance_max_abs_kwh=float(abs(q+pv+d-load-c-s).max()),
        soc_transition_max_abs_kwh=float(abs(np.diff(e)-eta*c+d/eta).max()),
        min_soc_kwh=float(e.min()),max_soc_kwh=float(e.max()),start_soc_kwh=float(e[0]),
        end_soc_kwh=float(e[-1]),max_charge_kw=float(c.max()*6),max_discharge_kw=float(d.max()*6),
        simultaneous_intervals=int(((c>1e-6)&(d>1e-6)).sum()),
        minimum_nonnegative_variable=float(np.min(np.r_[q,c,d,s])))
    assert checks['balance_max_abs_kwh']<1e-5
    assert checks['soc_transition_max_abs_kwh']<1e-5
    assert e.min()>=1200-1e-5 and e.max()<=10800+1e-5
    if integer:
        assert checks['simultaneous_intervals']==0
    info=h.getInfo()
    summary=dict(model='MILP' if integer else 'LP relaxation',eta_charge=eta,eta_discharge=eta,
        status=status,cost_yuan=float(price@q),purchase_kwh=float(q.sum()),
        charge_grid_side_kwh=float(c.sum()),discharge_grid_side_kwh=float(d.sum()),
        unused_kwh=float(s.sum()),checks=checks)
    if integer:
        summary.update(mip_gap=float(info.mip_gap),mip_dual_bound=float(info.mip_dual_bound))
    frame=pd.DataFrame(dict(interval_start=[f'{t//6:02d}:{t%6*10:02d}' for t in range(n)],
        interval_end=[f'{(t+1)//6:02d}:{(t+1)%6*10:02d}' for t in range(n)],
        price_yuan_per_kwh=price,load_kwh=load,pv_kwh=pv,purchase_kwh=q,
        charge_kwh=c,discharge_kwh=d,unused_kwh=s,soc_start_kwh=e[:-1],soc_end_kwh=e[1:]))
    return summary,frame


def main():
    dest=ROOT/'results'/'q1_pilot'; dest.mkdir(parents=True,exist_ok=True)
    source=ROOT/'CUMCM2026Problems'/'C题'/'附件'/'附件1.xlsx'
    a=pd.read_excel(source).iloc[:,1:].to_numpy(float)
    price,load,pv=a[:,0],a[:,1]/6,a[:,2]/6
    lp,_=solve(price,load,pv,integer=False)
    main,frame=solve(price,load,pv)
    alternative,_=solve(price,load,pv,eta=np.sqrt(0.9))
    frame.to_csv(dest/'schedule.csv',index=False,encoding='utf-8-sig')
    blocks=[]
    for start in range(0,144,24):
        f=frame.iloc[start:start+24]
        blocks.append(dict(interval=f'{start//6:02d}:00-{(start+24)//6:02d}:00',
            charge_kwh=float(f.charge_kwh.sum()),discharge_kwh=float(f.discharge_kwh.sum())))
    selected=frame.iloc[[60,72,84,96,108,120]][['interval_start','interval_end','purchase_kwh']].to_dict('records')
    baseline=float(price@np.maximum(load-pv,0))
    report=dict(scope='Q1 conditional pilot, not final submission files',
        time_convention='Endpoint timestamps are treated as the preceding 10-minute interval average; output is 00:00-24:00. Template label conflict remains open.',
        storage_initial_final_kwh=6000,
        solver=('HiGHS '+highspy.Highs().version()) if highspy else 'SciPy linprog (HiGHS backend)',
        no_storage_cost_yuan=baseline,no_storage_purchase_kwh=float(np.maximum(load-pv,0).sum()),
        primary=main,lp_lower_bound=lp,efficiency_roundtrip_90_percent=alternative,
        saving_yuan=baseline-main['cost_yuan'],saving_fraction=1-main['cost_yuan']/baseline,
        primary_minus_lp_bound_yuan=main['cost_yuan']-lp['cost_yuan'],
        specified_intervals=selected,four_hour_charge_discharge=blocks)
    (dest/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
