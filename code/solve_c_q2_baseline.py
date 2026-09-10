"""Causal Q2 baseline. Input XLSX files are read only; all energies are AC-side kWh.

The real-time step is an interval-average approximation to instantaneous feedback,
not a claim that interval-average demand is available at the interval's beginning.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    eta: float = 0.9
    emin: float = 1200.0
    emax: float = 10800.0
    initial: float = 6000.0
    target: float = 6000.0
    power_kw: float = 5000.0
    dt_hours: float = 1 / 6
    emergency_multiplier: float = 5.0
    time_limit: float = 60.0
    mip_gap: float = 0.001
    seed: int = 0
    tolerance: float = 1e-6

    @property
    def cap(self):
        return self.power_kw * self.dt_hours


def finite_nonnegative(x, name):
    a = np.asarray(x, dtype=float)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f'{name}: expected finite nonnegative values')
    return a


def forecast(history_load, history_pv):
    """Only completed-day histories are accepted; no current-day arguments."""
    l = finite_nonnegative(history_load, 'historical load')
    v = finite_nonnegative(history_pv, 'historical PV')
    if l.ndim != 2 or l.shape != v.shape or l.shape[1] != 144:
        raise ValueError('Expected matching (completed_days, 144) histories')
    if len(l) == 0:
        return np.zeros(144), np.zeros(144), 'cold_start_no_history'
    lag = 7 if len(l) >= 7 else 1
    return l[-lag].copy(), v[-1].copy(), f'load_lag_{lag}_pv_lag_1'


def validate_state(e, cfg):
    if not np.isfinite(e) or not cfg.emin <= e <= cfg.emax:
        raise ValueError('Invalid initial/actual SOC')


def fallback_plan(load, pv, e, status, cfg):
    n = len(load)
    return dict(q=np.maximum(load - pv, 0), c=np.zeros(n), d=np.zeros(n),
                s=np.maximum(pv - load, 0), e=np.full(n + 1, e),
                status=status, fallback=True, gap=None, objective=None,
                plan_cost=None, terminal_penalty=None, seconds=0.0)


def load_solver():
    local = ROOT / 'tmp' / 'c_solver_deps'
    if local.exists() and str(local) not in sys.path:
        sys.path.insert(0, str(local))
    import highspy
    if not hasattr(highspy, 'Highs'):
        raise ImportError('HiGHS unavailable or local dependency permission denied')
    return highspy


def check_plan(p, price, load, pv, initial, cfg):
    q, c, d, s, e = [np.asarray(p[k]) for k in ('q', 'c', 'd', 's', 'e')]
    tol = cfg.tolerance
    assert all(np.isfinite(a).all() for a in (q, c, d, s, e))
    assert min(q.min(), c.min(), d.min(), s.min()) >= -tol
    assert max(c.max(), d.max()) <= cfg.cap + tol
    assert e.min() >= cfg.emin - tol and e.max() <= cfg.emax + tol
    assert abs(e[0] - initial) <= tol
    assert np.max(abs(q + pv + d - load - c - s)) <= tol
    assert np.max(abs(np.diff(e) - cfg.eta * c + d / cfg.eta)) <= tol
    assert not ((c > tol) & (d > tol)).any()


def plan_day(price, load, pv, initial, cfg=Config(), solver_factory=None):
    price = finite_nonnegative(price, 'price')
    load = finite_nonnegative(load, 'forecast load')
    pv = finite_nonnegative(pv, 'forecast PV')
    if price.ndim != 1 or len(price) == 0 or price.shape != load.shape or load.shape != pv.shape:
        raise ValueError('Price/load/PV shapes differ')
    if (price <= 0).any():
        raise ValueError('Baseline requires strictly positive prices')
    validate_state(initial, cfg)
    # Dependency/environment failures are fatal, not silently downgraded for all days.
    hs = load_solver()
    started = time.perf_counter()
    try:
        h = solver_factory() if solver_factory else hs.Highs()
        for name, value in [('output_flag', False), ('threads', 1), ('random_seed', cfg.seed),
                            ('time_limit', cfg.time_limit), ('mip_rel_gap', cfg.mip_gap)]:
            h.setOptionValue(name, value)
        n = len(price)
        # q,c,d,s,E,z,absolute terminal deviation
        z0, u = 5 * n + 1, 6 * n + 1
        bounds = ([(0, hs.kHighsInf)] * n + [(0, cfg.cap)] * (2 * n)
                  + [(0, hs.kHighsInf)] * n + [(cfg.emin, cfg.emax)] * (n + 1)
                  + [(0, 1)] * n + [(0, hs.kHighsInf)])
        for lb, ub in bounds:
            h.addVar(lb, ub)
        for t in range(n):
            h.changeColCost(t, float(price[t]))
            h.changeColIntegrality(z0 + t, hs.HighsVarType.kInteger)
        lam = float(price.min() / cfg.eta)
        h.changeColCost(u, lam)
        h.changeColBounds(4 * n, initial, initial)

        def row(lb, ub, idx, values):
            h.addRow(lb, ub, len(idx), np.asarray(idx, dtype=np.int32), np.asarray(values, dtype=float))

        for t in range(n):
            rhs = float(load[t] - pv[t])
            row(rhs, rhs, [t, n+t, 2*n+t, 3*n+t], [1, -1, 1, -1])
            row(0, 0, [4*n+t+1, 4*n+t, n+t, 2*n+t], [1, -1, -cfg.eta, 1/cfg.eta])
            row(-hs.kHighsInf, 0, [n+t, z0+t], [1, -cfg.cap])
            row(-hs.kHighsInf, cfg.cap, [2*n+t, z0+t], [1, cfg.cap])
        row(-hs.kHighsInf, cfg.target, [5*n, u], [1, -1])
        row(-hs.kHighsInf, -cfg.target, [5*n, u], [-1, -1])
        h.run()
        status = h.modelStatusToString(h.getModelStatus())
        solution = h.getSolution()
        info = h.getInfo()
        if not solution.value_valid or info.primal_solution_status != hs.SolutionStatus.kSolutionStatusFeasible:
            raise RuntimeError(f'No feasible incumbent: {status}')
        x = np.asarray(solution.col_value)
        p = dict(q=x[:n], c=x[n:2*n], d=x[2*n:3*n], s=x[3*n:4*n],
                 e=x[4*n:5*n+1], status=status, fallback=False,
                 gap=float(info.mip_gap), objective=float(info.objective_function_value))
        check_plan(p, price, load, pv, initial, cfg)
        # Clip only solver roundoff after checking the raw incumbent.
        p['q'] = np.maximum(p['q'], 0)
        p['e'] = np.clip(p['e'], cfg.emin, cfg.emax)
        p['plan_cost'] = float(price @ p['q'])
        p['terminal_penalty'] = lam * abs(p['e'][-1] - cfg.target)
    except Exception as exc:
        p = fallback_plan(load, pv, initial, f'fallback:{type(exc).__name__}:{exc}', cfg)
        check_plan(p, price, load, pv, initial, cfg)
        p['plan_cost'] = float(price @ p['q'])
        p['terminal_penalty'] = float(price.min()/cfg.eta * abs(initial-cfg.target))
    p['seconds'] = time.perf_counter() - started
    return p


def check_step(q, load, pv, e, a, cfg):
    c, d, r, s, end = [a[k] for k in ('c', 'd', 'r', 's', 'end')]
    tol = cfg.tolerance
    assert np.isfinite([c, d, r, s, end]).all()
    assert min(c, d, r, s) >= -tol
    assert max(c, d) <= cfg.cap + tol
    assert cfg.emin-tol <= end <= cfg.emax+tol
    assert abs(q + pv + d + r - load - c - s) <= tol
    assert abs(end - e - cfg.eta*c + d/cfg.eta) <= tol
    assert not (c > tol and (d > tol or r > tol))


def dispatch(q, load, pv, e, reference, cfg=Config()):
    finite_nonnegative([q, load, pv], 'current inputs')
    validate_state(e, cfg)
    if not np.isfinite(reference) or not cfg.emin <= reference <= cfg.emax:
        raise ValueError('Invalid reference SOC')
    b = q + pv - load
    try:
        if b >= 0:
            c = min(b, cfg.cap, (cfg.emax-e)/cfg.eta, max(0, (reference-e)/cfg.eta))
            d, r, s = 0.0, 0.0, b-c
        else:
            d = min(-b, cfg.cap, (e-cfg.emin)*cfg.eta, max(0, (e-reference)*cfg.eta))
            c, r, s = 0.0, -b-d, 0.0
        a = dict(c=c, d=d, r=r, s=s,
                 end=float(np.clip(e+cfg.eta*c-d/cfg.eta, cfg.emin, cfg.emax)), fallback=False)
        check_step(q, load, pv, e, a, cfg)
    except (AssertionError, ArithmeticError):
        a = dict(c=0.0, d=0.0, r=max(-b, 0), s=max(b, 0), end=e, fallback=True)
        check_step(q, load, pv, e, a, cfg)
    return a


def simulate(dates, load, pv, price, cfg=Config(), planner=plan_day, progress=False):
    load, pv, price = [finite_nonnegative(x, name) for x, name in
                       ((load, 'load'), (pv, 'PV'), (price, 'price'))]
    if load.shape != pv.shape or load.shape != (len(dates), 144) or price.shape != (144,):
        raise ValueError('Invalid simulation dimensions')
    if (price <= 0).any() or len(dates) == 0:
        raise ValueError('Empty dates or nonpositive prices')
    dates = pd.DatetimeIndex(dates)
    if not dates.equals(pd.date_range('2025-01-01', periods=len(dates), freq='D')):
        raise ValueError('Simulation must start January 1 and use consecutive days')
    e = cfg.initial
    records, days = [], []
    for day, date in enumerate(dates):
        lh, vh, method = forecast(load[:day], pv[:day])
        if day == 0:
            p = fallback_plan(lh, vh, e, 'cold_start_no_history', cfg)
            p.update(fallback=False, plan_cost=0.0, terminal_penalty=0.0)
        else:
            p = planner(price, lh, vh, e, cfg)
        start_e = e
        for t in range(144):
            a = dispatch(float(p['q'][t]), float(load[day, t]), float(pv[day, t]),
                         e, float(p['e'][t+1]), cfg)
            records.append(dict(timestamp=date+pd.Timedelta(minutes=10*t), date=date.strftime('%Y-%m-%d'),
                interval_start=f'{t//6:02d}:{t%6*10:02d}',
                interval_end=f'{(t+1)//6:02d}:{(t+1)%6*10:02d}',
                evaluation=bool(date >= pd.Timestamp('2025-02-01')),
                price_yuan_per_kwh=float(price[t]), predicted_load_kwh=float(lh[t]),
                predicted_pv_kwh=float(vh[t]), load_kwh=float(load[day,t]), pv_kwh=float(pv[day,t]),
                purchase_kwh=float(p['q'][t]), planned_charge_kwh=float(p['c'][t]),
                planned_discharge_kwh=float(p['d'][t]), planned_unused_kwh=float(p['s'][t]),
                reference_soc_start_kwh=float(p['e'][t]), reference_soc_end_kwh=float(p['e'][t+1]),
                charge_kwh=a['c'], discharge_kwh=a['d'], emergency_kwh=a['r'], unused_kwh=a['s'],
                soc_start_kwh=e, soc_end_kwh=a['end'],
                plan_cost_yuan=float(price[t]*p['q'][t]),
                emergency_cost_yuan=float(cfg.emergency_multiplier*price[t]*a['r']),
                forecast_method=method, solver_status=p['status'], planner_fallback=p['fallback'],
                dispatch_fallback=a['fallback']))
            e = a['end']
        chunk = records[-144:]
        days.append(dict(date=date.strftime('%Y-%m-%d'), evaluation=bool(date >= pd.Timestamp('2025-02-01')),
            start_soc_kwh=start_e, end_soc_kwh=e, solver_status=p['status'], planner_fallback=p['fallback'],
            solver_gap=p['gap'], solver_objective=p['objective'], terminal_penalty_yuan=p['terminal_penalty'],
            solve_seconds=p['seconds'],
            **{k:sum(r[k] for r in chunk) for k in ('purchase_kwh','charge_kwh','discharge_kwh',
               'emergency_kwh','unused_kwh','plan_cost_yuan','emergency_cost_yuan','dispatch_fallback')}))
        if progress and (day % 15 == 0 or day == len(dates)-1):
            print(f'{date.date()} {p["status"]} SOC={e:.3f}', flush=True)
    ledger, daily = pd.DataFrame(records), pd.DataFrame(days)
    daily['total_cost_yuan'] = daily.plan_cost_yuan + daily.emergency_cost_yuan
    return ledger, daily


def validate_ledger(f, cfg):
    balance = f.purchase_kwh + f.pv_kwh + f.discharge_kwh + f.emergency_kwh - f.load_kwh - f.charge_kwh - f.unused_kwh
    soc = f.soc_end_kwh - f.soc_start_kwh - cfg.eta*f.charge_kwh + f.discharge_kwh/cfg.eta
    checks = dict(balance_max_abs_kwh=float(abs(balance).max()), soc_max_abs_kwh=float(abs(soc).max()),
        continuity_max_abs_kwh=float(np.max(abs(f.soc_start_kwh.to_numpy()[1:]-f.soc_end_kwh.to_numpy()[:-1]), initial=0)),
        min_soc_kwh=float(min(f.soc_start_kwh.min(), f.soc_end_kwh.min())),
        max_soc_kwh=float(max(f.soc_start_kwh.max(), f.soc_end_kwh.max())),
        max_charge_kw=float(f.charge_kwh.max()/cfg.dt_hours),
        max_discharge_kw=float(f.discharge_kwh.max()/cfg.dt_hours),
        simultaneous_intervals=int(((f.charge_kwh>cfg.tolerance)&(f.discharge_kwh>cfg.tolerance)).sum()),
        emergency_while_charging=int(((f.charge_kwh>cfg.tolerance)&(f.emergency_kwh>cfg.tolerance)).sum()))
    assert max(checks[k] for k in ('balance_max_abs_kwh','soc_max_abs_kwh','continuity_max_abs_kwh')) <= cfg.tolerance
    assert checks['min_soc_kwh'] >= cfg.emin-cfg.tolerance and checks['max_soc_kwh'] <= cfg.emax+cfg.tolerance
    assert max(checks['max_charge_kw'],checks['max_discharge_kw']) <= cfg.power_kw+cfg.tolerance/cfg.dt_hours
    assert checks['simultaneous_intervals'] == checks['emergency_while_charging'] == 0
    for k in ('purchase_kwh','charge_kwh','discharge_kwh','emergency_kwh','unused_kwh'):
        assert np.isfinite(f[k]).all() and f[k].min() >= -cfg.tolerance
    expected = pd.date_range(f.timestamp.iloc[0], periods=len(f), freq='10min')
    assert pd.DatetimeIndex(f.timestamp).equals(expected)
    return checks


def emergency_events(f, cfg):
    events, current = [], None
    for r in f.itertuples():
        if r.emergency_kwh <= cfg.tolerance:
            if current: events.append(current)
            current = None
            continue
        if current and current['date'] != r.date:
            events.append(current)
            current = None
        if current is None:
            current = dict(date=r.date, start=r.interval_start, end=r.interval_end,
                           intervals=0, emergency_kwh=0.0, cost_yuan=0.0)
        current['end'] = r.interval_end
        current['intervals'] += 1
        current['emergency_kwh'] += r.emergency_kwh
        current['cost_yuan'] += r.emergency_cost_yuan
    if current: events.append(current)
    return pd.DataFrame(events, columns=['date','start','end','intervals','emergency_kwh','cost_yuan'])


def summarize(f, daily, cfg):
    checks = validate_ledger(f, cfg)
    evaluated = f[f.evaluation]
    selected = evaluated if len(evaluated) else f
    days = daily[daily.evaluation] if len(evaluated) else daily
    events = emergency_events(selected, cfg)
    for k in ('plan_cost_yuan','emergency_cost_yuan','purchase_kwh','charge_kwh','discharge_kwh','emergency_kwh','unused_kwh'):
        assert abs(selected[k].sum()-days[k].sum()) <= 1e-6
    assert abs(events.emergency_kwh.sum()-selected.emergency_kwh.sum()) <= 1e-6
    assert abs(events.cost_yuan.sum()-selected.emergency_cost_yuan.sum()) <= 1e-6
    out = dict(scope='Q2 causal feasible baseline; not a global optimum or submission workbook',
        days_simulated=len(daily), evaluated_days=int(daily.evaluation.sum()), evaluated_intervals=len(evaluated),
        metric_scope='February-December available portion' if len(evaluated) else 'January warmup smoke test',
        period_start_soc_kwh=float(selected.soc_start_kwh.iloc[0]), period_end_soc_kwh=float(selected.soc_end_kwh.iloc[-1]),
        plan_cost_yuan=float(selected.plan_cost_yuan.sum()), emergency_cost_yuan=float(selected.emergency_cost_yuan.sum()),
        total_cost_yuan=float(selected.plan_cost_yuan.sum()+selected.emergency_cost_yuan.sum()),
        **{k:float(selected[k].sum()) for k in ('purchase_kwh','charge_kwh','discharge_kwh','emergency_kwh','unused_kwh')},
        emergency_intervals=int((selected.emergency_kwh>cfg.tolerance).sum()), emergency_events=len(events),
        planner_fallback_days=int(daily.planner_fallback.sum()), dispatch_fallback_intervals=int(f.dispatch_fallback.sum()),
        solver_status_counts=daily.solver_status.value_counts().to_dict(), checks=checks,
        warmup_plan_cost_yuan=float(f.loc[~f.evaluation,'plan_cost_yuan'].sum()),
        warmup_emergency_cost_yuan=float(f.loc[~f.evaluation,'emergency_cost_yuan'].sum()),
        selected_days=daily[daily.date.isin(['2025-03-20','2025-06-21','2025-09-23','2025-12-21'])]
            .drop(columns=['solver_gap','solver_objective']).to_dict('records'))
    if len(daily)==365:
        assert len(evaluated)==48096 and int(daily.evaluation.sum())==334
    return out, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'q2_baseline')
    args = parser.parse_args()
    if not 1 <= args.days <= 365: parser.error('--days must be 1..365')
    from audit_c_attachments import matrix
    source = ROOT/'CUMCM2026Problems'/'C题'/'附件'
    inputs = [source/'附件1.xlsx', source/'附件2.xlsx']
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    dates, l = matrix(inputs[1], '小区负载')
    _, v = matrix(inputs[1], '光伏发电实际功率')
    price = pd.read_excel(inputs[0]).iloc[:,1].to_numpy(float)
    cfg = Config()
    hs = load_solver()
    f, daily = simulate(dates[:args.days], l[:args.days]/6, v[:args.days]/6, price, cfg, progress=True)
    summary, events = summarize(f, daily, cfg)
    assert hashes == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    config = dict(**asdict(cfg), terminal_penalty_lambda=float(price.min()/cfg.eta),
        highs_version=hs.Highs().version(), python_version=sys.version, numpy_version=np.__version__,
        pandas_version=pd.__version__, input_sha256=hashes, days=args.days,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        time_convention='Right endpoint represents preceding ten-minute interval average',
        feedback_assumption='Instantaneous within-interval feedback approximated by interval energy; no future interval access',
        emergency_assumption='Unlimited residual-deficit purchases, never concurrent charging',
        terminal_assumption='Soft target 6000 kWh; no hard daily or annual cycle',
        cold_start='January 1 zero planned purchases; January excluded from formal costs')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in [('ledger',f),('daily',daily),('emergency_events',events)]:
        data.to_csv(args.output/f'{name}.csv', index=False, encoding='utf-8-sig')
    for name,data in [('summary',summary),('config',config)]:
        (args.output/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
