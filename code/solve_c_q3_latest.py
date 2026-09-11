"""Problem 3 rolling contract adjustment model.

Implements the latest repository modeling note:
  * 00:00/06:00/12:00/18:00 are the only contract publication boundaries;
  * each change is settled against the immediately previous contract version;
  * the main settlement is refund-plus-50%-penalty (net -0.5*p for a decrease);
  * a small paired-residual scenario MILP selects a shared future contract;
  * storage is dispatched causally from a common continuation value.

This is a reproducible rolling approximation, not a full multistage scenario
tree.  The reported cost is the realized settlement ledger, not the MILP
objective.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

import solve_c_q2_enhanced as en
import solve_c_q2_baseline as b
from audit_c_attachments import matrix, clock_minutes

ROOT = Path(__file__).resolve().parents[1]
N = 144
DT = 1/6
CFG = b.Config()
CAP, EMIN, EMAX = CFG.cap, CFG.emin, CFG.emax
ETA = CFG.eta
# The latest Q3 report explicitly defines this as a continuation-value
# regularizer inherited from Q2, not as a physical cash settlement value.
SOURCE = ROOT / "CUMCM2026Problems" / "C题" / "附件"
fixed = pd.read_excel(SOURCE / "附件1.xlsx")
FIXED_PRICE = fixed.iloc[:, 1].to_numpy(float)
LAMBDA = float(np.min(FIXED_PRICE) / ETA)
BOUNDARIES = (0, 36, 72, 108)


def load_data():
    dates, load_kw = matrix(SOURCE / "附件2.xlsx", "小区负载")
    _, pv_kw = matrix(SOURCE / "附件2.xlsx", "光伏发电实际功率")
    raw = pd.read_excel(SOURCE / "附件3.xlsx")
    issue_days = pd.to_datetime(raw.iloc[:, 0].replace("", np.nan).ffill())
    issue_hours = np.asarray([clock_minutes(v)//60 for v in raw.iloc[:, 1]], dtype=int)
    forecast = raw.iloc[:, 2:].apply(pd.to_numeric, errors="raise").to_numpy(float)
    lookup = {}
    for row, (day, hour) in enumerate(zip(issue_days, issue_hours)):
        lookup[(pd.Timestamp(day).normalize(), int(hour))] = forecast[row].copy()
    return dict(dates=pd.to_datetime(dates), load=load_kw/6, pv=pv_kw/6,
                fixed_price=FIXED_PRICE.copy(), forecast=lookup)


def pv_forecast_from_issue(data, day_index, issue_hour):
    """Causal nonnegative linear interpolation of the supplied hourly PV."""
    day = pd.Timestamp(data["dates"][day_index]).normalize()
    hourly = np.asarray(data["forecast"][(day, int(issue_hour))], float) / 6
    if issue_hour == 0:
        anchor = float(data["pv"][day_index-1, -1]) if day_index else float(hourly[0])
    else:
        anchor = float(data["pv"][day_index, issue_hour*6-1])
    out = np.empty(N)
    xp = np.arange(25, dtype=float)
    fp = np.r_[anchor, hourly]
    for t in range(N):
        delta = (t+1)/6 - issue_hour
        if delta <= 0:
            out[t] = data["pv"][day_index, t]
        else:
            out[t] = np.interp(min(delta, 24.0), xp, fp)
    return np.maximum(out, 0.0)


def paired_paths(data, pred_l, day, load_hat, pv_hat, issue_hour, count=7):
    """Return representative paired load/PV scenarios and weights."""
    first = max(8, day - 28)
    histories = list(range(first, day))
    if not histories:
        return [(load_hat.copy(), pv_hat.copy())], np.array([1.0]), []
    net_errors = []
    pairs = []
    for h in histories:
        hist_l = pred_l[h, 0]
        hist_v = pv_forecast_from_issue(data, h, issue_hour)
        le = data["load"][h] - hist_l
        ve = data["pv"][h] - hist_v
        pairs.append((le, ve, h))
        net_errors.append((le - ve).mean())
    order = np.argsort(np.asarray(net_errors), kind="stable")
    groups = np.array_split(order, min(count, len(order)))
    selected, weights, indices = [], [], []
    for group in groups:
        idx = int(group[len(group)//2])
        le, ve, hist_day = pairs[idx]
        selected.append((np.maximum(load_hat + le, 0.0),
                         np.maximum(pv_hat + ve, 0.0)))
        weights.append(len(group) / len(order))
        indices.append(int(hist_day))
    return selected, np.asarray(weights, dtype=float), indices


def _add_row(rows, cols, vals, lo, hi, entries):
    rows.append((float(lo), float(hi), entries))


def contract_milp(price, load_hat, pv_hat, scenarios, weights, e0,
                  previous_q=None, terminal_target=None, time_limit=60.0):
    """Optimize one rolling shared contract and scenario recourse.

    Previous contract is fixed on the suffix and adjustment variables are
    charged only for the current version.  Scenario recourse uses binaries to
    prohibit simultaneous charge/discharge and emergency charging.
    """
    price = np.asarray(price, float)
    load_hat = np.asarray(load_hat, float)
    pv_hat = np.asarray(pv_hat, float)
    n = len(price)
    k = len(scenarios)
    if previous_q is None:
        previous_q = np.zeros(n)
        first = True
    else:
        previous_q = np.asarray(previous_q, float)
        first = False
    if len(previous_q) != n:
        raise ValueError("previous_q length mismatch")
    qmax = np.maximum.reduce([np.maximum(L - V, 0.0) for L, V in scenarios]) + CAP
    # shared q plus current-version u/v, then scenario c,d,s,r,e,z
    n_shared = n if first else 3*n
    stride = 6*n + 1
    total = n_shared + k*stride
    q0 = 0
    u0 = n if not first else None
    v0 = 2*n if not first else None
    objective = np.zeros(total)
    if first:
        objective[q0:q0+n] = price
    else:
        objective[u0:u0+n] = 1.5*price
        objective[v0:v0+n] = -0.5*price
    rows = []
    # each row is (lower, upper, [(column, coefficient), ...])
    for t in range(n):
        if not first:
            _add_row(rows, [], [], previous_q[t], previous_q[t],
                     [(q0+t, 1), (u0+t, -1), (v0+t, 1)])
    for j, ((loads, pvs), weight) in enumerate(zip(scenarios, weights)):
        off = n_shared + j*stride
        c0, d0, s0, r0, e0i, z0 = off, off+n, off+2*n, off+3*n, off+4*n, off+5*n+1
        objective[r0:r0+n] = objective[r0:r0+n] + weight*5*price
        if terminal_target is None:
            objective[e0i+n] += -weight*LAMBDA
        for t in range(n):
            _add_row(rows, [], [], loads[t]-pvs[t], loads[t]-pvs[t],
                     [(q0+t, 1), (r0+t, 1), (d0+t, 1), (c0+t, -1), (s0+t, -1)])
            _add_row(rows, [], [], 0, 0,
                     [(e0i+t+1, 1), (e0i+t, -1), (c0+t, -ETA), (d0+t, 1/ETA)])
            _add_row(rows, [], [], -np.inf, 0, [(c0+t, 1), (z0+t, -CAP)])
            _add_row(rows, [], [], -np.inf, CAP, [(d0+t, 1), (z0+t, CAP)])
            # r <= L(1-z): emergency power cannot be used while charging.
            _add_row(rows, [], [], -np.inf, float(max(loads[t], 0.0)),
                     [(r0+t, 1), (z0+t, float(max(loads[t], 0.0)))])

    A = lil_matrix((len(rows), total), dtype=float)
    lower = np.empty(len(rows)); upper = np.empty(len(rows))
    for r, (lo, hi, entries) in enumerate(rows):
        lower[r], upper[r] = lo, hi
        for col, val in entries:
            A[r, col] += val
    lower_vars = np.zeros(total); upper_vars = np.full(total, np.inf)
    upper_vars[q0:q0+n] = qmax
    if not first:
        upper_vars[u0:u0+n] = qmax + previous_q
        upper_vars[v0:v0+n] = previous_q
    integrality = np.zeros(total, dtype=int)
    for j, ((loads, pvs), _) in enumerate(zip(scenarios, weights)):
        off = n_shared + j*stride
        c0, d0, s0, r0, e0i, z0 = off, off+n, off+2*n, off+3*n, off+4*n, off+5*n+1
        upper_vars[c0:c0+2*n] = CAP
        upper_vars[r0:r0+n] = np.maximum(loads, 0.0)
        lower_vars[e0i:e0i+n+1] = EMIN; upper_vars[e0i:e0i+n+1] = EMAX
        upper_vars[z0:z0+n] = 1; integrality[z0:z0+n] = 1
        lower_vars[e0i] = upper_vars[e0i] = float(e0)
        if terminal_target is not None:
            lower_vars[e0i+n] = upper_vars[e0i+n] = float(terminal_target)
    result = milp(objective, integrality=integrality, bounds=Bounds(lower_vars, upper_vars),
                  constraints=LinearConstraint(A.tocsr(), lower, upper),
                  options={"time_limit": float(time_limit), "mip_rel_gap": 1e-3})
    # HiGHS may return status=1 on a time limit while still providing a valid
    # feasible incumbent; that incumbent is usable and its status is recorded.
    if result.x is None or not np.isfinite(result.x).all():
        # Feasible causal fallback required by the latest modeling report:
        # use the predicted nonnegative net deficit at 00:00, and retain the
        # last valid contract at later update boundaries.
        fallback_q = previous_q.copy() if not first else np.maximum(load_hat - pv_hat, 0.0)
        return fallback_q, {"status": str(result.message), "objective": None,
                            "gap": None, "fallback": True}
    q = result.x[q0:q0+n].copy()
    return q, {"status": str(result.message), "objective": float(result.fun),
               "gap": float(getattr(result, "mip_gap", np.nan)), "fallback": False}


def value_table(q, price, scenarios, weights, terminal=False, grid_size=193):
    grid = np.linspace(EMIN, EMAX, grid_size)
    n = len(q); values = np.empty((n+1, grid_size))
    if terminal:
        values[n] = 1e12; values[n, 0] = 0.0
    else:
        values[n] = -LAMBDA*(grid-EMIN)
    delta = grid[None, :] - grid[:, None]
    c = np.maximum(delta, 0)/ETA; d = np.maximum(-delta, 0)*ETA
    physical = (c <= CAP+1e-8) & (d <= CAP+1e-8)
    for t in range(n-1, -1, -1):
        expected = np.zeros(grid_size)
        for (loads, pvs), w in zip(scenarios, weights):
            balance = q[t] + pvs[t] - loads[t]
            allowed = physical & ((c <= 1e-8) | (c <= balance+1e-8))
            r = np.maximum(-balance+c-d, 0)
            expected += w*np.where(allowed, 5*price[t]*r + values[t+1][None, :], 1e12).min(axis=1)
        values[t] = expected
    return grid, values


def dispatch(q, load, pv, e, grid, continuation, price, terminal=False, remaining=1):
    bbal = q + pv - load
    low = max(EMIN, e-CAP/ETA)
    high = min(EMAX, e+ETA*min(CAP, max(bbal, 0.0)))
    if terminal:
        high = min(high, EMIN + remaining*CAP/ETA)
    candidates = np.unique(np.r_[grid[(grid >= low) & (grid <= high)], low, high, e,
                                 np.clip(e-max(-bbal, 0)/ETA, low, high)])
    delta = candidates-e
    c = np.maximum(delta, 0)/ETA; d = np.maximum(-delta, 0)*ETA
    r = np.maximum(-bbal+c-d, 0)
    cost = 5*price*r + np.interp(candidates, grid, continuation)
    best = np.flatnonzero(cost <= cost.min()+1e-9)
    idx = int(best[np.argmin(np.abs(delta[best]))])
    e_next = float(candidates[idx])
    action = dict(c=float(c[idx]), d=float(d[idx]), r=float(r[idx]),
                  s=float(max(bbal-c[idx]+d[idx], 0)), end=e_next)
    b.check_step(float(q), float(load), float(pv), float(e), action, CFG)
    return action


price_global = np.zeros(N)


def run(data, start_day=31, end_day=365, scenario_count=7, grid_size=193,
        update_hours=(6, 12, 18), time_limit=60.0):
    global price_global
    price_global = data["fixed_price"].copy()
    pred_l, _, _ = en.forecast_archive(data["load"], data["pv"])
    records, daily, checks, versions = [], [], [], []
    # The latest report uses the second-question January warmup boundary.
    e = float(EMIN)
    for day in range(start_day, end_day):
        date = pd.Timestamp(data["dates"][day]); day_start = e
        load_hat = pred_l[day, 0]
        q0 = None; q = None; values = None; grid = None
        previous_q = None; version = 0; current_start = 0
        day_versions = []; version_changes = []
        for t in range(N):
            if t in BOUNDARIES:
                hour = t//6
                pv_hat = pv_forecast_from_issue(data, day, hour)
                scenarios, weights, hist = paired_paths(data, pred_l, day, load_hat, pv_hat, hour, scenario_count)
                terminal = (day == end_day-1)
                old_q_suffix = None if previous_q is None else previous_q[t:]
                suffix_q, meta = contract_milp(price_global[t:], load_hat[t:], pv_hat[t:],
                                                [(L[t:], V[t:]) for L, V in scenarios],
                                                weights, e, old_q_suffix,
                                                terminal_target=EMIN if terminal else None,
                                                time_limit=time_limit)
                if previous_q is None:
                    q0 = np.zeros(N); q0[t:] = suffix_q
                    q = q0.copy()
                else:
                    old_future = q[t:].copy()
                    q[t:] = suffix_q
                    inc = np.maximum(q[t:] - old_future, 0.0)
                    dec = np.maximum(old_future - q[t:], 0.0)
                    version_changes.append(dict(update_hour=hour, start=t,
                                                increase=inc.copy(), decrease=dec.copy()))
                    for local_t, (old_amount, new_amount, up, down) in enumerate(
                            zip(old_future, q[t:], inc, dec)):
                        if up > 1e-9 or down > 1e-9:
                            delivery = t + local_t
                            versions.append(dict(date=str(date.date()), record_type="transaction",
                                update_hour=hour, interval=delivery,
                                previous_kwh=float(old_amount), new_kwh=float(new_amount),
                                increase_kwh=float(up), decrease_kwh=float(down),
                                price_yuan_per_kwh=float(price_global[delivery]),
                                increase_fee_yuan=float(1.5*price_global[delivery]*up),
                                refund_yuan=float(-price_global[delivery]*down),
                                penalty_yuan=float(.5*price_global[delivery]*down)))
                previous_q = q.copy()
                current_start = t
                grid, values = value_table(q[t:], price_global[t:],
                                           [(L[t:], V[t:]) for L, V in scenarios], weights,
                                           terminal=terminal, grid_size=grid_size)
                day_versions.append(dict(update_hour=hour, version=version,
                                         history_days=hist, weights=weights.tolist(),
                                         status=meta["status"], objective=meta["objective"],
                                         gap=meta["gap"], fallback=meta.get("fallback", False),
                                         q_change_kwh=float(np.abs(q[t:]-q0[t:]).sum())))
                versions.append(dict(date=str(date.date()), record_type="solve", **day_versions[-1]))
                version += 1
            action = dispatch(float(q[t]), float(data["load"][day, t]),
                              float(data["pv"][day, t]), e, grid,
                              values[t-current_start+1], float(price_global[t]),
                              terminal=(day == end_day-1), remaining=N-t)
            row = dict(date=str(date.date()), interval=t, timestamp=date+pd.Timedelta(minutes=10*t),
                       contract_version=version-1, purchase_kwh=float(q0[t]),
                       adjusted_purchase_kwh=float(q[t]), charge_kwh=action["c"],
                       discharge_kwh=action["d"], emergency_kwh=action["r"],
                       unused_kwh=action["s"], soc_start_kwh=float(e), soc_end_kwh=action["end"],
                       price_yuan_per_kwh=float(price_global[t]), load_kwh=float(data["load"][day,t]),
                       pv_kwh=float(data["pv"][day,t]))
            records.append(row); e = action["end"]
        frame = pd.DataFrame([r for r in records if r["date"] == str(date.date())])
        bill, detail = compute_day_bill(frame, q0, version_changes, price_global)
        if day >= start_day:
            daily.append(dict(date=str(date.date()), **bill,
                              start_soc_kwh=day_start, end_soc_kwh=e,
                              updates=json.dumps(day_versions, ensure_ascii=False)))
            checks.append(check_frame(frame))
    return pd.DataFrame(daily), pd.DataFrame(checks), pd.DataFrame(records), pd.DataFrame(versions)


def compute_day_bill(frame, q0, version_changes, price):
    q = frame.adjusted_purchase_kwh.to_numpy(); q0 = np.asarray(q0, float)
    base_fee = float(np.dot(price, q0))
    increase_fee = refund = penalty = 0.0
    for change in version_changes:
        # The arrays start at the corresponding publication boundary.
        start = int(change.get("start", 0))
        p = price[start:start+len(change["increase"])]
        increase_fee += float(np.dot(1.5*p, change["increase"]))
        refund += float(np.dot(-p, change["decrease"]))
        penalty += float(np.dot(.5*p, change["decrease"]))
    plan = base_fee + increase_fee + refund + penalty
    emergency = float(np.dot(5*price, frame.emergency_kwh.to_numpy()))
    final_contract_fee = float(np.dot(price, q))
    variation_fee = float(sum(np.dot(price[start:start+len(c["increase"])],
                                      c["increase"] + c["decrease"])
                              for c in version_changes
                              for start in [int(c["start"])]))
    identity_total = final_contract_fee + 0.5*variation_fee + emergency
    return dict(plan_fee_yuan=plan, emergency_fee_yuan=emergency,
                emergency_kwh=float(frame.emergency_kwh.sum()),
                adjustment_abs_kwh=float(sum(np.sum(c["increase"]+c["decrease"]) for c in version_changes)),
                increase_fee_yuan=increase_fee, refund_yuan=refund,
                decrease_penalty_yuan=penalty, base_plan_fee_yuan=base_fee,
                final_contract_fee_yuan=final_contract_fee,
                bill_identity_error_yuan=float(plan + emergency - identity_total),
                total_fee_yuan=plan+emergency), dict()


def check_frame(frame):
    balance = frame.adjusted_purchase_kwh + frame.pv_kwh + frame.emergency_kwh + frame.discharge_kwh - frame.load_kwh - frame.charge_kwh - frame.unused_kwh
    transition = np.diff(np.r_[frame.soc_start_kwh.to_numpy(), frame.soc_end_kwh.iloc[-1]]) - ETA*frame.charge_kwh + frame.discharge_kwh/ETA
    return dict(date=frame.date.iloc[0], balance_max_abs_kwh=float(np.abs(balance).max()),
                soc_transition_max_abs_kwh=float(np.abs(transition).max()),
                min_soc_kwh=float(min(frame.soc_start_kwh.min(), frame.soc_end_kwh.min())),
                max_soc_kwh=float(max(frame.soc_start_kwh.max(), frame.soc_end_kwh.max())),
                simultaneous_intervals=int(((frame.charge_kwh>1e-7)&(frame.discharge_kwh>1e-7)).sum()),
                emergency_while_charging=int(((frame.charge_kwh>1e-7)&(frame.emergency_kwh>1e-7)).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start-day", type=int, default=31)
    ap.add_argument("--end-day", type=int, default=365); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scenario-count", type=int, default=7); ap.add_argument("--grid-size", type=int, default=193)
    ap.add_argument("--time-limit", type=float, default=10.0)
    args = ap.parse_args()
    data = load_data()
    daily, checks, ledger, versions = run(data, args.start_day, args.end_day, args.scenario_count, args.grid_size,
                                          time_limit=args.time_limit)
    args.output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output/"daily.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(args.output/"checks.csv", index=False, encoding="utf-8-sig")
    ledger.to_csv(args.output/"ledger.csv", index=False, encoding="utf-8-sig")
    versions.to_csv(args.output/"contract_versions.csv", index=False, encoding="utf-8-sig")
    summary = dict(name="q3_latest_rolling", days=len(daily), intervals=len(ledger),
                   total_fee_yuan=float(daily.total_fee_yuan.sum()),
                   plan_fee_yuan=float(daily.plan_fee_yuan.sum()),
                   emergency_fee_yuan=float(daily.emergency_fee_yuan.sum()),
                   emergency_kwh=float(daily.emergency_kwh.sum()),
                   start_soc_kwh=float(daily.start_soc_kwh.iloc[0]), end_soc_kwh=float(daily.end_soc_kwh.iloc[-1]),
                   planner_fallback_updates=int(versions.loc[versions.record_type == "solve", "fallback"].astype(str).str.lower().eq("true").sum()) if not versions.empty and "fallback" in versions else 0,
                   max_bill_identity_error_yuan=float(np.abs(daily.bill_identity_error_yuan).max()),
                   checks={c:float(checks[c].max()) for c in ["balance_max_abs_kwh","soc_transition_max_abs_kwh","simultaneous_intervals","emergency_while_charging"]})
    (args.output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
