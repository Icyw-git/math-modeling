"""Independent file-level verification for the latest Problem 3 run."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "q3_latest_full_3scen"
PRICE = pd.read_excel(ROOT / "CUMCM2026Problems" / "C题" / "附件" / "附件1.xlsx").iloc[:, 1].to_numpy(float)
CAP = 5000 / 6
ETA = 0.9
EMIN, EMAX = 1200.0, 10800.0


def main(output=None, scenario_count=3, grid_size=49, time_limit=0.2):
    global OUT
    if output is not None:
        OUT = Path(output)
    daily = pd.read_csv(OUT / "daily.csv")
    checks = pd.read_csv(OUT / "checks.csv")
    ledger = pd.read_csv(OUT / "ledger.csv")
    versions = pd.read_csv(OUT / "contract_versions.csv")
    assert len(daily) == 334 and len(ledger) == 334 * 144
    assert daily.date.is_unique and ledger.groupby("date").size().eq(144).all()

    balance = (ledger.adjusted_purchase_kwh + ledger.pv_kwh + ledger.emergency_kwh
               + ledger.discharge_kwh - ledger.load_kwh - ledger.charge_kwh
               - ledger.unused_kwh)
    soc_transition = (np.diff(np.r_[ledger.soc_start_kwh.to_numpy(),
                                    ledger.soc_end_kwh.iloc[-1]])
                      - ETA * ledger.charge_kwh.to_numpy()
                      + ledger.discharge_kwh.to_numpy() / ETA)
    continuity = np.abs(daily.end_soc_kwh.iloc[:-1].to_numpy()
                        - daily.start_soc_kwh.iloc[1:].to_numpy())
    checks_out = dict(intervals=len(ledger), days=len(daily),
                      balance_max_abs_kwh=float(np.abs(balance).max()),
                      soc_transition_max_abs_kwh=float(np.abs(soc_transition).max()),
                      continuity_max_abs_kwh=float(continuity.max()),
                      min_soc_kwh=float(min(ledger.soc_start_kwh.min(), ledger.soc_end_kwh.min())),
                      max_soc_kwh=float(max(ledger.soc_start_kwh.max(), ledger.soc_end_kwh.max())),
                      max_charge_kw=float(ledger.charge_kwh.max() * 6),
                      max_discharge_kw=float(ledger.discharge_kwh.max() * 6),
                      simultaneous_intervals=int(((ledger.charge_kwh > 1e-7)
                                                   & (ledger.discharge_kwh > 1e-7)).sum()),
                      emergency_while_charging=int(((ledger.charge_kwh > 1e-7)
                                                    & (ledger.emergency_kwh > 1e-7)).sum()),
                      contract_negative_intervals=int((ledger.adjusted_purchase_kwh < -1e-7).sum()),
                      bill_identity_max_abs_yuan=float(np.abs(daily.bill_identity_error_yuan).max()))
    assert checks_out["balance_max_abs_kwh"] < 1e-6
    assert checks_out["soc_transition_max_abs_kwh"] < 1e-6
    assert checks_out["continuity_max_abs_kwh"] < 1e-6
    assert checks_out["min_soc_kwh"] >= EMIN - 1e-6
    assert checks_out["max_soc_kwh"] <= EMAX + 1e-6
    assert checks_out["max_charge_kw"] <= 5000 + 1e-6
    assert checks_out["max_discharge_kw"] <= 5000 + 1e-6
    assert checks_out["simultaneous_intervals"] == 0
    assert checks_out["emergency_while_charging"] == 0
    assert checks_out["contract_negative_intervals"] == 0
    assert checks_out["bill_identity_max_abs_yuan"] < 1e-6

    # Rebuild the realized invoice from the original plan and every version
    # transaction.  This catches accidentally charging only the final net
    # deviation instead of each successive contract change.
    base = ledger.groupby("date").apply(
        lambda g: float(np.dot(PRICE, g.sort_values("interval").purchase_kwh.to_numpy())),
        include_groups=False)
    tx = versions[versions.record_type == "transaction"].copy()
    if not tx.empty:
        assert (tx.interval >= tx.update_hour * 6).all()
        checks_out["transaction_boundary_violations"] = int((tx.interval < tx.update_hour * 6).sum())
    else:
        checks_out["transaction_boundary_violations"] = 0
    tx_cost = tx.groupby("date")[["increase_fee_yuan", "refund_yuan", "penalty_yuan"]].sum().sum(axis=1)
    rebuilt_plan = base.add(tx_cost, fill_value=0)
    plan_error = rebuilt_plan - daily.set_index("date").plan_fee_yuan
    checks_out["invoice_rebuild_max_abs_yuan"] = float(np.abs(plan_error).max())
    assert checks_out["invoice_rebuild_max_abs_yuan"] < 1e-6

    summary = dict(name="q3_latest_rolling", model_scope="causal rolling contract adjustment",
                   settlement="refund original decrease plus 50% penalty; increase at 150%",
                   scenario_count=int(scenario_count), control_grid_points=int(grid_size),
                   control_grid_kwh=200, solve_time_limit_seconds=float(time_limit),
                   days=len(daily), intervals=len(ledger),
                   total_fee_yuan=float(daily.total_fee_yuan.sum()),
                   plan_fee_yuan=float(daily.plan_fee_yuan.sum()),
                   emergency_fee_yuan=float(daily.emergency_fee_yuan.sum()),
                   emergency_kwh=float(daily.emergency_kwh.sum()),
                   adjustment_abs_kwh=float(daily.adjustment_abs_kwh.sum()),
                   increase_fee_yuan=float(daily.increase_fee_yuan.sum()),
                   refund_yuan=float(daily.refund_yuan.sum()),
                   decrease_penalty_yuan=float(daily.decrease_penalty_yuan.sum()),
                   start_soc_kwh=float(daily.start_soc_kwh.iloc[0]),
                   end_soc_kwh=float(daily.end_soc_kwh.iloc[-1]),
                   planner_fallback_updates=int(versions.loc[versions.record_type == "solve", "fallback"].astype(str).str.lower().eq("true").sum()),
                   solver_status_counts=versions.loc[versions.record_type == "solve", "status"].value_counts().to_dict(),
                   max_reported_solver_gap=float(pd.to_numeric(versions.loc[versions.record_type == "solve", "gap"], errors="coerce").max()),
                   checks=checks_out)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    baseline_path = ROOT / "results" / "q234_baseline" / "q3_daily.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        reference = dict(model="existing_q3_baseline", total_fee_yuan=float(baseline.total_fee_yuan.sum()),
                         plan_fee_yuan=float(baseline.adjustment_fee_yuan.sum()),
                         emergency_fee_yuan=float(baseline.emergency_fee_yuan.sum()),
                         emergency_kwh=float(baseline.emergency_kwh.sum()))
    else:
        # The checked-in workspace may contain only the exported templates;
        # retain a comparison to the prior local Problem-3 run when its
        # independent daily ledger is available.
        prior = pd.read_csv(ROOT / "results" / "q3_enhanced" / "q3_daily.csv")
        reference = dict(model="prior_local_q3_enhanced", total_fee_yuan=float(prior.total_fee_yuan.sum()),
                         plan_fee_yuan=float(prior.plan_fee_yuan.sum()),
                         emergency_fee_yuan=float(prior.emergency_fee_yuan.sum()),
                         emergency_kwh=float(prior.emergency_kwh.sum()))
    comparison = pd.DataFrame([reference,
        dict(model="q3_latest_rolling", total_fee_yuan=summary["total_fee_yuan"],
             plan_fee_yuan=summary["plan_fee_yuan"], emergency_fee_yuan=summary["emergency_fee_yuan"],
             emergency_kwh=summary["emergency_kwh"])])
    comparison["difference_vs_reference_yuan"] = comparison.total_fee_yuan.iloc[0] - comparison.total_fee_yuan
    comparison["difference_vs_reference_fraction"] = 1 - comparison.total_fee_yuan / comparison.total_fee_yuan.iloc[0]
    comparison.to_csv(OUT / "comparison.csv", index=False, encoding="utf-8-sig")
    (OUT / "verification.json").write_text(json.dumps(checks_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--scenario-count", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=49)
    parser.add_argument("--time-limit", type=float, default=0.2)
    args = parser.parse_args()
    main(args.output, args.scenario_count, args.grid_size, args.time_limit)
