"""Independent saved-output checks; optionally compare a complete repeated run."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from solve_c_q2_baseline import ROOT, Config, validate_ledger, emergency_events


def verify(folder, repeat=None):
    cfg = Config()
    ledger = pd.read_csv(folder/'ledger.csv', parse_dates=['timestamp'])
    daily = pd.read_csv(folder/'daily.csv')
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    checks = validate_ledger(ledger, cfg)
    assert len(ledger) == 52560 and len(daily) == 365
    assert ledger.evaluation.sum() == 48096 and daily.evaluation.sum() == 334
    np.testing.assert_allclose(ledger.plan_cost_yuan, ledger.purchase_kwh*ledger.price_yuan_per_kwh, atol=1e-8, rtol=0)
    np.testing.assert_allclose(ledger.emergency_cost_yuan,
        5*ledger.emergency_kwh*ledger.price_yuan_per_kwh, atol=1e-8, rtol=0)
    np.testing.assert_allclose(daily.total_cost_yuan, daily.plan_cost_yuan+daily.emergency_cost_yuan, atol=1e-8, rtol=0)
    group = ledger.groupby('date', sort=True)
    np.testing.assert_allclose(group.plan_cost_yuan.sum(), daily.plan_cost_yuan, atol=1e-7, rtol=0)
    np.testing.assert_allclose(group.emergency_cost_yuan.sum(), daily.emergency_cost_yuan, atol=1e-7, rtol=0)
    evaluated = ledger[ledger.evaluation]
    assert abs(daily.loc[daily.evaluation,'total_cost_yuan'].sum()-summary['total_cost_yuan'])<1e-6
    actual_events = emergency_events(evaluated, cfg)
    saved_events = pd.read_csv(folder/'emergency_events.csv')
    pd.testing.assert_frame_equal(saved_events, actual_events, check_exact=False, atol=1e-8, rtol=0)
    for _, day in group:
        balance = day.purchase_kwh+day.predicted_pv_kwh+day.planned_discharge_kwh-day.predicted_load_kwh-day.planned_charge_kwh-day.planned_unused_kwh
        assert abs(balance).max()<1e-6
        shift = day.reference_soc_end_kwh-day.reference_soc_start_kwh-.9*day.planned_charge_kwh+day.planned_discharge_kwh/.9
        assert abs(shift).max()<1e-6
        assert abs(day.reference_soc_start_kwh.iloc[0]-day.soc_start_kwh.iloc[0])<1e-6
    for name, expected in config['input_sha256'].items():
        source = ROOT/'CUMCM2026Problems'/'C题'/'附件'/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    audit = json.loads((ROOT/'results'/'attachment_audit'/'audit_summary.json').read_text(encoding='utf-8'))
    for item in audit['input_inventory']:
        source = ROOT/'CUMCM2026Problems'/'C题'/'附件'/item['file']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256']
    result = dict(passed=True, all_nine_original_workbooks_unchanged=True,
        full_year_intervals=len(ledger), formal_intervals=len(evaluated),
        saved_costs_recomputed=True, saved_plans_balance_verified=True, checks=checks,
        repeated_full_run_compared=False)
    if repeat:
        other = pd.read_csv(repeat/'ledger.csv', parse_dates=['timestamp'])
        pd.testing.assert_frame_equal(ledger, other, check_exact=False, atol=1e-6, rtol=0)
        other_days = pd.read_csv(repeat/'daily.csv')
        pd.testing.assert_frame_equal(daily.drop(columns='solve_seconds'), other_days.drop(columns='solve_seconds'),
                                      check_exact=False, atol=1e-6, rtol=0)
        result['repeated_full_run_compared'] = True
        result['repeat_tolerance'] = 1e-6
    (folder/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'q2_baseline')
    parser.add_argument('--repeat',type=Path)
    args=parser.parse_args()
    verify(args.output,args.repeat)
