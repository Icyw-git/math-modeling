"""Independently verify saved enhanced-model ledgers and forecast-only plans."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import solve_c_q2_baseline as b
from solve_c_q2_enhanced import ROOT, CFG


def verify(folder,repeat=None):
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    for path,expected in config['input_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    original=json.loads((ROOT/'results'/'attachment_audit'/'audit_summary.json').read_text(encoding='utf-8'))
    for item in original['input_inventory']:
        p=ROOT/'CUMCM2026Problems'/'C题'/'附件'/item['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    records={}
    for name in config['variants']:
        f=pd.read_csv(folder/name/'ledger.csv',parse_dates=['timestamp'])
        daily=pd.read_csv(folder/name/'daily.csv')
        plans=pd.read_csv(folder/name/'planning.csv',parse_dates=['target_time'])
        s=json.loads((folder/name/'summary.json').read_text(encoding='utf-8'))
        checks=b.validate_ledger(f,CFG)
        assert len(f)==48096 and len(daily)==334
        assert abs(f.soc_start_kwh.iloc[0]-1200)<1e-6 and abs(f.soc_end_kwh.iloc[-1]-1200)<1e-6
        np.testing.assert_allclose(f.plan_cost_yuan,f.price_yuan_per_kwh*f.purchase_kwh,atol=1e-8,rtol=0)
        np.testing.assert_allclose(f.emergency_cost_yuan,5*f.price_yuan_per_kwh*f.emergency_kwh,atol=1e-8,rtol=0)
        for k in ('plan_cost_yuan','emergency_cost_yuan','purchase_kwh','emergency_kwh','unused_kwh'):
            np.testing.assert_allclose(f.groupby('date')[k].sum(),daily[k],atol=1e-7,rtol=0)
            assert abs(f[k].sum()-s[k])<1e-6
        assert abs(daily.total_cost_yuan.sum()-s['total_cost_yuan'])<1e-6
        expected=b.emergency_events(f,CFG)
        saved=pd.read_csv(folder/name/'emergency_events.csv')
        pd.testing.assert_frame_equal(saved,expected,atol=1e-8,rtol=0)
        balance=plans.planned_purchase_kwh+plans.forecast_pv_kwh+plans.planned_discharge_kwh-plans.forecast_load_kwh-plans.safety_margin_kwh-plans.planned_charge_kwh-plans.planned_unused_kwh
        transition=plans.planned_soc_end_kwh-plans.planned_soc_start_kwh-.9*plans.planned_charge_kwh+plans.planned_discharge_kwh/.9
        assert abs(balance).max()<1e-6 and abs(transition).max()<1e-6
        committed=plans[~plans.virtual_next_day]
        np.testing.assert_array_equal(committed.target_time.to_numpy(),f.timestamp.to_numpy())
        np.testing.assert_allclose(committed.planned_purchase_kwh,f.purchase_kwh,atol=1e-8,rtol=0)
        np.testing.assert_allclose(plans.groupby('issue_date').planned_soc_start_kwh.first(),daily.start_soc_kwh,atol=1e-6,rtol=0)
        repeated=False
        if repeat and (repeat/name/'ledger.csv').exists():
            other=pd.read_csv(repeat/name/'ledger.csv',parse_dates=['timestamp'])
            pd.testing.assert_frame_equal(f,other,atol=1e-6,rtol=0)
            repeated=True
        records[name]=dict(passed=True,checks=checks,rows=len(f),same_initial_final=True,
                           cost_recomputed=True,virtual_plan_not_executed_early=True,repeated_run_compared=repeated)
    result=dict(original_attachments_and_baseline_unchanged=True,variants=records)
    (folder/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=ROOT/'results'/'q2_enhanced')
    p.add_argument('--repeat',type=Path); a=p.parse_args(); verify(a.output,a.repeat)
