"""Independent physical, accounting, attribution and provenance verification."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import solve_c_q2_baseline as b
from solve_c_q2_low_waste import ROOT, CFG


def verify(folder,repeat=None):
    manifest=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    for relative,digest in manifest['input_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    audit=json.loads((ROOT/'results'/'attachment_audit'/'audit_summary.json').read_text(encoding='utf-8'))
    for item in audit['input_inventory']:
        path=ROOT/'CUMCM2026Problems'/'C题'/'附件'/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    f=pd.read_csv(folder/'ledger.csv',parse_dates=['timestamp']); daily=pd.read_csv(folder/'daily.csv')
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    checks=b.validate_ledger(f,CFG)
    assert len(f)==48096 and len(daily)==334
    assert abs(f.soc_start_kwh.iloc[0]-1200)<1e-6 and abs(f.soc_end_kwh.iloc[-1]-1200)<1e-6
    np.testing.assert_allclose(f.plan_cost_yuan,f.price_yuan_per_kwh*f.purchase_kwh,atol=1e-8,rtol=0)
    np.testing.assert_allclose(f.emergency_cost_yuan,5*f.price_yuan_per_kwh*f.emergency_kwh,atol=1e-8,rtol=0)
    np.testing.assert_allclose(f.conversion_loss_kwh,.1*f.charge_kwh+(1/.9-1)*f.discharge_kwh,atol=1e-8,rtol=0)
    np.testing.assert_allclose(f.total_energy_waste_kwh,f.unused_kwh+f.conversion_loss_kwh,atol=1e-8,rtol=0)
    identity=f.purchase_kwh+f.emergency_kwh+f.pv_kwh-f.load_kwh-(f.soc_end_kwh-f.soc_start_kwh)
    np.testing.assert_allclose(identity,f.total_energy_waste_kwh,atol=1e-7,rtol=0)
    for key in ['purchase_kwh','emergency_kwh','unused_kwh','conversion_loss_kwh','total_energy_waste_kwh','plan_cost_yuan','emergency_cost_yuan']:
        np.testing.assert_allclose(f.groupby('date')[key].sum(),daily[key],atol=1e-7,rtol=0)
        assert abs(float(f[key].sum())-summary[key])<1e-6
    assert abs(daily.total_cost_yuan.sum()-summary['total_cost_yuan'])<1e-6
    events=pd.read_csv(folder/'emergency_events.csv')
    pd.testing.assert_frame_equal(events,b.emergency_events(f,CFG),atol=1e-8,rtol=0)
    plans=pd.read_csv(folder/'planning.csv',parse_dates=['target_time'])
    current=plans[~plans.virtual_next_day]
    np.testing.assert_array_equal(current.target_time.to_numpy(),f.timestamp.to_numpy())
    np.testing.assert_allclose(current.planned_purchase_kwh,f.purchase_kwh,atol=1e-8,rtol=0)
    scen=pd.read_csv(folder/'scenario_recourse.csv')
    assert np.isfinite(scen.select_dtypes(include='number').to_numpy()).all()
    assert scen[['q','r','c','d','s']].min().min()>-1e-6
    assert scen[['c','d']].max().max()<=CFG.cap+1e-6
    assert scen[['e_start','e_end']].min().min()>=CFG.emin-1e-6
    assert scen[['e_start','e_end']].max().max()<=CFG.emax+1e-6
    assert abs(scen.q+scen.r+scen.d-scen.demand_kwh-scen.c-scen.s).max()<1e-6
    assert abs(scen.e_end-scen.e_start-.9*scen.c+scen.d/.9).max()<1e-6
    assert not ((scen.c>1e-6)&((scen.d>1e-6)|(scen.r>1e-6))).any()
    assert scen.groupby(['issue_date','slot']).q.agg(lambda a:a.max()-a.min()).max()<1e-6
    for _,part in scen.groupby(['issue_date','scenario']):
        np.testing.assert_allclose(part.e_start.to_numpy()[1:],part.e_end.to_numpy()[:-1],atol=1e-6,rtol=0)
    reference=ROOT/'results'/'q2_enhanced_grid193'/'enhanced'
    rf=pd.read_csv(reference/'ledger.csv'); rs=json.loads((reference/'summary.json').read_text(encoding='utf-8'))
    assert abs(summary['saving_yuan']-(rs['total_cost_yuan']-summary['total_cost_yuan']))<1e-6
    assert abs(summary['unused_reduction_kwh']-(rf.unused_kwh.sum()-f.unused_kwh.sum()))<1e-6
    accepted=summary['total_cost_yuan']<=rs['total_cost_yuan']+1e-6 and summary['unused_kwh']<rs['unused_kwh']
    assert accepted==summary['accepted_cost_and_unused']
    repeated=False
    if repeat:
        other=pd.read_csv(repeat/'ledger.csv',parse_dates=['timestamp'])
        subset=f[f.date.isin(other.date)].reset_index(drop=True)
        pd.testing.assert_frame_equal(subset,other,atol=1e-6,rtol=0)
        repeated=True
    result=dict(passed=True,checks=checks,rows=len(f),scenario_rows=len(scen),
        original_inputs_and_prior_model_unchanged=True,common_boundary_soc=True,
        scenario_shared_plan_checked=True,costs_recomputed=True,energy_loss_identity_checked=True,
        virtual_plan_not_executed_early=True,accepted_cost_and_unused=bool(accepted),
        prefix_repeat_checked=repeated)
    (folder/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=ROOT/'results'/'q2_low_waste')
    p.add_argument('--repeat',type=Path); a=p.parse_args(); verify(a.output,a.repeat)
