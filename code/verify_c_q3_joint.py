"""Rebuild invoices and audit joint-search and terminal-value archives."""
import argparse
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import q3_joint_policy as joint
from verify_c_q3_refined import verify


def check(out):
    out=Path(out); verify(out)
    ledger=pd.read_csv(out/'ledger.csv'); daily=pd.read_csv(out/'daily.csv').set_index('date'); tx=pd.read_csv(out/'transactions.csv'); summary=json.loads((out/'summary.json').read_text())
    for date,g in ledger.groupby('date'):
        metrics=dict(emergency_kwh=g.emergency_kwh.sum(),unused_kwh=g.unused_kwh.sum(),conversion_loss_kwh=(.1*g.charge_kwh+(1/.9-1)*g.discharge_kwh).sum(),emergency_fee_yuan=(5*g.price_yuan_per_kwh*g.emergency_kwh).sum())
        for k,v in metrics.items(): assert abs(daily.loc[date,k]-v)<1e-6
    for k in ['plan_fee_yuan','total_fee_yuan','emergency_fee_yuan','emergency_kwh','unused_kwh','conversion_loss_kwh','adjustment_abs_kwh']: assert abs(daily[k].sum()-summary[k])<1e-6
    windows=0
    for line in (out/'searches.jsonl').read_text().splitlines():
        r=json.loads(line); windows+=1; day=(pd.Timestamp(r['date'])-pd.Timestamp('2025-01-01')).days
        assert all(h<day for h in r['history']); assert abs(r['objective']-min(c['value'] for c in r['search']))<1e-6
        for node,q in zip(r['nodes'],r['contracts']): assert len(q)==144-node['start'] and np.isfinite(q).all() and min(q)>=0
        actual=ledger[ledger.date==r['date']].purchase_kwh if r['hour']==0 else tx[(tx.date==r['date'])&(tx.update_hour==r['hour'])].sort_values('interval').new_kwh
        np.testing.assert_allclose(r['contracts'][0],actual,rtol=0,atol=1e-6)
    terminal_fallbacks=0; curves=0
    for line in (out/'terminal_curves.jsonl').read_text().splitlines():
        r=json.loads(line); curves+=1; day=(pd.Timestamp(r['date'])-pd.Timestamp('2025-01-01')).days
        assert np.isfinite(r['values']).all() and len(r['values'])==9
        assert all(h<day for h in r.get('history',[])); terminal_fallbacks+=sum(s['fallback'] for s in r['solves'])
    assert windows==len(daily)*4 and curves==len(daily)
    result=dict(passed=True,windows=windows,terminal_curves=curves,terminal_solver_fallbacks=terminal_fallbacks,all_selected_candidates_minimum=True,history_strictly_past=True,energy_totals_match=True)
    (out/'joint_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print(result)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('output',nargs='?'); ap.add_argument('--tests',action='store_true'); a=ap.parse_args()
    if a.tests:
        suite=unittest.defaultTestLoader.discover(str(joint.ref.ROOT/'code'),pattern='test_c_q3*.py'); result=unittest.TextTestRunner(verbosity=2).run(suite)
        (joint.ref.ROOT/'results/q3_joint/tests.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures)),indent=2),encoding='utf8')
        if not result.wasSuccessful(): raise SystemExit(1)
    if a.output: check(a.output)
