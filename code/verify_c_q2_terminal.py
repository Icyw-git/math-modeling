"""Independent ledger arithmetic and fixed-purchase verification."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import solve_c_q2_baseline as b

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('output',type=Path); args=ap.parse_args(); out=args.output
    config=json.loads((out/'config.json').read_text(encoding='utf8'))
    for path,digest in config['hashes'].items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    f=pd.read_csv(out/'ledger.csv',parse_dates=['timestamp']); d=pd.read_csv(out/'daily.csv'); summary=json.loads((out/'summary.json').read_text())
    ref=pd.read_csv(b.ROOT/'results/q2_enhanced_grid193/enhanced/ledger.csv').iloc[:len(f)]
    assert len(f)==144*(config['end_day']-31) and len(d)==config['end_day']-31
    checks=b.validate_ledger(f,b.Config())
    np.testing.assert_allclose(f.purchase_kwh,ref.purchase_kwh,atol=1e-9,rtol=0)
    np.testing.assert_allclose(f.plan_cost_yuan,f.purchase_kwh*f.price_yuan_per_kwh,atol=1e-6,rtol=0)
    np.testing.assert_allclose(f.emergency_cost_yuan,5*f.emergency_kwh*f.price_yuan_per_kwh,atol=1e-6,rtol=0)
    loss=.1*f.charge_kwh+(1/.9-1)*f.discharge_kwh
    np.testing.assert_allclose(loss,f.conversion_loss_kwh,atol=1e-6,rtol=0)
    np.testing.assert_allclose(f.purchase_kwh+f.emergency_kwh+f.pv_kwh-f.load_kwh-f.soc_end_kwh+f.soc_start_kwh,f.unused_kwh+loss,atol=1e-6,rtol=0)
    for key in ['purchase_kwh','charge_kwh','discharge_kwh','emergency_kwh','unused_kwh','plan_cost_yuan','emergency_cost_yuan','conversion_loss_kwh']:
        np.testing.assert_allclose(f.groupby('date')[key].sum(),d[key],atol=1e-6,rtol=0)
        if key in summary: assert abs(float(f[key].sum())-summary[key])<1e-6
    assert abs(summary['total_cost_yuan']-float((f.plan_cost_yuan+f.emergency_cost_yuan).sum()))<1e-6
    assert abs(f.soc_start_kwh.iloc[0]-1200)<1e-6
    if config['end_day']==365: assert abs(f.soc_end_kwh.iloc[-1]-1200)<1e-6
    result=dict(passed=True,intervals=len(f),checks=checks,hashes_preserved=True,fixed_purchase=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print(result)

if __name__=='__main__': main()
