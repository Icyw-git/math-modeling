"""Reconstruct all contract versions and invoices independently."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

def verify(out):
    out=Path(out); f=pd.read_csv(out/'ledger.csv'); d=pd.read_csv(out/'daily.csv'); tx=pd.read_csv(out/'transactions.csv'); cfg=json.loads((out/'config.json').read_text()); summary=json.loads((out/'summary.json').read_text())
    for file,digest in cfg['hashes'].items(): assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
    assert len(f)==(cfg['end_day']-31)*144 and len(d)==cfg['end_day']-31
    fields=['purchase_kwh','adjusted_purchase_kwh','load_kwh','pv_kwh','charge_kwh','discharge_kwh','emergency_kwh','unused_kwh']
    assert np.isfinite(f[fields].to_numpy()).all() and f[fields].min().min()>=-1e-6
    assert f.soc_start_kwh.min()>=1200-1e-6 and f.soc_end_kwh.min()>=1200-1e-6 and max(f.soc_start_kwh.max(),f.soc_end_kwh.max())<=10800+1e-6
    assert max(f.charge_kwh.max(),f.discharge_kwh.max())<=5000/6+1e-6
    assert not ((f.charge_kwh>1e-6)&((f.discharge_kwh>1e-6)|(f.emergency_kwh>1e-6))).any()
    bal=f.adjusted_purchase_kwh+f.pv_kwh+f.emergency_kwh+f.discharge_kwh-f.load_kwh-f.charge_kwh-f.unused_kwh
    soc=f.soc_end_kwh-f.soc_start_kwh-.9*f.charge_kwh+f.discharge_kwh/.9
    assert abs(bal).max()<1e-6 and abs(soc).max()<1e-6
    np.testing.assert_allclose(f.soc_end_kwh.iloc[:-1],f.soc_start_kwh.iloc[1:],atol=1e-6,rtol=0)
    assert abs(f.soc_start_kwh.iloc[0]-1200)<1e-6
    if cfg['end_day']==365: assert abs(f.soc_end_kwh.iloc[-1]-1200)<1e-6
    for day,g in f.groupby('date',sort=True):
        assert np.array_equal(g.interval,np.arange(144)); q=g.purchase_kwh.to_numpy().copy(); p=g.price_yuan_per_kwh.to_numpy(); invoice=float(q@p); variation=0.
        for hour in cfg['update_hours']:
            changes=tx[(tx.date==day)&(tx.update_hour==hour)].sort_values('interval')
            if changes.empty: continue
            ids=changes.interval.to_numpy(int); assert (ids>=hour*6).all() and len(np.unique(ids))==len(ids)
            np.testing.assert_allclose(q[ids],changes.previous_kwh,atol=1e-6,rtol=0)
            up=np.maximum(changes.new_kwh.to_numpy()-q[ids],0); down=np.maximum(q[ids]-changes.new_kwh.to_numpy(),0)
            np.testing.assert_allclose(up,changes.increase_kwh,atol=1e-6,rtol=0); np.testing.assert_allclose(down,changes.decrease_kwh,atol=1e-6,rtol=0)
            invoice+=float(p[ids]@(1.5*up-.5*down)); variation+=float(p[ids]@(up+down)); q[ids]=changes.new_kwh
        np.testing.assert_allclose(q,g.adjusted_purchase_kwh,atol=1e-6,rtol=0)
        row=d[d.date==day].iloc[0]; emergency=float((5*p*g.emergency_kwh).sum())
        assert abs(invoice-row.plan_fee_yuan)<1e-6 and abs(invoice+emergency-row.total_fee_yuan)<1e-6
        assert abs(invoice-float(p@q)-.5*variation)<1e-6
    assert abs(d.total_fee_yuan.sum()-summary['total_fee_yuan'])<1e-6
    result=dict(passed=True,intervals=len(f),max_balance=float(abs(bal).max()),max_soc=float(abs(soc).max()),contract_versions_rebuilt=True,source_hashes_preserved=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print(result)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('output'); verify(ap.parse_args().output)
