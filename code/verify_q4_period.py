"""Independent verification for arbitrary causal validation periods."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_c_attachments import matrix

ROOT=Path(__file__).resolve().parents[1]


def tail_mean(x,alpha=.9):
    # Equal-weight observed days, fractional final observation included.
    a=np.sort(np.asarray(x,float))[::-1]; mass=len(a)*(1-alpha); full=int(np.floor(mass))
    return float((a[:full].sum()+(mass-full)*a[full])/mass)


def verify(out,prefix=None):
    out=Path(out); cfg=json.loads((out/'config.json').read_text(encoding='utf8'))
    for p,digest in {**cfg['input_hashes'],**cfg['source_hashes']}.items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest, p
    source=ROOT/'CUMCM2026Problems'/'C题'/'附件'
    dates,load=matrix(source/'附件2.xlsx','小区负载'); _,pv=matrix(source/'附件2.xlsx','光伏发电实际功率'); _,price=matrix(source/'附件4.xlsx')
    files=sorted(out.glob('day_*.json')); start_day=cfg['period_start_day']; end_day=cfg['period_end_day']; assert len(files)==end_day-start_day
    state=float(cfg['initial_soc']); fee=0.; days=[]; max_error=0.; max_prefix=0.; fallback=0; windows=0; prefix_days=0
    for offset,path in enumerate(files):
        obj=json.loads(path.read_text(encoding='utf8')); day=start_day+offset
        assert obj['day']==day and obj['daily']['date']==str(dates.iloc[day].date())
        d=pd.DataFrame(obj['ledger']); assert d.interval.tolist()==list(range(144))
        np.testing.assert_allclose(d.load_kwh,load[day]/6,rtol=0,atol=1e-9); np.testing.assert_allclose(d.pv_kwh,pv[day]/6,rtol=0,atol=1e-9); np.testing.assert_array_equal(d.price,price[day])
        assert abs(d.soc_start_kwh.iloc[0]-state)<1e-6
        assert np.isfinite(d.to_numpy()).all()
        q=d.q0.to_numpy().copy(); units=q.copy(); variation=np.zeros(144)
        for change in obj['changes']:
            start=change['start']; assert cfg['mode']=='q3' and start in (36,72,108)
            prev=np.asarray(change['previous']); new=np.asarray(change['new']); up=np.asarray(change['up']); down=np.asarray(change['down'])
            np.testing.assert_allclose(q[start:],prev,atol=1e-6,rtol=0)
            np.testing.assert_allclose(up,np.maximum(new-prev,0),atol=1e-6,rtol=0); np.testing.assert_allclose(down,np.maximum(prev-new,0),atol=1e-6,rtol=0)
            q[start:]=new; units[start:]+=1.5*up-.5*down; variation[start:]+=up+down
        np.testing.assert_allclose(q,d.q,atol=1e-6,rtol=0); np.testing.assert_allclose(units,d.charged_units,atol=1e-6,rtol=0)
        if cfg['mode']=='q2': assert not obj['changes']; np.testing.assert_array_equal(d.q0,d.q)
        balance=d.q+d.pv_kwh+d.discharge_kwh+d.emergency_kwh-d.load_kwh-d.charge_kwh-d.unused_kwh
        transition=d.soc_end_kwh-d.soc_start_kwh-.9*d.charge_kwh+d.discharge_kwh/.9
        error=max(abs(balance).max(),abs(transition).max()); max_error=max(max_error,float(error)); assert error<1e-6
        assert d[['q0','q','charge_kwh','discharge_kwh','emergency_kwh','unused_kwh']].min().min()>=-1e-6
        assert d[['charge_kwh','discharge_kwh']].max().max()<=5000/6+1e-6
        assert d[['soc_start_kwh','soc_end_kwh']].min().min()>=1200-1e-6 and d[['soc_start_kwh','soc_end_kwh']].max().max()<=10800+1e-6
        assert not ((d.charge_kwh>1e-6)&((d.discharge_kwh>1e-6)|(d.emergency_kwh>1e-6))).any()
        assert not ((d.emergency_kwh>1e-6)&(d.unused_kwh>1e-6)).any()
        np.testing.assert_allclose(d.soc_start_kwh.iloc[1:],d.soc_end_kwh.iloc[:-1],atol=1e-6,rtol=0)
        actual=float(np.dot(price[day],units+5*d.emergency_kwh)); alternate=float(np.dot(price[day],q+.5*variation+5*d.emergency_kwh))
        assert abs(actual-alternate)<1e-6 and abs(actual-obj['daily']['total_fee'])<1e-6
        for window in obj['windows']:
            assert window['history']==list(range(max(1,day-28),day)); assert window['price_scenario_min']>0
            assert window['scenario_count']==len(window['history'])
            if not window['fallback'] and window['status']!='contract_frozen':
                assert abs(tail_mean(window['scenario_costs'])-window['scenario_invoice_cvar90'])<1e-6
            fallback+=int(window['fallback']); windows+=1
        if prefix:
            pp=Path(prefix)/path.name
            if pp.exists():
                prefix_days+=1
                before=pd.DataFrame(json.loads(pp.read_text(encoding='utf8'))['ledger'])
                max_prefix=max(max_prefix,float(np.max(abs(before.to_numpy()-d.to_numpy()))))
        fee+=actual; days.append(actual); state=float(d.soc_end_kwh.iloc[-1])
    assert files
    if end_day==365: assert abs(state-cfg['annual_terminal_soc'])<1e-6
    result=dict(days=len(files),intervals=144*len(files),cost=fee,realized_daily_cvar90=tail_mean(days),max_physical_error=max_error,final_soc=state,windows=windows,fallbacks=fallback,prefix_days=prefix_days,max_prefix_difference=max_prefix if prefix_days else None,prefix_match=max_prefix<1e-6 if prefix_days else None,passed=True)
    sp=out/'summary.json'
    if sp.exists():
        summary=json.loads(sp.read_text()); assert abs(fee-summary['total_fee'])<1e-5; assert abs(result['realized_daily_cvar90']-summary['realized_daily_cvar90'])<1e-6
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(out.name,result,flush=True); return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); p.add_argument('--prefix',type=Path); a=p.parse_args(); verify(a.output,a.prefix)


