"""Independent archive checks for information-node plans and realized contracts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from verify_c_q3_refined import verify


def verify_tree(out):
    out=Path(out); verify(out)
    cfg=json.loads((out/'config.json').read_text()); ledger=pd.read_csv(out/'ledger.csv'); tx=pd.read_csv(out/'transactions.csv')
    daily=pd.read_csv(out/'daily.csv').set_index('date'); summary=json.loads((out/'summary.json').read_text())
    for date,frame in ledger.groupby('date'):
        expected={'emergency_kwh':frame.emergency_kwh.sum(),'unused_kwh':frame.unused_kwh.sum(),
                  'conversion_loss_kwh':(.1*frame.charge_kwh+(1/.9-1)*frame.discharge_kwh).sum(),
                  'emergency_fee_yuan':(5*frame.price_yuan_per_kwh*frame.emergency_kwh).sum(),
                  'start_soc_kwh':frame.soc_start_kwh.iloc[0],'end_soc_kwh':frame.soc_end_kwh.iloc[-1]}
        for field,value in expected.items(): assert abs(daily.loc[date,field]-value)<1e-6,(date,field)
    for field in ['emergency_kwh','unused_kwh','conversion_loss_kwh','emergency_fee_yuan','plan_fee_yuan','total_fee_yuan','adjustment_abs_kwh']:
        assert abs(daily[field].sum()-summary[field])<1e-6,field
    count=0; failures=0; maximum=0.; leaves=0
    with (out/'trees.jsonl').open(encoding='utf8') as stream:
        for line in stream:
            record=json.loads(line); count+=1; nodes=record['nodes']; sol=record['solution']; w=np.array(record['weights'])
            assert all(h<record['day'] for h in record['history'])
            assert abs(w.sum()-1)<1e-9
            assert nodes[0]['start']==record['hour']*6
            for node in nodes:
                assert abs(w[node['members']].sum()-node['mass'])<1e-9
                assert node['end']-node['start']==36
                if node['children']: assert sorted(sum([nodes[j]['members'] for j in node['children']],[]))==sorted(node['members'])
            leaves=max(leaves,sum(not n['children'] for n in nodes))
            if sol is None:
                assert record['meta']['fallback']; failures+=1; continue
            for node,plan in zip(nodes,sol):
                e=np.array(plan['e']); c=np.array(plan['c']); d=np.array(plan['d'])
                error=float(abs(np.diff(e)-.9*c+d/.9).max()); maximum=max(maximum,error); assert error<1e-6
                assert min(e)>=1200-1e-6 and max(e)<=10800+1e-6
                assert min(c)>=-1e-6 and min(d)>=-1e-6 and max(max(c),max(d))<=5000/6+1e-6
                assert not ((c>1e-6)&(d>1e-6)).any()
                if node['parent'] is not None:
                    p=node['parent']; assert abs(e[0]-sol[p]['e'][-1])<1e-6
                    if not cfg['allow_future_adjustments']: np.testing.assert_allclose(plan['q'],sol[p]['q'][node['start']-nodes[p]['start']:],rtol=0,atol=1e-6)
                if record['day']==364 and not node['children']: assert abs(e[-1]-1200)<1e-6
            date=str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=record['day'])).date()); hour=record['hour']
            actual=ledger[ledger.date==date].purchase_kwh.to_numpy() if hour==0 else tx[(tx.date==date)&(tx.update_hour==hour)].sort_values('interval').new_kwh.to_numpy()
            np.testing.assert_allclose(sol[0]['q'],actual,rtol=0,atol=1e-6)
    assert count==(cfg['end_day']-31)*4
    assert failures==summary['fallbacks']
    result=dict(passed=True,windows=count,fallbacks=failures,max_planned_soc_error=maximum,max_leaves=leaves,history_strictly_past=True,node_contracts_shared=True,actual_root_versions_match=True,daily_annual_energy_and_cost_totals_match=True)
    (out/'tree_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print(result)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('output'); verify_tree(ap.parse_args().output)
