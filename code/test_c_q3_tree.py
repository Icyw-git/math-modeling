"""Tree topology, causality, physical optimization and annual boundary checks."""
import copy
import unittest
import tempfile
import json
from unittest.mock import patch
from pathlib import Path
import numpy as np
import q3_scenario_tree as tr
import q3_tree_compact as compact


class TreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=tr.ref.old.load_data()
        cls.pred,_,_=tr.ref.en.forecast_archive(cls.data['load'],cls.data['pv'])

    def test_partition(self):
        tree=tr.build(self.data,self.pred,31,0)
        self.assertLessEqual(sum(not n['children'] for n in tree['nodes']),27)
        for n in tree['nodes']:
            if n['children']:
                children=[tree['nodes'][i] for i in n['children']]
                self.assertEqual(sorted(sum([c['members'] for c in children],[])),sorted(n['members']))
                self.assertAlmostEqual(sum(c['mass'] for c in children),n['mass'])
        self.assertTrue(all(h<31 for h in tree['history']))

    def test_identical_features(self):
        self.assertEqual(tr.cluster(np.ones((5,4)),np.ones(5)/5,list(range(5))),[list(range(5))])

    def test_no_future_information(self):
        for hour in (0,6,12,18):
            a=tr.build(self.data,self.pred,31,hour)
            altered=copy.deepcopy(self.data)
            altered['load'][31,hour*6:]+=12345
            altered['pv'][31,hour*6:]+=23456
            for key in altered['forecast']:
                if key[0]>self.data['dates'][31] or (key[0]==self.data['dates'][31] and key[1]>hour): altered['forecast'][key]=np.ones(24)*99999
            b=tr.build(altered,self.pred,31,hour)
            for field in ('loads','pvs','weights'): np.testing.assert_array_equal(a[field][:,hour*6:] if field!='weights' else a[field],b[field][:,hour*6:] if field!='weights' else b[field])
            self.assertEqual(a['nodes'],b['nodes'])

    def test_solver_and_frozen_contract(self):
        tree=tr.build(self.data,self.pred,31,12,branches=2)
        solution,meta=tr.optimize(tree,self.data['fixed_price'],6000,time_limit=30,allow_future=False)
        self.assertLessEqual(meta['row_error'],1e-6)
        for n in tree['nodes'][1:]:
            p=tree['nodes'][n['parent']]
            np.testing.assert_allclose(solution[n['id']]['q'],solution[p['id']]['q'][n['start']-p['start']:],atol=1e-6)
        grid,values=tr.control_value(tree,solution,self.data['fixed_price'])
        self.assertEqual(values.shape,(37,97)); self.assertTrue(np.isfinite(values).all())

    def test_hard_terminal(self):
        tree=tr.build(self.data,self.pred,364,18)
        solution,meta=tr.optimize(tree,self.data['fixed_price'],6000,terminal=True,time_limit=30)
        self.assertAlmostEqual(solution[0]['e'][-1],1200,places=6)

    def test_compact_future_information(self):
        for hour in (0,6,12,18):
            a=compact.build(self.data,self.pred,31,hour)
            changed=copy.deepcopy(self.data)
            changed['load'][31,hour*6:]+=12345; changed['pv'][31,hour*6:]+=23456
            for key in changed['forecast']:
                if key[0]>self.data['dates'][31] or (key[0]==self.data['dates'][31] and key[1]>hour): changed['forecast'][key]=np.ones(24)*99999
            b=compact.build(changed,self.pred,31,hour)
            for field in ('loads','pvs'): np.testing.assert_array_equal(a[field][:,hour*6:],b[field][:,hour*6:])
            np.testing.assert_array_equal(a['weights'],b['weights']); self.assertEqual(a['nodes'],b['nodes'])

    def test_failed_tree_falls_back_to_checked_planner(self):
        import solve_c_q3_tree as runner
        with tempfile.TemporaryDirectory(dir=tr.ref.ROOT/'tmp') as directory:
            with patch.object(runner,'tree',compact),patch.object(compact,'optimize',side_effect=RuntimeError('injected failure')):
                runner.run(self.data,Path(directory),end_day=32)
            result=json.loads((Path(directory)/'summary.json').read_text())
            self.assertEqual(result['fallbacks'],4)
            self.assertTrue(json.loads((Path(directory)/'verification.json').read_text())['passed'])

    def test_future_mutation_does_not_change_earlier_execution(self):
        import solve_c_q3_tree as runner
        import pandas as pd
        changed=copy.deepcopy(self.data)
        changed['load'][31,36:]+=1000; changed['pv'][31,36:]+=500
        for key in changed['forecast']:
            if key[0]==self.data['dates'][31] and key[1]>6: changed['forecast'][key]=np.ones(24)*12000
        with tempfile.TemporaryDirectory(dir=tr.ref.ROOT/'tmp') as directory:
            a=Path(directory)/'original'; b=Path(directory)/'changed'
            with patch.object(runner,'tree',compact):
                runner.run(self.data,a,end_day=32); runner.run(changed,b,end_day=32)
            fa=pd.read_csv(a/'ledger.csv'); fb=pd.read_csv(b/'ledger.csv')
            fields=['purchase_kwh','adjusted_purchase_kwh','charge_kwh','discharge_kwh','emergency_kwh','unused_kwh','soc_end_kwh']
            np.testing.assert_allclose(fa[fields].iloc[:36],fb[fields].iloc[:36],atol=1e-6,rtol=0)
            np.testing.assert_allclose(fa.purchase_kwh,fb.purchase_kwh,atol=1e-6,rtol=0)
            ta=pd.read_csv(a/'transactions.csv'); tb=pd.read_csv(b/'transactions.csv')
            np.testing.assert_allclose(ta[ta.update_hour==6].new_kwh,tb[tb.update_hour==6].new_kwh,atol=1e-6,rtol=0)

    def test_compact_same_scenarios_and_fast_dp(self):
        tree=compact.build(self.data,self.pred,31,12)
        paths,w,_=compact.baseline_paths(self.data,self.pred,31,self.pred[31,0],tr.ref.old.pv_forecast_from_issue(self.data,31,12),12)
        np.testing.assert_array_equal(tree['loads'],np.array([p[0] for p in paths])); np.testing.assert_array_equal(tree['weights'],w)
        solution,_=tr.optimize(tree,self.data['fixed_price'],6000,time_limit=30)
        for terminal in (False,True):
            g,a=tr.control_value(tree,solution,self.data['fixed_price'],terminal)
            _,b=compact.control_value(tree,solution,self.data['fixed_price'],terminal)
            np.testing.assert_allclose(a,b,atol=1e-7,rtol=1e-14)


if __name__=='__main__': unittest.main()
