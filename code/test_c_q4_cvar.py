import copy
import unittest
import numpy as np
import q4_model as m


class RiskTests(unittest.TestCase):
    def test_cvar(self):
        self.assertAlmostEqual(m.cvar([7,7,7]),7)
        self.assertAlmostEqual(m.cvar([0,10],[.95,.05]),5)
        self.assertAlmostEqual(m.cvar([0,10],[.9,.1]),10)
        self.assertAlmostEqual(m.cvar([0,10],[.7,.3],0),3)
        rng=np.random.default_rng(0)
        for _ in range(20):
            x=rng.uniform(-30,100,28); w=rng.dirichlet(np.ones(28))
            independent=min(z+10*np.dot(w,np.maximum(x-z,0)) for z in x)
            self.assertAlmostEqual(m.cvar(x,w),independent)

    def test_invalid(self):
        with self.assertRaises(ValueError): m.cvar([1,np.nan])
        with self.assertRaises(ValueError): m.cvar([1,2],[1,1])

    def test_price_prefix(self):
        p=np.ones(144); a=m.price_update(np.ones(36)*2,p,36)
        self.assertAlmostEqual(a[36],2)
        self.assertTrue(a[-1]>1)
        with self.assertRaises(ValueError): m.price_update(np.ones(37),p,36)

    def test_contract(self):
        l=np.array([[100.],[200.]]); v=np.zeros_like(l); p=np.ones_like(l); w=np.array([.5,.5])
        for rho in (0,.2):
            q,s=m.contract(l,v,p,w,1200,[150],[0],rho=rho,terminal=True)
            self.assertFalse(s['fallback']); self.assertAlmostEqual(q[0],200,places=5)
        q,s=m.contract(l,v,p,w,1200,[150],[0],previous=np.array([200.]),charged=np.array([250.]),rho=.2,terminal=True)
        self.assertFalse(s['fallback']); self.assertAlmostEqual(s['scenario_invoice_mean'],250,places=5)

    def test_current_price_dp(self):
        grid,values=m.values(np.array([0.]),np.array([[100.]]),np.array([[0.]]),np.array([[1.]]),np.array([1.]),.4,True)
        a=m.en.dp_dispatch(0,100,0,1200,grid,values[1],0,True)
        self.assertAlmostEqual(a['r'],100); self.assertAlmostEqual(a['end'],1200)

    def test_risk_changes_tail_exposure(self):
        l=np.array([[100.],[200.]]); v=np.zeros_like(l); p=np.ones_like(l); w=np.array([.9,.1])
        neutral,_=m.contract(l,v,p,w,1200,[110],[0],rho=0,terminal=True)
        risk,_=m.contract(l,v,p,w,1200,[110],[0],rho=.2,terminal=True)
        self.assertAlmostEqual(neutral[0],100,places=4)
        self.assertAlmostEqual(risk[0],200,places=4)

    def test_unsettled_fee_remains_in_scenario_risk(self):
        l=np.array([[100.],[100.]]); v=np.zeros_like(l); p=np.array([[1.],[2.]]); w=np.array([.5,.5])
        q,s=m.contract(l,v,p,w,1200,[100],[0],previous=np.array([100.]),charged=np.array([150.]),past_cash=17,rho=.2,terminal=True)
        self.assertFalse(s['fallback']); self.assertAlmostEqual(q[0],100,places=4)
        np.testing.assert_allclose(s['scenario_costs'],[167,317],atol=1e-5)
        self.assertAlmostEqual(s['scenario_invoice_cvar90'],317,places=5)

    def test_terminal_value_is_not_invoice(self):
        l=np.zeros((1,1)); p=np.ones((1,1)); w=np.ones(1)
        for rate in (.2,.4):
            q,s=m.contract(l,l,p,w,6000,[0],[0],rho=.2,rate=rate)
            self.assertFalse(s['fallback']); self.assertAlmostEqual(q[0],0,places=5)
            self.assertAlmostEqual(s['scenario_invoice_cvar90'],0,places=5)
            self.assertAlmostEqual(s['objective_rebuilt'],-rate*4800,places=4)

    def test_causality_and_q2_isolation(self):
        data=m.load_data('q2'); self.assertNotIn('forecast',data)
        arc=m.archive(data); before=m.scenarios(data,arc,45,36,'q2')
        changed=copy.deepcopy(data)
        for field in ('load','pv','price'):
            changed[field][45,36:]*=1.7; changed[field][46:]*=1.4
        after=m.scenarios(changed,m.archive(changed),45,36,'q2')
        for x,y in zip(before[:4],after[:4]): np.testing.assert_allclose(x,y,rtol=0,atol=0)
        self.assertTrue(np.all(before[2]>0)); self.assertEqual(before[4]['history'],list(range(17,45)))

    def test_unpublished_forecast(self):
        data=m.load_data('q3'); arc=m.archive(data); before=m.scenarios(data,arc,45,36,'q3')
        other=copy.deepcopy(data)
        for (d,h),v in other['forecast'].items():
            if d>data['dates'][45] or (d==data['dates'][45] and h>6): v[:]*=10
        after=m.scenarios(other,arc,45,36,'q3')
        for x,y in zip(before[:4],after[:4]): np.testing.assert_array_equal(x,y)


if __name__=='__main__': unittest.main(verbosity=2)
