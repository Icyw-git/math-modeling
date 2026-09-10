import unittest
import numpy as np
import solve_c_q2_low_waste as m
import solve_c_q2_enhanced as old


class LowWasteTests(unittest.TestCase):
    def test_scenario_plan_physics(self):
        prices=np.array([.4,.4,1.4,1.4,.8,.4])
        scenarios=np.array([[300,400,-900,-600,800,400],[400,300,-400,-200,1100,700],[800,800,200,300,1600,800]],float)
        p=m.scenario_plan(prices,scenarios,np.array([.25,.5,.25]),1200.)
        self.assertFalse(p['fallback'],p['status'])
        m.check_scenarios(p,scenarios,1200.)

    def test_bad_input(self):
        with self.assertRaises(ValueError): m.scenario_plan(np.ones(3),np.zeros((2,3)),np.ones(2),1200)
        with self.assertRaises(ValueError): m.scenario_plan(np.ones(3),np.full((2,3),np.nan),np.ones(2)/2,1200)

    def test_conversion_loss_identity(self):
        # No benefit from relabelling discarded energy as conversion loss.
        c=100.; d=.81*c
        loss=.1*c+(1/.9-1)*d
        self.assertAlmostEqual(loss,c-d)
        self.assertAlmostEqual(.9*c-d/.9,0.)

    def test_plan_future_invariance_and_reproducible(self):
        rng=np.random.default_rng(17)
        l=rng.uniform(300,1000,(35,144)); v=rng.uniform(0,700,(35,144))
        a=old.forecast_archive(l,v)
        changed=l.copy(); changed[31:]*=4
        other=old.forecast_archive(changed,v)
        pool=old.residual_pool(31,l-v,a[0]-a[1])
        pool2=old.residual_pool(31,changed-v,other[0]-other[1])
        np.testing.assert_array_equal(pool,pool2)
        paths,w,_=old.representative_paths(pool,3)
        paths2,w2,_=old.representative_paths(pool2,3)
        demand=(a[0][31,0]-a[1][31,0])[None,:6]+paths[:,:6]
        demand2=(other[0][31,0]-other[1][31,0])[None,:6]+paths2[:,:6]
        prices=np.linspace(.4,1.4,6)
        p=m.scenario_plan(prices,demand,w,1200)
        p2=m.scenario_plan(prices,demand2,w2,1200)
        np.testing.assert_allclose(p['q'],p2['q'],atol=1e-6,rtol=0)


if __name__=='__main__': unittest.main()
