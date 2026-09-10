import unittest
import numpy as np
import pandas as pd
import solve_c_q2_baseline as b
import solve_c_q2_enhanced as m


class EnhancedTests(unittest.TestCase):
    def test_forecast_prefix_invariance(self):
        rng=np.random.default_rng(8)
        l=rng.uniform(300,1000,(40,144)); v=rng.uniform(0,500,(40,144))
        first=m.forecast_archive(l,v)
        altered=l.copy(); altered[35:]*=3
        later=m.forecast_archive(altered,v)
        np.testing.assert_array_equal(first[0][:36],later[0][:36])
        np.testing.assert_array_equal(first[1][:36],later[1][:36])
        self.assertEqual(first[2][:36],later[2][:36])
        p=m.residual_pool(35,l-v,first[0]-first[1])
        q=m.residual_pool(35,altered-v,later[0]-later[1])
        np.testing.assert_array_equal(p,q)

    def test_belief_prefix_and_paths(self):
        pool=np.repeat(np.arange(28.)[:,None],144,axis=1)
        paths,w,indices=m.representative_paths(pool)
        np.testing.assert_array_equal(paths,pool[indices])
        self.assertAlmostEqual(w.sum(),1)
        with self.assertRaises(ValueError):
            m.scenario_update(np.zeros(144),paths,w,[0]*37,36,10)
        scenario,weights,c=m.scenario_update(np.zeros(144),paths,w,[1]*36,36,10)
        self.assertTrue(np.isfinite(scenario).all())
        self.assertAlmostEqual(weights.sum(),1)

    def test_dp_feasibility_and_terminal(self):
        rng=np.random.default_rng(3)
        q=np.full(144,500.); prices=np.linspace(.4,1.4,144)
        scenarios=rng.uniform(-200,1200,(7,144))
        for terminal in [False,True]:
            grid,v=m.value_table(q,prices,scenarios,np.ones(7)/7,grid_size=49,terminal=terminal)
            e=6000.
            for t in range(144):
                load,pv=rng.uniform(0,1600,2)
                a=m.dp_dispatch(q[t],load,pv,e,grid,v[t+1]/prices[t],143-t,terminal)
                b.check_step(q[t],load,pv,e,a,b.Config())
                e=a['end']
            if terminal: self.assertAlmostEqual(e,1200)

    def test_bellman_observe_then_act(self):
        # No uncertainty: a feasible current action agrees with single-step cost.
        q=np.array([0.]); prices=np.array([1.]); scenarios=np.array([[100.]])
        grid,v=m.value_table(q,prices,scenarios,np.ones(1),grid_size=49,terminal=True)
        a=m.dp_dispatch(0,100,0,1200,grid,v[1],0,True)
        self.assertEqual(a['r'],100)
        self.assertEqual(a['c'],0)
        self.assertAlmostEqual(v[0,0],500)

    def test_same_day_plan_and_actions_causal(self):
        rng=np.random.default_rng(5)
        l=rng.uniform(400,800,(33,144)); v=rng.uniform(0,300,(33,144))
        dates=pd.date_range('2025-01-01',periods=33)
        prices=np.linspace(.4,1.4,144)
        warm=pd.DataFrame({'soc_end_kwh':[1200.]})
        archive=m.forecast_archive(l,v)
        a,_,_,_=m.run_variant(dates,l,v,prices,archive,warm,'enhanced',17,32)
        altered=l.copy(); altered[31,72:]*=2; altered[32]*=3
        other=m.forecast_archive(altered,v)
        c,_,_,_=m.run_variant(dates,altered,v,prices,other,warm,'enhanced',17,32)
        np.testing.assert_array_equal(a.purchase_kwh,c.purchase_kwh)
        for key in ['charge_kwh','discharge_kwh','emergency_kwh','soc_end_kwh']:
            np.testing.assert_array_equal(a[key].iloc[:72],c[key].iloc[:72])


if __name__=='__main__': unittest.main()
