import unittest
import numpy as np
import q2_terminal_value as tv
import solve_c_q2_enhanced as en

class Tests(unittest.TestCase):
    def test_linear_replay_and_shift(self):
        q=np.array([100.,200.]); p=np.array([1.,2.]); s=np.array([[300.,0.],[0.,400.]]); w=np.array([.5,.5])
        grid,old=en.value_table(q,p,s,w,grid_size=193)
        _,new=tv.value_table(q,p,s,w,old[-1]); np.testing.assert_allclose(new,old,atol=1e-9)
        _,shift=tv.value_table(q,p,s,w,old[-1]+123)
        np.testing.assert_allclose(shift,new+123,atol=1e-9)
    def test_fallback(self):
        def fail(*args): raise RuntimeError('injected')
        grid,val,rec,fallback=tv.curve(np.ones(2),np.ones((1,2)),np.ones(1),solver=fail)
        self.assertTrue(fallback); np.testing.assert_allclose(val,-(grid-1200)/.9)
    def test_efficiency_and_interpolation(self):
        p=tv.estimate(np.ones(2),np.zeros((1,2)),np.ones(1),1200)
        # Buying to preserve terminal energy at its replacement cost is neutral.
        self.assertAlmostEqual(p['value'],0,places=6)
        self.assertAlmostEqual(np.interp(1400,[1200,1600],[0,-200]),-100)
    def test_invalid(self):
        with self.assertRaises(ValueError): tv.estimate(np.ones(2),np.array([[np.nan,0]]),np.ones(1),1200)

    def test_future_history_invariance(self):
        rng=np.random.default_rng(17); load=rng.uniform(0,1000,(34,144)); pv=rng.uniform(0,300,(34,144))
        a=en.forecast_archive(load,pv); changed=load.copy(); changed[31:]*=2
        other=en.forecast_archive(changed,pv)
        np.testing.assert_array_equal(a[0][31],other[0][31])
        np.testing.assert_array_equal(a[1][31],other[1][31])
        np.testing.assert_array_equal(en.residual_pool(31,load-pv,a[0]-a[1]),en.residual_pool(31,changed-pv,other[0]-other[1]))

    def test_boundary_execution(self):
        import solve_c_q2_baseline as b
        prices=np.ones(144); q=np.zeros(144); scenarios=np.full((1,144),2000.)
        tail=np.full(193,1e12); tail[0]=0
        grid,values=tv.value_table(q,prices,scenarios,np.ones(1),tail)
        e=10800.
        for t in range(144):
            a=en.dp_dispatch(0,2000,0,e,grid,values[t+1],143-t,True)
            b.check_step(0,2000,0,e,a,b.Config()); e=a['end']
        self.assertAlmostEqual(e,1200)

if __name__=='__main__': unittest.main()
