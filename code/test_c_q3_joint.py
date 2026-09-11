import unittest
import copy
import numpy as np
import q3_joint_policy as j


class Tests(unittest.TestCase):
    def test_physical(self):
        p=np.ones(144); curve=-.4*(j.GRID-1200)
        for e in (1200,6000,10800):
            for load,pv,q in ((0,0,0),(5000,0,0),(0,5000,0),(1500,300,800)):
                a=j.feedback(q,load,pv,e,p,0,np.zeros(144),.5,curve)
                j.ref.old.b.check_step(q,load,pv,e,{k:float(v) for k,v in a.items()},j.CFG)
    def test_terminal(self):
        a=j.feedback(500,0,0,1800,np.ones(144),143,np.zeros(144),1,np.zeros(9),True)
        self.assertAlmostEqual(float(a['end']),1200)
        j.ref.old.b.check_step(500,0,0,1800,{k:float(v) for k,v in a.items()},j.CFG)
    def test_no_future_observations(self):
        pool=np.arange(7*144).reshape(7,144).astype(float); obs=np.ones(144); b=obs.copy(); b[20:]+=10000
        a=j.forecast(pool,np.ones(7)/7,obs,0,19); c=j.forecast(pool,np.ones(7)/7,b,0,19)
        np.testing.assert_array_equal(a,c)
    def test_terminal_translation(self):
        curve=-.4*(j.GRID-1200)
        a=j.feedback(0,1000,0,6000,np.ones(144),0,np.ones(144)*50,.5,curve)
        b=j.feedback(0,1000,0,6000,np.ones(144),0,np.ones(144)*50,.5,curve+123)
        self.assertEqual(float(a['end']),float(b['end']))

    def test_terminal_history_only(self):
        data=j.ref.old.load_data(); pred,_,_=j.ref.en.forecast_archive(data['load'],data['pv'])
        other=copy.deepcopy(data); other['load'][31:]+=10000; other['pv'][31:]+=20000
        a,_=j.terminal_value(data,pred,31); b,_=j.terminal_value(other,pred,31)
        np.testing.assert_allclose(a,b,atol=1e-6,rtol=0)

    def test_search_and_causal_forecast(self):
        data=j.ref.old.load_data(); pred,_,_=j.ref.en.forecast_archive(data['load'],data['pv'])
        tree=j.compact.build(data,pred,31,18); base=np.maximum(tree['load_hat'][108:]-tree['pv_hat'][108:],0)
        contracts=[base.copy()]; curve=-.4*(j.GRID-1200)
        best,logs=j.search(tree,contracts,base,6000,data['fixed_price'],curve,base)
        self.assertAlmostEqual(best[0],min(x['value'] for x in logs))
        value,details=j.rollout(tree,best[1],6000,data['fixed_price'],curve,best[2],best[3],j.prepared(tree),base,return_details=True)
        self.assertTrue(np.isfinite(value)); self.assertTrue((details['end']>=1200-1e-6).all()); self.assertTrue((details['end']<=10800+1e-6).all())


if __name__=='__main__': unittest.main()
