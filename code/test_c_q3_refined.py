import unittest
import numpy as np
import solve_c_q3_refined as m

class Tests(unittest.TestCase):
    def test_solver_and_contract_retention(self):
        p=np.ones(3); l=np.array([100.,200.,300.]); v=np.zeros(3)
        q,s=m.opt.solve(p,l,v,[(l,v)],np.ones(1),1200,previous_q=np.full(3,2000.))
        self.assertFalse(s['fallback']); self.assertLess(s['physical_error'],1e-6); self.assertTrue((q>=0).all())
    def test_annual_terminal(self):
        grid=np.linspace(1200,10800,97); terminal=np.full(97,1e12); terminal[0]=0
        a=m.en.dp_dispatch(0,0,0,1500,grid,terminal,0,True)
        self.assertAlmostEqual(a['end'],1200)
    def test_forecast_causal(self):
        data=m.old.load_data(); pred,_,_=m.en.forecast_archive(data['load'],data['pv']); day=31; hour=6
        hat=m.old.pv_forecast_from_issue(data,day,hour)
        a=m.conditioned_paths(data,pred,day,pred[day,0],hat,hour)
        data2=dict(data,load=data['load'].copy(),pv=data['pv'].copy()); data2['load'][day,36:]*=2; data2['pv'][day,36:]*=2
        other=m.old.pv_forecast_from_issue(data2,day,hour)
        c=m.conditioned_paths(data2,pred,day,pred[day,0],other,hour)
        np.testing.assert_array_equal(a[0],c[0]); np.testing.assert_array_equal(a[1],c[1])
    def test_invalid_data(self):
        with self.assertRaises(ValueError): m.opt.solve(np.ones(2),np.ones(2),np.zeros(2),[(np.array([np.nan,1]),np.zeros(2))],np.ones(1),1200)
    def test_hour_boundary_has_no_current_actual_access(self):
        data=m.old.load_data(); day=31
        for hour in (0,6,12,18):
            a=m.old.pv_forecast_from_issue(data,day,hour)
            other=dict(data,pv=data['pv'].copy()); other['pv'][day,hour*6:]+=10000
            c=m.old.pv_forecast_from_issue(other,day,hour)
            np.testing.assert_array_equal(a[hour*6:],c[hour*6:])
    def test_fallback_is_visible(self):
        # A one-slot hard terminal cannot empty a full battery: no feasible plan.
        q,meta=m.opt.solve(np.ones(1),np.zeros(1),np.zeros(1),[(np.zeros(1),np.zeros(1))],np.ones(1),10800,previous_q=np.ones(1),terminal_target=1200,time_limit=1)
        self.assertTrue(meta['fallback']); np.testing.assert_array_equal(q,np.ones(1))
    def test_no_adjustment(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as d:
            m.run(m.old.load_data(),Path(d),'solver',32,25,2.,())
            f=m.pd.read_csv(Path(d)/'ledger.csv'); np.testing.assert_array_equal(f.purchase_kwh,f.adjusted_purchase_kwh)

if __name__=='__main__': unittest.main()
