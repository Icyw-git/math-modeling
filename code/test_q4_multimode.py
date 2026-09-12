import unittest
import test_q4_cycle as common
import q4_solver_multimode as multimode
common.cycle=multimode
common.common.improved=multimode
common.common.original.m.contract=multimode.contract

class MultiModeTests(common.CycleTests):
    def test_candidate_metadata(self):
        import numpy as np
        l=np.array([[100.,200.],[120.,180.]])
        _,s=multimode.contract(l,np.zeros_like(l),np.ones_like(l),np.array([.5,.5]),1200,[110,190],[0,0],limit=2)
        self.assertIn('candidate_count',s); self.assertGreaterEqual(s['candidate_count'],0)
        self.assertIn(s['status'],{'seed','cycle_only','charge_preferred','emergency_preferred','balanced','strict_charge','lenient_charge'})

if __name__=='__main__': unittest.main(verbosity=2)
