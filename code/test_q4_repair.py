"""Mathematical regression and certified-bound checks for the LP repair."""
import unittest
import numpy as np
import test_q4_solver_quality as common
import q4_solver_repair as repair
common.improved=repair
common.original.m.contract=repair.contract

class RepairTests(common.QualityTests):
    def test_lower_bound_matches_offset(self):
        l=np.zeros((1,1)); p=np.ones((1,1))
        _,s=repair.contract(l,l,p,np.ones(1),6000,[0],[0],rate=.4)
        self.assertAlmostEqual(s['objective_rebuilt'],-1920.,places=5)
        self.assertAlmostEqual(s['lower_bound'],-1920.,places=5)
        self.assertTrue(s['proven_optimal'])

    def test_seed_is_explicit_when_no_budget(self):
        l=np.full((1,1),100.)
        _,s=repair.contract(l,np.zeros_like(l),np.ones_like(l),np.ones(1),1200,[100],[0],limit=0)
        self.assertTrue(s['seed_used']); self.assertIsNone(s['lower_bound'])
        self.assertIsNone(s['gap']); self.assertFalse(s['proven_optimal'])

if __name__=='__main__': unittest.main(verbosity=2)
