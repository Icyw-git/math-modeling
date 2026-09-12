"""Run original mathematical tests plus seed-specific edge cases."""
import unittest
import numpy as np
import test_c_q4_cvar as original
import q4_solver_quality as improved
original.m.contract=improved.contract

class QualityTests(original.RiskTests):
    def test_seed_survives_tiny_budget(self):
        l=np.full((2,12),100.); p=np.ones_like(l)
        q,s=improved.contract(l,np.zeros_like(l),p,np.array([.5,.5]),1200,np.full(12,100.),np.zeros(12),limit=.001)
        self.assertFalse(s['fallback'])
        self.assertLessEqual(s['physical_error'],1e-6)
        self.assertTrue(np.isfinite(s['objective_rebuilt']))
        if s['seed_used']: self.assertIsNone(s['gap'])

    def test_terminal_seed(self):
        l=np.zeros((1,12)); p=np.ones_like(l)
        q,s=improved.contract(l,l,p,np.ones(1),6000,np.zeros(12),np.zeros(12),terminal=True,limit=.001)
        self.assertFalse(s['fallback'])
        self.assertLessEqual(s['physical_error'],1e-6)

if __name__=='__main__': unittest.main(verbosity=2)
