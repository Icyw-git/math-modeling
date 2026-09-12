import unittest
import numpy as np
import test_q4_solver_quality as common
import q4_solver_cycle as cycle
common.improved=cycle
common.original.m.contract=cycle.contract

class CycleTests(common.QualityTests):
    def test_exact_cycle_elimination(self):
        eta=.9; n=2; k=1; ns=n; stride=6*n+1
        x=np.zeros(ns+k*stride)
        o=ns; x[o:o+n]=[10,4]; x[o+n:o+2*n]=[3,8]; x[o+2*n:o+3*n]=[2,5]
        before_soc=eta*x[o:o+n]-x[o+n:o+2*n]/eta
        before_balance=x[o+n:o+2*n]-x[o:o+n]-x[o+2*n:o+3*n]
        y,count,removed=cycle.eliminate_cycles(x,k,n,ns,stride,eta)
        np.testing.assert_allclose(eta*y[o:o+n]-y[o+n:o+2*n]/eta,before_soc,atol=1e-12)
        np.testing.assert_allclose(y[o+n:o+2*n]-y[o:o+n]-y[o+2*n:o+3*n],before_balance,atol=1e-12)
        self.assertFalse(np.any((y[o:o+n]>1e-9)&(y[o+n:o+2*n]>1e-9)))
        self.assertEqual(count,2); self.assertGreater(removed,0)

    def test_idempotent(self):
        x=np.zeros(15); x[2:4]=[3,0]; x[4:6]=[0,4]
        y,_,_=cycle.eliminate_cycles(x,1,2,2,13,.9)
        z,count,removed=cycle.eliminate_cycles(y,1,2,2,13,.9)
        np.testing.assert_array_equal(y,z); self.assertEqual(count,0); self.assertEqual(removed,0)

if __name__=='__main__': unittest.main(verbosity=2)
