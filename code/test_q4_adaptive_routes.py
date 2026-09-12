import unittest
from unittest.mock import patch
import numpy as np
import q4_solver_adaptive as a

class Routes(unittest.TestCase):
    def test_preserve_qualified(self):
        q=np.array([7.]); stats=dict(status='Optimal',fallback=False,gap=0.,integer_pairs=0)
        with patch.object(a.original,'contract',return_value=(q,stats)), patch.object(a.recovery,'contract') as repair:
            result,s=a.contract(None)
            np.testing.assert_array_equal(result,q); repair.assert_not_called()
            self.assertEqual(s['gap'],0.); self.assertEqual(s['route'],'original_qualified')

    def test_recover_failure(self):
        first=dict(status='Mutual exclusion unresolved',fallback=True,gap=None)
        second=dict(status='Time limit reached',fallback=False,gap=None,seed_used=True)
        with patch.object(a.original,'contract',return_value=(np.array([1.]),first)), patch.object(a.recovery,'contract',return_value=(np.array([2.]),second)):
            q,s=a.contract(None)
            self.assertEqual(q[0],2); self.assertEqual(s['route'],'lp_repair'); self.assertEqual(len(s['stages']),2)

    def test_reject_worse_and_clear_false_gap(self):
        first=dict(status='Time limit reached',fallback=False,gap=0.,integer_pairs=0,objective_rebuilt=100.)
        second=dict(status='fixed_mode',fallback=False,gap=.1,objective_rebuilt=120.)
        with patch.object(a.original,'contract',return_value=(np.array([1.]),first)), patch.object(a.recovery,'contract',return_value=(np.array([2.]),second)):
            q,s=a.contract(None)
            self.assertEqual(q[0],1); self.assertIsNone(s['gap'])
            self.assertEqual(s['route'],'original_after_comparison')

    def test_accept_better(self):
        first=dict(status='Time limit reached',fallback=False,gap=.2,integer_pairs=10,objective_rebuilt=100.)
        second=dict(status='fixed_mode',fallback=False,gap=.1,objective_rebuilt=90.)
        with patch.object(a.original,'contract',return_value=(np.array([1.]),first)), patch.object(a.recovery,'contract',return_value=(np.array([2.]),second)):
            q,s=a.contract(None)
            self.assertEqual(q[0],2); self.assertEqual(s['route'],'lp_repair')

    def test_preserve_safe_fallback(self):
        first=dict(status='failed',fallback=True,gap=None)
        with patch.object(a.original,'contract',return_value=(np.array([1.]),first)), patch.object(a.recovery,'contract',return_value=(np.array([2.]),first)):
            q,s=a.contract(None)
            self.assertEqual(q[0],1); self.assertTrue(s['fallback'])

if __name__=='__main__': unittest.main(verbosity=2)
