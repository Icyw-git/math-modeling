import unittest

import numpy as np
import pandas as pd

import solve_c_q3_latest as q3


class Q3LatestTests(unittest.TestCase):
    def test_contract_milp_returns_bounded_nonnegative_contract(self):
        n = 4
        price = np.array([0.4, 0.8, 1.0, 0.6])
        load = np.array([2.0, 3.0, 2.5, 1.5])
        pv = np.array([0.0, 0.5, 1.0, 0.0])
        q, meta = q3.contract_milp(
            price, load, pv, [(load, pv)], np.array([1.0]),
            q3.EMIN, time_limit=1.0,
        )
        qmax = np.maximum(load - pv, 0.0) + q3.CAP
        self.assertEqual(len(q), n)
        self.assertTrue(np.isfinite(q).all())
        self.assertGreaterEqual(q.min(), -1e-7)
        self.assertTrue(np.all(q <= qmax + 1e-7))
        self.assertIn("fallback", meta)

    def test_later_contract_adjustment_is_relative_to_previous_version(self):
        n = 3
        price = np.ones(n)
        load = np.array([2.0, 2.0, 2.0])
        pv = np.zeros(n)
        previous = np.array([1.0, 1.5, 1.0])
        q, _ = q3.contract_milp(
            price, load, pv, [(load, pv)], np.array([1.0]),
            q3.EMIN, previous_q=previous, time_limit=1.0,
        )
        self.assertEqual(q.shape, previous.shape)
        self.assertTrue(np.isfinite(q).all())
        self.assertGreaterEqual(q.min(), -1e-7)

    def test_dispatch_never_charges_while_emergency_purchase_is_positive(self):
        grid = np.linspace(q3.EMIN, q3.EMAX, 9)
        continuation = np.zeros(len(grid))
        action = q3.dispatch(
            q=0.0, load=1000.0, pv=0.0, e=q3.EMIN,
            grid=grid, continuation=continuation, price=1.0,
        )
        self.assertGreater(action["r"], 0.0)
        self.assertLessEqual(action["c"], 1e-8)
        self.assertGreaterEqual(action["end"], q3.EMIN - 1e-8)

    def test_value_table_terminal_prefers_minimum_soc(self):
        q = np.zeros(3)
        price = np.ones(3)
        load = np.ones(3)
        pv = np.zeros(3)
        grid, values = q3.value_table(
            q, price, [(load, pv)], np.array([1.0]),
            terminal=True, grid_size=9,
        )
        self.assertEqual(values.shape, (4, 9))
        self.assertAlmostEqual(values[-1, 0], 0.0)
        self.assertGreater(values[-1, -1], values[-1, 0])

    def test_billing_identity_with_multiple_contract_changes(self):
        n = q3.N
        q0 = np.ones(n)
        final_q = q0.copy()
        final_q[0] += 2.0
        final_q[1] -= 1.0
        final_q[2] += 1.0
        final_q[3] -= 0.5
        frame = pd.DataFrame({
            "adjusted_purchase_kwh": final_q,
            "emergency_kwh": np.zeros(n),
        })
        changes = [
            {"start": 0, "increase": np.array([2.0, 0.0, 0.0, 0.0]),
             "decrease": np.array([0.0, 1.0, 0.0, 0.0])},
            {"start": 2, "increase": np.array([1.0, 0.0]),
             "decrease": np.array([0.0, 0.5])},
        ]
        bill, _ = q3.compute_day_bill(frame, q0, changes, q3.FIXED_PRICE)
        expected = (
            np.dot(q3.FIXED_PRICE, q0)
            + np.dot(1.5 * q3.FIXED_PRICE[:4], [2.0, 0.0, 0.0, 0.0])
            - np.dot(q3.FIXED_PRICE[:4], [0.0, 1.0, 0.0, 0.0])
            + np.dot(0.5 * q3.FIXED_PRICE[:4], [0.0, 1.0, 0.0, 0.0])
            + np.dot(1.5 * q3.FIXED_PRICE[2:4], [1.0, 0.0])
            - np.dot(q3.FIXED_PRICE[2:4], [0.0, 0.5])
            + np.dot(0.5 * q3.FIXED_PRICE[2:4], [0.0, 0.5])
        )
        self.assertAlmostEqual(bill["total_fee_yuan"], expected, places=8)
        self.assertAlmostEqual(bill["bill_identity_error_yuan"], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
