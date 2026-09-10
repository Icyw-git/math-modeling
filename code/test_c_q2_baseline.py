"""Run: python -m unittest discover -s code -p test_c_q2_baseline.py -v"""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import solve_c_q2_baseline as m


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = m.Config()

    def test_forecast_lags_and_copy(self):
        x = np.repeat(np.arange(10.)[:, None], 144, axis=1)
        self.assertEqual(m.forecast(x[:0], x[:0])[2], 'cold_start_no_history')
        self.assertTrue((m.forecast(x[:5], x[:5])[0] == 4).all())
        l, v, _ = m.forecast(x, x)
        self.assertTrue((l == 3).all())
        self.assertTrue((v == 9).all())
        l[:] = 99
        self.assertTrue((x[3] == 3).all())

    def test_dispatch_edges(self):
        cases = [(0, 2000, 0, 1200, 1200), (2000, 0, 0, 10800, 10800),
                 (3000, 0, 0, 1200, 10800), (0, 3000, 0, 10800, 1200),
                 (0, 1000, 0, 6000, 6000), (0, 0, 3000, 6000, 10800),
                 (0, 0, 0, 6000, 6000)]
        for args in cases:
            with self.subTest(args=args):
                a = m.dispatch(*args)
                m.check_step(*args[:4], a, self.cfg)
        self.assertEqual(m.dispatch(*cases[0])['r'], 2000)
        self.assertEqual(m.dispatch(*cases[1])['s'], 2000)
        self.assertAlmostEqual(m.dispatch(*cases[2])['c'], self.cfg.cap)
        self.assertAlmostEqual(m.dispatch(*cases[3])['d'], self.cfg.cap)
        self.assertEqual(m.dispatch(*cases[4])['r'], 1000)  # reserve is retained

    def test_random_feasibility(self):
        rng = np.random.default_rng(42)
        for _ in range(2000):
            q, l, v = rng.uniform(0, 4000, 3)
            e, ref = rng.uniform(1200, 10800, 2)
            a = m.dispatch(q, l, v, e, ref)
            m.check_step(q, l, v, e, a, self.cfg)

    def test_bad_inputs_fatal(self):
        for invalid in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                m.dispatch(invalid, 1, 0, 6000, 6000)
        with self.assertRaises(ValueError):
            m.dispatch(0, 1, 0, 11000, 6000)
        with self.assertRaises(ValueError):
            m.plan_day(np.array([-1.]), np.ones(1), np.zeros(1), 6000)

    def test_dispatch_safety_fallback(self):
        original = m.check_step
        with patch.object(m, 'check_step', side_effect=[AssertionError('injected'), None]):
            a = m.dispatch(100, 1000, 0, 6000, 1200)
        self.assertTrue(a['fallback'])
        self.assertEqual(a['r'], 900)
        original(100, 1000, 0, 6000, a, self.cfg)

    def test_solver_feasible_and_fallback(self):
        p = np.array([.4, .5, 1.4, 1.0])
        l, v = np.full(4, 600.), np.array([0., 1000., 0., 0.])
        result = m.plan_day(p, l, v, 6000)
        self.assertFalse(result['fallback'], result['status'])
        m.check_plan(result, p, l, v, 6000, self.cfg)
        self.assertAlmostEqual(result['objective'], result['plan_cost']+result['terminal_penalty'], places=6)

        def broken():
            raise RuntimeError('injected solver failure')
        result = m.plan_day(p, l, v, 6000, solver_factory=broken)
        self.assertTrue(result['fallback'])
        m.check_plan(result, p, l, v, 6000, self.cfg)

    def test_causality_and_repeatability(self):
        rng = np.random.default_rng(0)
        l = rng.uniform(300, 900, (3, 144))
        v = rng.uniform(0, 300, (3, 144))
        price = np.linspace(.4, 1.4, 144)
        dates = pd.date_range('2025-01-01', periods=3)
        first, days = m.simulate(dates, l, v, price)
        repeat, _ = m.simulate(dates, l, v, price)
        pd.testing.assert_frame_equal(first, repeat)
        altered_l, altered_v = l.copy(), v.copy()
        altered_l[2] *= 10
        altered_v[2] *= 0
        changed, _ = m.simulate(dates, altered_l, altered_v, price)
        pd.testing.assert_frame_equal(first.iloc[:288], changed.iloc[:288])
        for key in ('purchase_kwh', 'reference_soc_end_kwh', 'predicted_load_kwh', 'predicted_pv_kwh'):
            np.testing.assert_array_equal(first[key].iloc[288:], changed[key].iloc[288:])
        # Future within-day measurements cannot change earlier controls.
        altered_l, altered_v = l.copy(), v.copy()
        altered_l[1, 72:] *= 4
        changed, _ = m.simulate(dates, altered_l, altered_v, price)
        pd.testing.assert_frame_equal(first.iloc[:216], changed.iloc[:216])
        m.validate_ledger(first, self.cfg)
        m.summarize(first, days, self.cfg)

    def test_event_midnight_split(self):
        f = pd.DataFrame(dict(date=['2025-02-01','2025-02-01','2025-02-02'],
            interval_start=['23:40','23:50','00:00'], interval_end=['23:50','24:00','00:10'],
            emergency_kwh=[1.,2.,3.], emergency_cost_yuan=[5.,10.,15.]))
        events = m.emergency_events(f, self.cfg)
        self.assertEqual(len(events), 2)
        self.assertEqual(events.emergency_kwh.tolist(), [3.,3.])


if __name__ == '__main__':
    unittest.main()
