"""Bootstrap project-local SciPy before discovering legacy Q3 tests."""
import json
import unittest
import solve_c_q3_refined as ref

if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(ref.ROOT/'code'),pattern='test_c_q3*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=ref.ROOT/'results/q3_tree'; out.mkdir(parents=True,exist_ok=True)
    (out/'test_results.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)),indent=2),encoding='utf8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
