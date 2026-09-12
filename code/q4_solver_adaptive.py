"""Opt-in recovery: preserve a qualified original solution, repair failures only."""
import time
import q4_model as original
import q4_solver_multimode as recovery

def contract(*args, recovery_limit=30., refine_gap=.001, **kwargs):
    begun=time.perf_counter()
    q,first=original.contract(*args,**kwargs)
    first=dict(first)
    first['proven_optimal']=first['status']=='Optimal'
    if first.get('integer_pairs',0)==0 and not first['proven_optimal']:
        first['gap']=None
    needs_repair=first['fallback'] or (not first['proven_optimal'] and (first.get('gap') is None or first['gap']>refine_gap))
    if not needs_repair:
        return q,dict(first,route='original_qualified',stages=[dict(first)])
    recovery_kwargs=dict(kwargs,limit=recovery_limit)
    candidate,second=recovery.contract(*args,**recovery_kwargs)
    stages=[first,dict(second)]
    if not second['fallback'] and (first['fallback'] or second['objective_rebuilt']<first['objective_rebuilt']-1e-6):
        return candidate,dict(second,route='lp_repair',stages=stages,seconds=time.perf_counter()-begun)
    if not first['fallback']:
        return q,dict(first,route='original_after_comparison',stages=stages,seconds=time.perf_counter()-begun)
    return q,dict(first,route='both_failed',stages=stages,seconds=time.perf_counter()-begun)
