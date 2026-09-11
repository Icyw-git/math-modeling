"""Historical two-stage terminal-value approximation (not a scenario tree)."""
import time
import numpy as np
import solve_c_q2_baseline as b
from solve_c_q2_low_waste import check_scenarios

def estimate(prices, demand, weights, initial, final=False, cfg=b.Config()):
    prices=b.finite_nonnegative(prices,'prices')
    demand=np.asarray(demand,float); weights=np.asarray(weights,float)
    b.validate_state(initial,cfg)
    if demand.ndim!=2 or demand.shape[1]!=len(prices) or not np.isfinite(demand).all():
        raise ValueError('Invalid demand')
    k,n=demand.shape
    if weights.shape!=(k,) or not np.isfinite(weights).all() or min(weights)<0 or abs(weights.sum()-1)>1e-9 or min(prices)<=0:
        raise ValueError('Invalid weights/prices')
    hs=b.load_solver(); h=hs.Highs(); begun=time.perf_counter()
    for key,val in [('output_flag',False),('threads',1),('random_seed',0),('time_limit',cfg.time_limit),('mip_rel_gap',cfg.mip_gap)]:
        h.setOptionValue(key,val)
    stride=6*n+1; qmax=np.maximum(demand.max(axis=0),0)+cfg.cap
    for t in range(n):
        h.addVar(0,float(qmax[t])); h.changeColCost(t,float(prices[t]))
    def row(lo,hi,idx,val):
        h.addRow(float(lo),float(hi),len(idx),np.asarray(idx,np.int32),np.asarray(val,float))
    for j in range(k):
        off=n+j*stride; rmax=np.maximum(demand[j],0)
        bounds=list(zip(np.zeros(n),rmax))+[(0,cfg.cap)]*(2*n)+list(zip(np.zeros(n),qmax+np.maximum(-demand[j],0)+cfg.cap))+[(cfg.emin,cfg.emax)]*(n+1)+[(0,1)]*n
        for lo,hi in bounds: h.addVar(float(lo),float(hi))
        h.changeColBounds(off+4*n,initial,initial)
        if final: h.changeColBounds(off+5*n,cfg.emin,cfg.emin)
        else: h.changeColCost(off+5*n,-float(weights[j]*prices.min()/cfg.eta))
        for t in range(n):
            z=off+5*n+1+t
            h.changeColCost(off+t,float(weights[j]*cfg.emergency_multiplier*prices[t]))
            row(demand[j,t],demand[j,t],[t,off+t,off+n+t,off+2*n+t,off+3*n+t],[1,1,-1,1,-1])
            row(0,0,[off+4*n+t+1,off+4*n+t,off+n+t,off+2*n+t],[1,-1,-cfg.eta,1/cfg.eta])
            row(-hs.kHighsInf,0,[off+n+t,z],[1,-cfg.cap])
            row(-hs.kHighsInf,cfg.cap,[off+2*n+t,z],[1,cfg.cap])
            row(-hs.kHighsInf,rmax[t],[off+t,z],[1,rmax[t]])
    # Refine the relaxation only where physical mutual exclusion is violated.
    enforced=set()
    for iteration in range(7):
        h.setOptionValue('time_limit',max(.001,cfg.time_limit-(time.perf_counter()-begun)))
        h.run(); sol=h.getSolution(); info=h.getInfo()
        if not sol.value_valid or info.primal_solution_status!=hs.SolutionStatus.kSolutionStatusFeasible:
            raise RuntimeError(h.modelStatusToString(h.getModelStatus()))
        x=np.asarray(sol.col_value); rec=[]; bad=set()
        for j in range(k):
            o=n+j*stride
            p=dict(r=x[o:o+n],c=x[o+n:o+2*n],d=x[o+2*n:o+3*n],s=x[o+3*n:o+4*n],e=x[o+4*n:o+5*n+1]); rec.append(p)
            bad.update(o+5*n+1+int(t) for t in np.flatnonzero((p['c']>cfg.tolerance)&((p['r']>cfg.tolerance)|(p['d']>cfg.tolerance))))
        if not bad:
            check_scenarios(dict(q=x[:n],recourse=rec),demand,initial,cfg)
            if final: assert all(abs(p['e'][-1]-cfg.emin)<=cfg.tolerance for p in rec)
            value=float(prices@x[:n]+sum(weights[j]*(cfg.emergency_multiplier*prices@p['r']-(0 if final else prices.min()/cfg.eta*(p['e'][-1]-cfg.emin))) for j,p in enumerate(rec)))
            return dict(value=value,status=h.modelStatusToString(h.getModelStatus()),gap=float(info.mip_gap) if np.isfinite(info.mip_gap) else None,seconds=time.perf_counter()-begun)
        new=bad-enforced
        if not new or time.perf_counter()-begun>=cfg.time_limit: raise RuntimeError('No physically valid incumbent within limit')
        if iteration>=4: new={n+j*stride+5*n+1+t for j in range(k) for t in range(n)}-enforced
        for z in sorted(new): h.changeColIntegrality(z,hs.HighsVarType.kInteger)
        enforced.update(new)
    raise RuntimeError('Refinement exhausted')

def curve(prices,demand,weights,final=False,solver=estimate):
    cfg=b.Config(); grid=np.linspace(cfg.emin,cfg.emax,25); records=[]
    for e in grid:
        try: records.append(dict(energy=float(e),**solver(prices,demand,weights,float(e),final)))
        except (RuntimeError,AssertionError) as exc:
            records.append(dict(energy=float(e),status='fallback',reason=str(exc)))
            return grid,-min(prices)/cfg.eta*(grid-cfg.emin),records,True
    values=np.array([r['value'] for r in records]); return grid,values-values[0],records,False

def value_table(q,prices,scenarios,weights,terminal_values,start=0):
    cfg=b.Config(); grid=np.linspace(cfg.emin,cfg.emax,193); n=len(q)
    values=np.empty((n+1,len(grid))); values[n]=terminal_values
    delta=grid[None,:]-grid[:,None]; c=np.maximum(delta,0)/cfg.eta; d=np.maximum(-delta,0)*cfg.eta
    physical=(c<=cfg.cap+1e-8)&(d<=cfg.cap+1e-8)
    for t in range(n-1,start-1,-1):
        expected=np.zeros(len(grid))
        for scenario,w in zip(scenarios,weights):
            balance=q[t]-scenario[t]; allowed=physical&((c<=1e-8)|(c<=balance+1e-8))
            cost=cfg.emergency_multiplier*prices[t]*np.maximum(-balance+c-d,0)+values[t+1][None,:]
            expected+=w*np.where(allowed,cost,1e12).min(axis=1)
        values[t]=expected
    return grid,values
