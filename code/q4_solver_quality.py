"""Experimental full-MIP solver; original model and annual checkpoints untouched."""
import time
import numpy as np
from q4_model import base, CFG, cvar

def contract(loads,pvs,prices,w,e0,lh,vh,previous=None,charged=None,past_cash=0.,rho=.2,alpha=.9,terminal=False,rate=.4,limit=30.):
    """Common contract; per-scenario full outstanding invoice inside CVaR."""
    loads=base.finite_nonnegative(loads,'loads'); pvs=base.finite_nonnegative(pvs,'pvs'); prices=base.finite_nonnegative(prices,'prices'); w=np.asarray(w,float)
    k,n=loads.shape; base.validate_state(e0,CFG)
    if pvs.shape!=loads.shape or prices.shape!=loads.shape or min(prices.ravel())<=0 or w.shape!=(k,) or not np.isfinite(w).all() or min(w)<0 or abs(w.sum()-1)>1e-9 or not 0<=rho<=1 or not 0<alpha<1: raise ValueError('Invalid contract inputs')
    first=previous is None; prev=np.zeros(n) if first else base.finite_nonnegative(previous,'previous')
    units=np.zeros(n) if first else base.finite_nonnegative(charged,'charged units')
    if prev.shape!=(n,) or units.shape!=(n,): raise ValueError('Contract shape')
    hs=base.load_solver(); h=hs.Highs(); begun=time.perf_counter()
    for name,val in [('output_flag',False),('threads',1),('random_seed',0),('mip_rel_gap',.001)]: h.setOptionValue(name,val)
    cap=CFG.cap; eta=CFG.eta; demand=loads-pvs; ns=n if first else 3*n; stride=6*n+1
    qmax=np.maximum(np.maximum(demand.max(axis=0),0)+cap,prev)
    def var(lo,hi,cost=0.):
        idx=h.getNumCol(); h.addVar(float(lo),float(hi))
        if cost: h.changeColCost(idx,float(cost))
        return idx
    def row(lo,hi,idx,val): h.addRow(float(lo),float(hi),len(idx),np.asarray(idx,np.int32),np.asarray(val,float))
    for t in range(n): var(0,qmax[t])
    if not first:
        for t in range(n): var(0,qmax[t])
        for t in range(n): var(0,prev[t])
        for t in range(n): row(prev[t],prev[t],[t,n+t,2*n+t],[1,-1,1])
    for j in range(k):
        o=ns+j*stride
        for lo,hi in ([(0,cap)]*(2*n)+[(0,hs.kHighsInf)]*n+list(zip(np.zeros(n),np.maximum(demand[j],0)))+[(1200,10800)]*(n+1)+[(0,1)]*n): var(lo,hi)
        h.changeColBounds(o+4*n,e0,e0)
        if terminal: h.changeColBounds(o+5*n,1200,1200)
        else: h.changeColCost(o+5*n,-w[j]*rate)
        for t in range(n):
            z=o+5*n+1+t; rm=max(demand[j,t],0)
            row(demand[j,t],demand[j,t],[t,o+t,o+n+t,o+2*n+t,o+3*n+t],[1,-1,1,-1,1])
            row(0,0,[o+4*n+t+1,o+4*n+t,o+t,o+n+t],[1,-1,-eta,1/eta])
            row(-hs.kHighsInf,0,[o+t,z],[1,-cap]); row(-hs.kHighsInf,cap,[o+n+t,z],[1,cap]); row(-hs.kHighsInf,rm,[o+3*n+t,z],[1,rm])
    ci=[var(-hs.kHighsInf,hs.kHighsInf,(1-rho)*w[j]) for j in range(k)]
    zeta=var(-hs.kHighsInf,hs.kHighsInf,rho) if rho else None
    xi=[var(0,hs.kHighsInf,rho*w[j]/(1-alpha)) for j in range(k)] if rho else []
    for j in range(k):
        o=ns+j*stride
        idx=[ci[j]]+list(range(o+3*n,o+4*n)); val=[1.]+list(-5*prices[j])
        if first: idx+=list(range(n)); val+=list(-prices[j]); const=past_cash
        else:
            idx+=list(range(n,2*n))+list(range(2*n,3*n)); val+=list(-1.5*prices[j])+list(.5*prices[j]); const=past_cash+float(prices[j]@units)
        row(const,const,idx,val)
        if rho: row(0,hs.kHighsInf,[xi[j],ci[j],zeta],[1,-1,1])
    # Build a complete physical incumbent; no emergency charging, no cycling.
    seed=np.zeros(h.getNumCol())
    seed[:n]=np.minimum(qmax,np.maximum(np.asarray(lh)-vh,0)) if first else prev
    if not first:
        seed[n:2*n]=np.maximum(seed[:n]-prev,0)
        seed[2*n:3*n]=np.maximum(prev-seed[:n],0)
    for j in range(k):
        o=ns+j*stride
        energy=e0
        seed[o+4*n]=energy
        for t in range(n):
            discharge=min(cap,max(0,(energy-1200)*eta)) if terminal else 0.
            energy-=discharge/eta
            surplus=seed[t]+discharge-demand[j,t]
            seed[o+n+t]=discharge
            seed[o+2*n+t]=max(surplus,0)
            seed[o+3*n+t]=max(-surplus,0)
            seed[o+4*n+t+1]=energy
        if terminal and abs(energy-1200)>1e-6:
            raise ValueError('Terminal state unreachable within remaining power limits')
        seed[ci[j]]=past_cash+prices[j]@(seed[:n] if first else units+1.5*seed[n:2*n]-.5*seed[2*n:3*n])+5*prices[j]@seed[o+3*n:o+4*n]
    if rho: seed[zeta]=max(seed[ci])
    # Check all linear rows and variable bounds independently before warm start.
    lp=h.getLp(); mat=lp.a_matrix_
    activity=np.zeros(h.getNumRow())
    for col in range(h.getNumCol()):
        lo,hi=mat.start_[col:col+2]
        for pos in range(lo,hi): activity[mat.index_[pos]]+=mat.value_[pos]*seed[col]
    assert np.all(seed>=np.asarray(lp.col_lower_)-1e-6) and np.all(seed<=np.asarray(lp.col_upper_)+1e-6)
    assert np.all(activity>=np.asarray(lp.row_lower_)-1e-6) and np.all(activity<=np.asarray(lp.row_upper_)+1e-6)
    enforced={ns+j*stride+5*n+1+t for j in range(k) for t in range(n)}
    for z in sorted(enforced): h.changeColIntegrality(z,hs.HighsVarType.kInteger)
    initial=hs.HighsSolution(); initial.col_value=seed.tolist(); initial.value_valid=True
    h.setSolution(initial)
    reason='No feasible incumbent'; status='not run'
    for iteration in range(1):
        h.setOptionValue('solver','ipm' if not enforced else 'choose')
        h.setOptionValue('time_limit',max(.001,limit-(time.perf_counter()-begun))); h.run()
        sol=h.getSolution(); info=h.getInfo(); status=h.modelStatusToString(h.getModelStatus())
        use_seed=not sol.value_valid or info.primal_solution_status!=hs.SolutionStatus.kSolutionStatusFeasible
        x=seed.copy() if use_seed else np.asarray(sol.col_value); bad=set(); residual=0.; costs=[]; ends=[]
        for j in range(k):
            o=ns+j*stride; c,d,s,r,e=x[o:o+n],x[o+n:o+2*n],x[o+2*n:o+3*n],x[o+3*n:o+4*n],x[o+4*n:o+5*n+1]
            bad.update(o+5*n+1+int(t) for t in np.flatnonzero((c>1e-6)&((d>1e-6)|(r>1e-6))))
            residual=max(residual,float(abs(x[:n]+r+d-c-s-demand[j]).max()),float(abs(np.diff(e)-eta*c+d/eta).max()))
            if min(c.min(),d.min(),s.min(),r.min()) < -1e-6 or max(c.max(),d.max())>cap+1e-6 or e.min()<1200-1e-6 or e.max()>10800+1e-6 or abs(e[0]-e0)>1e-6 or (terminal and abs(e[-1]-1200)>1e-6): residual=np.inf
            bill=past_cash+float(prices[j]@(x[:n] if first else units+1.5*x[n:2*n]-.5*x[2*n:3*n])+5*prices[j]@r)
            if abs(bill-x[ci[j]])>1e-5: residual=np.inf
            costs.append(bill); ends.append(float(e[-1]))
        if not bad:
            if not np.isfinite(x).all() or x[:n].min()<-1e-6 or np.any(x[:n]>qmax+1e-6) or residual>1e-6: reason='Physical verification failed'; break
            if not first and (abs(x[:n]-x[n:2*n]+x[2*n:3*n]-prev).max()>1e-6 or np.any((x[n:2*n]>1e-6)&(x[2*n:3*n]>1e-6))): reason='Version verification failed'; break
            risk=cvar(costs,w,alpha); mean=float(w@costs); continuation=0. if terminal else -rate*float(w@(np.asarray(ends)-1200))
            return np.maximum(x[:n],0),dict(status=status,fallback=False,gap=float(info.mip_gap) if not use_seed and np.isfinite(info.mip_gap) else None,dual_bound=float(info.mip_dual_bound) if np.isfinite(info.mip_dual_bound) else None,seed_used=use_seed,proven_optimal=status=='Optimal',seconds=time.perf_counter()-begun,scenario_invoice_mean=mean,scenario_invoice_cvar90=risk,objective_rebuilt=(1-rho)*mean+rho*risk+continuation,scenario_costs=costs,physical_error=residual,integer_pairs=len(enforced))
        new=bad-enforced
        if not new or time.perf_counter()-begun>=limit: reason='Mutual exclusion unresolved'; break
        if iteration>=4: new={ns+j*stride+5*n+1+t for j in range(k) for t in range(n)}-enforced
        for z in sorted(new): h.changeColIntegrality(z,hs.HighsVarType.kInteger)
        enforced.update(new)
    return prev.copy() if not first else np.maximum(np.asarray(lh)-vh,0),dict(status=reason,fallback=True,gap=None,seconds=time.perf_counter()-begun,scenario_invoice_mean=None,scenario_invoice_cvar90=None)


