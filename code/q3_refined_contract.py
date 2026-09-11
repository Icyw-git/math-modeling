"""Physically checked shared-contract MILP with selective integrality."""
import time
import numpy as np
import solve_c_q2_baseline as b
CFG=b.Config()

def solve(price,load_hat,pv_hat,scenarios,weights,e0,previous_q=None,terminal_target=None,time_limit=10.,terminal_rate=None):
    price=b.finite_nonnegative(price,'price'); b.validate_state(e0,CFG)
    loads=np.asarray([s[0] for s in scenarios]); pvs=np.asarray([s[1] for s in scenarios]); w=np.asarray(weights,float)
    b.finite_nonnegative(loads,'scenario loads'); b.finite_nonnegative(pvs,'scenario PV')
    k,n=loads.shape
    if pvs.shape!=loads.shape or n!=len(price) or w.shape!=(k,) or not np.isfinite(w).all() or min(w)<0 or abs(w.sum()-1)>1e-9 or min(price)<=0: raise ValueError('Invalid scenarios')
    first=previous_q is None; previous=np.zeros(n) if first else b.finite_nonnegative(previous_q,'previous')
    if previous.shape!=(n,): raise ValueError('Invalid previous shape')
    demand=loads-pvs; hs=b.load_solver(); h=hs.Highs(); begun=time.perf_counter()
    for name,value in [('output_flag',False),('threads',1),('random_seed',0),('mip_rel_gap',.001)]: h.setOptionValue(name,value)
    cap=CFG.cap; eta=CFG.eta; ns=n if first else 3*n; stride=6*n+1
    # Keep the old contract in the feasible set even after a lower forecast.
    qmax=np.maximum(np.maximum(demand.max(axis=0),0)+cap,previous)
    for t in range(n):
        h.addVar(0,float(qmax[t])); h.changeColCost(t,float(price[t]) if first else 0.)
    if not first:
        for t in range(n): h.addVar(0,float(qmax[t])); h.changeColCost(n+t,float(1.5*price[t]))
        for t in range(n): h.addVar(0,float(previous[t])); h.changeColCost(2*n+t,float(-.5*price[t]))
    def row(lo,hi,idx,val): h.addRow(float(lo),float(hi),len(idx),np.asarray(idx,np.int32),np.asarray(val,float))
    if not first:
        for t in range(n): row(previous[t],previous[t],[t,n+t,2*n+t],[1,-1,1])
    for j in range(k):
        o=ns+j*stride
        for lo,hi in ([(0,cap)]*(2*n)+[(0,hs.kHighsInf)]*n+list(zip(np.zeros(n),np.maximum(demand[j],0)))+[(CFG.emin,CFG.emax)]*(n+1)+[(0,1)]*n): h.addVar(float(lo),float(hi))
        h.changeColBounds(o+4*n,e0,e0)
        if terminal_target is None: h.changeColCost(o+5*n,-float(w[j]*(price.min()/eta if terminal_rate is None else terminal_rate)))
        else: h.changeColBounds(o+5*n,terminal_target,terminal_target)
        for t in range(n):
            z=o+5*n+1+t; rm=max(demand[j,t],0)
            h.changeColCost(o+3*n+t,float(w[j]*5*price[t]))
            row(demand[j,t],demand[j,t],[t,o+t,o+n+t,o+2*n+t,o+3*n+t],[1,-1,1,-1,1])
            row(0,0,[o+4*n+t+1,o+4*n+t,o+t,o+n+t],[1,-1,-eta,1/eta])
            row(-hs.kHighsInf,0,[o+t,z],[1,-cap]); row(-hs.kHighsInf,cap,[o+n+t,z],[1,cap]); row(-hs.kHighsInf,rm,[o+3*n+t,z],[1,rm])
    enforced=set(); error='No incumbent'; status='not solved'
    for iteration in range(7):
        h.setOptionValue('time_limit',max(.001,time_limit-(time.perf_counter()-begun))); h.run()
        sol=h.getSolution(); info=h.getInfo(); status=h.modelStatusToString(h.getModelStatus())
        if not sol.value_valid or info.primal_solution_status!=hs.SolutionStatus.kSolutionStatusFeasible: error=status; break
        x=np.asarray(sol.col_value); bad=set()
        for j in range(k):
            o=ns+j*stride
            bad.update(o+5*n+1+int(t) for t in np.flatnonzero((x[o:o+n]>1e-6)&((x[o+n:o+2*n]>1e-6)|(x[o+3*n:o+4*n]>1e-6))))
        if not bad:
            try:
                assert np.isfinite(x).all() and x[:n].min()>=-1e-6 and (x[:n]<=qmax+1e-6).all()
                residual=0.
                for j in range(k):
                    o=ns+j*stride; c,d,s,r,e=x[o:o+n],x[o+n:o+2*n],x[o+2*n:o+3*n],x[o+3*n:o+4*n],x[o+4*n:o+5*n+1]
                    residual=max(residual,float(abs(x[:n]+r+d-c-s-demand[j]).max()),float(abs(np.diff(e)-eta*c+d/eta).max()))
                    assert min(c.min(),d.min(),r.min(),s.min())>=-1e-6 and max(c.max(),d.max())<=cap+1e-6
                    assert e.min()>=CFG.emin-1e-6 and e.max()<=CFG.emax+1e-6 and abs(e[0]-e0)<=1e-6
                    if terminal_target is not None: assert abs(e[-1]-terminal_target)<=1e-6
                assert residual<=1e-6
                if not first: assert abs(x[:n]-x[n:2*n]+x[2*n:3*n]-previous).max()<=1e-6
            except AssertionError: error='incumbent verification failed'; break
            return np.maximum(x[:n],0),dict(status=status,objective=float(info.objective_function_value),gap=float(info.mip_gap) if np.isfinite(info.mip_gap) else 0.,fallback=False,seconds=time.perf_counter()-begun,physical_error=residual,integer_pairs=len(enforced))
        new=bad-enforced
        if not new or time.perf_counter()-begun>=time_limit: error='mutual exclusion unresolved'; break
        if iteration>=4: new={ns+j*stride+5*n+1+t for j in range(k) for t in range(n)}-enforced
        for z in sorted(new): h.changeColIntegrality(z,hs.HighsVarType.kInteger)
        enforced.update(new)
    fallback=previous if not first else np.maximum(np.asarray(load_hat)-pv_hat,0)
    return fallback,dict(status=error,objective=None,gap=None,fallback=True,seconds=time.perf_counter()-begun,physical_error=None,integer_pairs=len(enforced))
