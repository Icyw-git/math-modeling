"""LP relaxation with exact charge/discharge cycle elimination."""
import time
import numpy as np
from q4_model import base, CFG, cvar

def eliminate_cycles(x,k,n,ns,stride,eta,tol=1e-9):
    """Remove c/d cycling exactly; preserve balance, SOC, invoice and objective."""
    y=np.asarray(x,float).copy(); removed=0; energy=0.
    for j in range(k):
        o=ns+j*stride
        c=y[o:o+n]; d=y[o+n:o+2*n]; spill=y[o+2*n:o+3*n]
        delta=np.minimum(c,d/(eta*eta))
        active=delta>tol
        if np.any(active):
            c-=delta; d-=eta*eta*delta; spill+=(1-eta*eta)*delta
            removed+=int(active.sum()); energy+=float(delta.sum())
            c[abs(c)<tol]=0.; d[abs(d)<tol]=0.
    return y,removed,energy

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
    starts=np.asarray(mat.start_); indices=np.asarray(mat.index_); coefficients=np.asarray(mat.value_)
    if mat.format_==hs.MatrixFormat.kRowwise:
        for ri in range(h.getNumRow()):
            lo,hi=starts[ri:ri+2]
            activity[ri]=coefficients[lo:hi]@seed[indices[lo:hi]]
    else:
        for col in range(h.getNumCol()):
            lo,hi=starts[col:col+2]
            np.add.at(activity,indices[lo:hi],coefficients[lo:hi]*seed[col])
    assert np.all(seed>=np.asarray(lp.col_lower_)-1e-6) and np.all(seed<=np.asarray(lp.col_upper_)+1e-6)
    assert np.all(activity>=np.asarray(lp.row_lower_)-1e-6) and np.all(activity<=np.asarray(lp.row_upper_)+1e-6)

    offset=0. if terminal else rate*1200
    def evaluate(x):
        if not np.isfinite(x).all(): return None
        if np.any(x[:n]<-1e-6) or np.any(x[:n]>qmax+1e-6): return None
        if not first and (abs(x[:n]-x[n:2*n]+x[2*n:3*n]-prev).max()>1e-6 or np.any((x[n:2*n]>1e-6)&(x[2*n:3*n]>1e-6))): return None
        costs=[]; ends=[]; error=0.
        for j in range(k):
            o=ns+j*stride
            c,d,s,r,e=x[o:o+n],x[o+n:o+2*n],x[o+2*n:o+3*n],x[o+3*n:o+4*n],x[o+4*n:o+5*n+1]
            if min(c.min(),d.min(),s.min(),r.min()) < -1e-6 or max(c.max(),d.max())>cap+1e-6 or e.min()<1200-1e-6 or e.max()>10800+1e-6: return None
            if np.any((c>1e-6)&((d>1e-6)|(r>1e-6))): return None
            error=max(error,float(abs(x[:n]+r+d-c-s-demand[j]).max()),float(abs(np.diff(e)-eta*c+d/eta).max()),abs(e[0]-e0))
            if terminal: error=max(error,abs(e[-1]-1200))
            bill=past_cash+float(prices[j]@(x[:n] if first else units+1.5*x[n:2*n]-.5*x[2*n:3*n])+5*prices[j]@r)
            error=max(error,abs(bill-x[ci[j]]))
            costs.append(bill); ends.append(e[-1])
        if error>1e-6: return None
        mean=float(w@costs); risk=cvar(costs,w,alpha)
        value=(1-rho)*mean+rho*risk+(0. if terminal else -rate*float(w@(np.asarray(ends)-1200)))
        return dict(objective_rebuilt=value,scenario_invoice_mean=mean,scenario_invoice_cvar90=risk,scenario_costs=costs,physical_error=float(error))
    best=seed.copy(); best_stats=evaluate(best); selected='seed'
    if best_stats is None: raise ValueError('Initial feasible solution failed independent check')
    stages=[]; lower=None; cycle_intervals=0; cycle_charge_removed=0.; relaxation_charge_emergency=0
    h.setOptionValue('solver','ipm')
    relaxation_budget=max(.001,min(limit*.35,limit-(time.perf_counter()-begun)))
    h.setOptionValue('time_limit',relaxation_budget)
    phase_start=time.perf_counter(); h.run()
    sol=h.getSolution(); info=h.getInfo(); relaxation_status=h.modelStatusToString(h.getModelStatus())
    stages.append(dict(phase='relaxation',status=relaxation_status,seconds=time.perf_counter()-phase_start))
    mode_source=None
    if relaxation_status=='Optimal': lower=float(h.getObjectiveValue()+(0. if terminal else rate*1200))
    if sol.value_valid and info.primal_solution_status==hs.SolutionStatus.kSolutionStatusFeasible:
        raw=np.asarray(sol.col_value)
        mode_source,cycle_intervals,cycle_charge_removed=eliminate_cycles(raw,k,n,ns,stride,eta)
        checked=evaluate(mode_source)
        if checked is not None and checked['objective_rebuilt']<=best_stats['objective_rebuilt']+1e-7:
            best=mode_source.copy(); best_stats=checked; selected='cycle_only'
        relaxation_charge_emergency=sum(int(np.count_nonzero((mode_source[ns+j*stride:ns+j*stride+n]>1e-6)&(mode_source[ns+j*stride+3*n:ns+j*stride+4*n]>1e-6))) for j in range(k))
    candidates=[]
    if mode_source is not None and relaxation_charge_emergency:
        base_mode=np.zeros((k,n),bool); conflict=np.zeros((k,n),bool); cflow=np.zeros((k,n)); rflow=np.zeros((k,n))
        for j in range(k):
            o=ns+j*stride
            cflow[j]=mode_source[o:o+n]; rflow[j]=mode_source[o+3*n:o+4*n]
            base_mode[j]=cflow[j]>1e-7; conflict[j]=base_mode[j]&(rflow[j]>1e-7)
        raw_candidates=[
            ('charge_preferred',base_mode),
            ('emergency_preferred',base_mode&~conflict),
            ('balanced',np.where(conflict,cflow>=rflow,base_mode)),
            ('strict_charge',np.where(conflict,cflow>=5*rflow,base_mode)),
            ('lenient_charge',np.where(conflict,5*cflow>=rflow,base_mode)),
        ]
        seen=set()
        for name,mode in raw_candidates:
            key=mode.tobytes()
            if key not in seen: seen.add(key); candidates.append((name,mode.copy()))
    for ci_mode,(name,mode) in enumerate(candidates):
        remaining=limit-(time.perf_counter()-begun)
        if remaining<=0: break
        for j in range(k):
            o=ns+j*stride
            for t in range(n):
                z=o+5*n+1+t; value=float(mode[j,t]); h.changeColBounds(z,value,value)
        h.setOptionValue('time_limit',max(.001,remaining/(len(candidates)-ci_mode)))
        phase_start=time.perf_counter(); h.run()
        sol=h.getSolution(); info=h.getInfo(); status=h.modelStatusToString(h.getModelStatus())
        record=dict(phase=name,status=status,seconds=time.perf_counter()-phase_start,accepted=False)
        if sol.value_valid and info.primal_solution_status==hs.SolutionStatus.kSolutionStatusFeasible:
            x=np.asarray(sol.col_value); checked=evaluate(x)
            if checked is not None:
                record['objective_rebuilt']=checked['objective_rebuilt']
                if checked['objective_rebuilt']<best_stats['objective_rebuilt']-1e-7:
                    best=x.copy(); best_stats=checked; selected=name; record['accepted']=True
        stages.append(record)
    difference=None if lower is None else max(0.,best_stats['objective_rebuilt']-lower)
    gap=None if difference is None else difference/max(1.,abs(best_stats['objective_rebuilt']))
    return np.maximum(best[:n],0),dict(best_stats,status=selected,fallback=False,seed_used=selected=='seed',
        seconds=time.perf_counter()-begun,stages=stages,lower_bound=lower,absolute_gap=difference,gap=gap,
        proven_optimal=difference is not None and difference<=1e-6,integer_pairs=0,
        gap_definition='certified relaxation gap in rebuilt objective units; not a solver MIP gap',
        cycle_intervals=cycle_intervals,cycle_charge_removed=cycle_charge_removed,
        relaxation_charge_emergency=relaxation_charge_emergency,candidate_count=len(candidates))

