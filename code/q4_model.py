"""Q4 causal joint forecasts, positive price scenarios and mean-CVaR contracts.

No input workbook is modified. Q2 mode never opens attachment 3.
Scenario recourse is two-stage; real execution uses a common expectation value.
"""
from pathlib import Path
import hashlib
import sys
import time
import numpy as np
import pandas as pd
import solve_c_q2_baseline as base
import solve_c_q2_enhanced as en
from audit_c_attachments import matrix, clock_minutes

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'CUMCM2026Problems'/'C题'/'附件'
CFG = base.Config()
N = 144


def cvar(cost, weights=None, alpha=.9):
    x = np.asarray(cost, float)
    w = np.ones(len(x))/len(x) if weights is None else np.asarray(weights, float)
    if x.ndim != 1 or not len(x) or w.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w < 0) or abs(w.sum()-1)>1e-9 or not 0<=alpha<1:
        raise ValueError('Invalid CVaR inputs')
    left=1-alpha; total=0.
    for i in np.argsort(-x,kind='stable'):
        take=min(left,w[i]); total+=take*x[i]; left-=take
        if left<=1e-14: break
    return float(total/(1-alpha))


def load_data(mode):
    if mode not in ('q2','q3'): raise ValueError(mode)
    dates,l=matrix(SOURCE/'附件2.xlsx','小区负载')
    _,v=matrix(SOURCE/'附件2.xlsx','光伏发电实际功率')
    _,p=matrix(SOURCE/'附件4.xlsx')
    for name,a in [('load',l),('pv',v),('price',p)]: base.finite_nonnegative(a,name)
    if np.any(p<=0): raise ValueError('Price must be positive')
    files=[SOURCE/'附件2.xlsx',SOURCE/'附件4.xlsx']
    data=dict(dates=pd.to_datetime(dates),load=l/6,pv=v/6,price=p)
    if mode=='q3':
        f=pd.read_excel(SOURCE/'附件3.xlsx')
        ds=pd.to_datetime(f.iloc[:,0].replace('',np.nan).ffill())
        hours=[clock_minutes(x)//60 for x in f.iloc[:,1]]
        a=f.iloc[:,2:].to_numpy(float)/6
        base.finite_nonnegative(a,'forecast')
        if a.shape!=(1460,24) or hours!=[0,6,12,18]*365: raise ValueError('Forecast layout')
        data['forecast']={(pd.Timestamp(d).normalize(),h):row.copy() for d,h,row in zip(ds,hours,a)}
        if len(data['forecast'])!=1460: raise ValueError('Duplicate forecast issue')
        files.append(SOURCE/'附件3.xlsx')
    data['hashes']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    return data


def archive(data):
    l,v,_=en.forecast_archive(data['load'],data['pv'])
    p=np.ones_like(data['price'])  # day zero untrained, excluded from residuals
    for day in range(1,len(p)): p[day]=data['price'][day-(7 if day>=7 else 1)]
    return dict(load=l[:,0],pv=v[:,0],price=p)


def pv_issue(data,day,hour):
    f=data['forecast'][(pd.Timestamp(data['dates'][day]).normalize(),hour)]
    start=hour*6
    anchor=data['pv'][day,start-1] if start else (data['pv'][day-1,-1] if day else f[0])
    # Values before publication are not used by this forecast or residual pool.
    return np.interp(np.maximum((np.arange(N)+1)/6-hour,0),np.arange(25),np.r_[anchor,f])


def price_update(actual_prefix,point,start):
    if len(actual_prefix)!=start: raise ValueError('Expected completed prefix')
    shift=0. if start==0 else float(np.log(np.asarray(actual_prefix)[-12:]/point[max(0,start-12):start]).mean())
    corrected=point.copy()
    corrected[start:]=point[start:]*np.exp(shift*np.exp(-np.arange(N-start)/12))
    return corrected


def scenarios(data,arc,day,start,mode,known=False):
    if day<1 or start not in (0,36,72,108): raise ValueError('Invalid information boundary')
    hist=list(range(max(1,day-28),day))
    lh=arc['load'][day].copy()
    vh=arc['pv'][day].copy() if mode=='q2' else pv_issue(data,day,start//6)
    ph=price_update(data['price'][day,:start],arc['price'][day],start)
    le=np.asarray([data['load'][h]-arc['load'][h] for h in hist])
    ve=np.asarray([data['pv'][h]-(arc['pv'][h] if mode=='q2' else pv_issue(data,h,start//6)) for h in hist])
    if start:
        x=np.asarray([e[max(0,start-12):start].mean() for e in le])
        obs=float((data['load'][day,max(0,start-12):start]-lh[max(0,start-12):start]).mean())
        beta=np.clip(x@le/(x@x+len(x)*25**2),0,1.5)
        lh=np.maximum(lh+beta*obs,0); le=le-x[:,None]*beta
    lp=[]
    for h in hist:
        hp=price_update(data['price'][h,:start],arc['price'][h],start)
        lp.append(np.log(data['price'][h]/hp))
    prices=ph[None,:]*np.exp(np.asarray(lp))
    if known:
        ph=data['price'][day].copy(); prices=np.repeat(ph[None,:],len(hist),axis=0)
    loads=np.maximum(lh[None,:]+le,0); pvs=np.maximum(vh[None,:]+ve,0)
    if not np.isfinite(prices).all() or np.any(prices<=0): raise ValueError('Invalid reconstructed price')
    w=np.ones(len(hist))/len(hist)
    return loads[:,start:],pvs[:,start:],prices[:,start:],w,dict(history=hist,scenario_count=len(hist),effective_samples=float(1/(w@w)),load_point=lh[start:].tolist(),pv_point=vh[start:].tolist(),price_point=ph[start:].tolist())


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
    enforced=set(); reason='No feasible incumbent'; status='not run'
    for iteration in range(7):
        h.setOptionValue('solver','ipm' if not enforced else 'choose')
        h.setOptionValue('time_limit',max(.001,limit-(time.perf_counter()-begun))); h.run()
        sol=h.getSolution(); info=h.getInfo(); status=h.modelStatusToString(h.getModelStatus())
        if not sol.value_valid or info.primal_solution_status!=hs.SolutionStatus.kSolutionStatusFeasible: reason=status; break
        x=np.asarray(sol.col_value); bad=set(); residual=0.; costs=[]; ends=[]
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
            return np.maximum(x[:n],0),dict(status=status,fallback=False,gap=float(info.mip_gap) if enforced and np.isfinite(info.mip_gap) else 0.,seconds=time.perf_counter()-begun,scenario_invoice_mean=mean,scenario_invoice_cvar90=risk,objective_rebuilt=(1-rho)*mean+rho*risk+continuation,scenario_costs=costs,physical_error=residual,integer_pairs=len(enforced))
        new=bad-enforced
        if not new or time.perf_counter()-begun>=limit: reason='Mutual exclusion unresolved'; break
        if iteration>=4: new={ns+j*stride+5*n+1+t for j in range(k) for t in range(n)}-enforced
        for z in sorted(new): h.changeColIntegrality(z,hs.HighsVarType.kInteger)
        enforced.update(new)
    return prev.copy() if not first else np.maximum(np.asarray(lh)-vh,0),dict(status=reason,fallback=True,gap=None,seconds=time.perf_counter()-begun,scenario_invoice_mean=None,scenario_invoice_cvar90=None)


def values(q,loads,pvs,prices,w,rate,terminal=False,grid_size=97):
    """Expected marginal Bellman; current demand AND price may be observed."""
    grid=np.linspace(1200,10800,grid_size); n=len(q)
    v=np.empty((n+1,grid_size)); v[n]=-rate*(grid-1200)
    if terminal: v[n]=1e15; v[n,0]=0.
    delta=grid[None,:]-grid[:,None]; c=np.maximum(delta,0)/.9; d=np.maximum(-delta,0)*.9
    physical=(c<=CFG.cap+1e-8)&(d<=CFG.cap+1e-8)
    for t in range(n-1,-1,-1):
        b=q[t]+pvs[:,t]-loads[:,t]
        allowed=physical[None,:,:]&((c[None,:,:]<=1e-8)|(c[None,:,:]<=b[:,None,None]+1e-8))
        r=np.maximum(-b[:,None,None]+c[None,:,:]-d[None,:,:],0)
        cost=5*prices[:,t,None,None]*r+v[t+1][None,None,:]
        v[t]=w@np.where(allowed,cost,1e15).min(axis=2)
    return grid,v
