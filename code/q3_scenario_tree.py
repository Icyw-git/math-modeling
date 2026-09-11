"""Four-publication scenario tree generated only from completed historical days.

Contracts and scheduled battery actions are shared within each information node.
Instantaneous emergency/spill can respond to each scenario's current demand.
The six-hour scheduled battery action restriction is conservative, not a claim
of exact 10-minute multistage stochastic optimal control.
"""
import time
import numpy as np
import solve_c_q3_refined as ref

CFG=ref.old.CFG

def cluster(features,weights,members,branches=3):
    """Deterministic weighted clustering of information revealed at this stage."""
    ids=np.asarray(members,int); x=features[ids]
    if len(ids)<=1: return [ids.tolist()]
    scale=np.maximum(np.std(x,axis=0),1.)
    z=x/scale; centers=[int(np.argmax(weights[ids]))]
    for _ in range(min(branches,len(ids))-1):
        dist=np.min(((z[:,None,:]-z[centers][None,:,:])**2).sum(axis=2),axis=1)
        if dist.max()<1e-12: break
        centers.append(int(np.argmax(dist)))
    means=z[centers].copy()
    for _ in range(12):
        labels=np.argmin(((z[:,None,:]-means[None,:,:])**2).sum(axis=2),axis=1)
        changed=means.copy()
        for j in range(len(means)):
            mask=labels==j
            if mask.any(): changed[j]=np.average(z[mask],axis=0,weights=weights[ids[mask]])
        if np.max(abs(changed-means))<1e-10: break
        means=changed
    return [ids[labels==j].tolist() for j in range(len(means)) if (labels==j).any()]

def build(data,pred,day,hour,branches=3):
    start=hour*6; history=list(range(max(8,day-28),day))
    if not history: raise ValueError('Tree requires completed historical days')
    load_hat=pred[day,0]; pv_hat=ref.old.pv_forecast_from_issue(data,day,hour)
    le=np.array([data['load'][h]-pred[h,0] for h in history])
    historical_pv=np.array([ref.old.pv_forecast_from_issue(data,h,hour) for h in history])
    ve=data['pv'][history]-historical_pv
    weights=np.ones(len(history))/len(history); correction=np.zeros(144)
    if start:
        x=le[:,start-12:start].mean(axis=1)
        observed=float((data['load'][day,start-12:start]-load_hat[start-12:start]).mean())
        beta=np.clip(x@le/(x@x+len(x)*25.**2),0,1.5)
        correction=beta*observed; le=le-x[:,None]*beta
        likelihood=np.exp(-.5*((x-observed)/max(25.,float(np.std(x))))**2)
        if likelihood.sum()>0: weights=.1*weights+.9*likelihood/likelihood.sum()
    loads=np.maximum(load_hat+correction+le,0); pvs=np.maximum(pv_hat+ve,0)
    nodes=[dict(id=0,parent=None,start=start,end=min(start+36,144),members=list(range(len(history))),mass=1.,children=[])]
    level=[0]
    for boundary in range(start+36,144,36):
        # Only completed block observations and forecasts released at boundary.
        revised=np.array([ref.old.pv_forecast_from_issue(data,h,boundary//6) for h in history])
        revision=revised-historical_pv
        features=np.column_stack([le[:,boundary-36:boundary].mean(axis=1),ve[:,boundary-36:boundary].mean(axis=1),
                                  revision[:,boundary:min(boundary+36,144)].mean(axis=1),revision[:,boundary:].mean(axis=1)])
        next_level=[]
        for parent in level:
            for members in cluster(features,weights,nodes[parent]['members'],branches):
                node=dict(id=len(nodes),parent=parent,start=boundary,end=min(boundary+36,144),members=members,mass=float(weights[members].sum()),children=[])
                nodes[parent]['children'].append(node['id']); nodes.append(node); next_level.append(node['id'])
        level=next_level
    return dict(nodes=nodes,loads=loads,pvs=pvs,weights=weights,history=history,start=start,load_hat=load_hat,pv_hat=pv_hat)

def optimize(tree,price,e0,previous=None,terminal=False,time_limit=15.,allow_future=True):
    """Extensive form with node variables: nonanticipativity by construction."""
    ref.old.b.validate_state(e0,CFG); price=ref.old.b.finite_nonnegative(price,'price')
    loads=ref.old.b.finite_nonnegative(tree['loads'],'loads'); pvs=ref.old.b.finite_nonnegative(tree['pvs'],'pv')
    weights=np.asarray(tree['weights']); nodes=tree['nodes']; start=tree['start']
    if previous is not None:
        previous=ref.old.b.finite_nonnegative(previous,'previous')
        if previous.shape!=(144-start,): raise ValueError('Previous contract shape')
    if loads.shape!=pvs.shape or loads.shape[1]!=144 or not np.isfinite(weights).all() or (weights<=0).any() or abs(weights.sum()-1)>1e-9: raise ValueError('Invalid tree inputs')
    hs=ref.old.b.load_solver(); h=hs.Highs(); begun=time.perf_counter(); demand=loads-pvs
    for key,val in [('output_flag',False),('threads',1),('random_seed',0),('mip_rel_gap',.001)]: h.setOptionValue(key,val)
    costs=[]; lowers=[]; uppers=[]; equations=[]; layout=[]
    def variables(n,lo=0.,hi=np.inf,cost=0.):
        offset=len(costs); costs.extend(np.broadcast_to(cost,(n,)).tolist()); lowers.extend(np.broadcast_to(lo,(n,)).tolist()); uppers.extend(np.broadcast_to(hi,(n,)).tolist()); return np.arange(offset,offset+n)
    def row(lo,hi,ids,coeff): equations.append((lo,hi,list(ids),list(coeff)))
    qmax=np.maximum(demand.max(axis=0),0)+CFG.cap
    if previous is not None: qmax[start:]=np.maximum(qmax[start:],previous)
    for node in nodes:
        s=node['start']; end=node['end']; n=144-s; m=end-s; mass=node['mass']; first=node['parent'] is None and previous is None
        q=variables(n,hi=qmax[s:],cost=price[s:] if first else 0.)
        u=v=None
        if not first:
            u=variables(n,hi=qmax[s:],cost=mass*1.5*price[s:]); v=variables(n,hi=qmax[s:],cost=-mass*.5*price[s:])
            if node['parent'] is None:
                for t in range(n): row(previous[t],previous[t],[q[t],u[t],v[t]],[1,-1,1])
            else:
                parent=layout[node['parent']]; delta=s-nodes[node['parent']]['start']
                for t in range(n):
                    row(0,0,[q[t],parent['q'][delta+t],u[t],v[t]],[1,-1,-1,1])
                    if not allow_future: row(0,0,[u[t],v[t]],[1,1])
        c=variables(m,hi=CFG.cap); d=variables(m,hi=CFG.cap); e=variables(m+1,lo=CFG.emin,hi=CFG.emax); z=variables(m,hi=1.)
        if node['parent'] is None: row(e0,e0,[e[0]],[1])
        else: row(0,0,[e[0],layout[node['parent']]['e'][-1]],[1,-1])
        if not node['children']:
            if terminal: row(CFG.emin,CFG.emin,[e[-1]],[1])
            else: costs[e[-1]]=-mass*ref.old.LAMBDA
        for t in range(m):
            row(0,0,[e[t+1],e[t],c[t],d[t]],[1,-1,-CFG.eta,1/CFG.eta])
            row(-np.inf,0,[c[t],z[t]],[1,-CFG.cap]); row(-np.inf,CFG.cap,[d[t],z[t]],[1,CFG.cap])
        recourse=[]
        for j in node['members']:
            maxr=np.maximum(demand[j,s:end],0); r=variables(m,hi=maxr,cost=weights[j]*5*price[s:end]); spill=variables(m)
            for t in range(m):
                row(demand[j,s+t],demand[j,s+t],[q[t],r[t],d[t],c[t],spill[t]],[1,1,1,-1,-1])
                row(-np.inf,maxr[t],[r[t],z[t]],[1,maxr[t]])
            recourse.append((j,r,spill))
        layout.append(dict(q=q,u=u,v=v,c=c,d=d,e=e,z=z,recourse=recourse))
    size=len(costs); h.addVars(size,np.asarray(lowers),np.asarray(uppers)); h.changeColsCost(size,np.arange(size,dtype=np.int32),np.asarray(costs))
    starts=[0]; indices=[]; values=[]; lb=[]; ub=[]
    for lo,hi,ids,coef in equations: lb.append(lo); ub.append(hi); indices.extend(ids); values.extend(coef); starts.append(len(indices))
    h.addRows(len(equations),np.asarray(lb),np.asarray(ub),len(indices),np.asarray(starts,np.int32),np.asarray(indices,np.int32),np.asarray(values,float))
    enforced=set(); status='not run'
    for iteration in range(7):
        h.setOptionValue('time_limit',max(.001,time_limit-(time.perf_counter()-begun))); h.run(); sol=h.getSolution(); info=h.getInfo(); status=h.modelStatusToString(h.getModelStatus())
        if not sol.value_valid or info.primal_solution_status!=hs.SolutionStatus.kSolutionStatusFeasible: raise RuntimeError('Tree solve: '+status)
        x=np.asarray(sol.col_value); bad=set()
        for a in layout:
            emergency=np.max(np.array([x[r] for _,r,_ in a['recourse']]),axis=0)
            bad.update(a['z'][(x[a['c']]>1e-6)&((x[a['d']]>1e-6)|(emergency>1e-6))].tolist())
        if not bad: break
        new=bad-enforced
        if not new or time.perf_counter()-begun>=time_limit: raise RuntimeError('Tree mutual exclusion unresolved')
        if iteration>=4: new=set(np.concatenate([a['z'] for a in layout]).tolist())-enforced
        for i in sorted(new): h.changeColIntegrality(int(i),hs.HighsVarType.kInteger)
        enforced.update(new)
    else: raise RuntimeError('Tree refinement exhausted')
    # Check every bound and row, including all parent contracts and SOC links.
    bound_error=max(float(np.max(np.asarray(lowers)-x)),float(np.max(x-np.asarray(uppers))),0.)
    row_error=0.
    for lo,hi,ids,coef in equations:
        val=float(x[ids]@coef); row_error=max(row_error,lo-val,val-hi)
    if not np.isfinite(x).all() or max(bound_error,row_error)>1e-6: raise RuntimeError('Tree incumbent check failed')
    solution=[]
    for node,a in zip(nodes,layout):
        solution.append(dict(id=node['id'],q=np.maximum(x[a['q']],0),c=x[a['c']],d=x[a['d']],e=x[a['e']]))
    meta=dict(status=status,gap=float(info.mip_gap) if np.isfinite(info.mip_gap) else 0.,objective=float(info.objective_function_value),seconds=time.perf_counter()-begun,nodes=len(nodes),leaves=sum(not n['children'] for n in nodes),variables=size,integer_pairs=len(enforced),row_error=row_error,bound_error=bound_error,fallback=False)
    return solution,meta

def control_value(tree,solution,price,terminal=False,grid_size=97):
    """Backward node-conditioned DP under the optimized future contracts.

    At the next publication, child continuation values are probability mixed.
    Belief is fixed inside a six-hour node (same approximation as old controller).
    """
    grid=np.linspace(CFG.emin,CFG.emax,grid_size); delta=grid[None,:]-grid[:,None]
    c=np.maximum(delta,0)/CFG.eta; d=np.maximum(-delta,0)*CFG.eta; physical=(c<=CFG.cap+1e-8)&(d<=CFG.cap+1e-8)
    tables={}
    for node in reversed(tree['nodes']):
        s=node['start']; end=node['end']; m=end-s; vals=np.empty((m+1,grid_size))
        if node['children']: vals[m]=sum(tree['nodes'][j]['mass']/node['mass']*tables[j][0] for j in node['children'])
        elif terminal: vals[m]=1e12; vals[m,0]=0.
        else: vals[m]=-ref.old.LAMBDA*(grid-CFG.emin)
        q=solution[node['id']]['q']; ids=np.asarray(node['members']); weights=tree['weights'][ids]/node['mass']
        for t in range(m-1,-1,-1):
            expected=np.zeros(grid_size)
            for j,w in zip(ids,weights):
                balance=q[t]+tree['pvs'][j,s+t]-tree['loads'][j,s+t]
                allowed=physical&((c<=1e-8)|(c<=balance+1e-8)); r=np.maximum(-balance+c-d,0)
                expected+=w*np.where(allowed,5*price[s+t]*r+vals[t+1][None,:],1e12).min(axis=1)
            vals[t]=expected
        tables[node['id']]=vals
    return grid,tables[0]
