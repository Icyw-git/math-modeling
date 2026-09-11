"""Causal feedback rollouts and low-dimensional joint contract/policy search."""
import numpy as np
import q3_tree_compact as compact
ref=compact.ref
CFG=ref.old.CFG
GRID=np.linspace(1200,10800,9)


def forecast(pool,weights,observed,start,t,members=None):
    """All observations supplied are at or before t; never access future actuals."""
    ids=np.arange(len(weights)) if members is None else np.asarray(members)
    prior=weights[ids]/weights[ids].sum(); left=max(start,t-5)
    error=np.mean(np.asarray(observed)[left:t+1][None,:]-pool[ids,left:t+1],axis=1)
    width=max(25.,float(np.std(error))); likelihood=np.exp(-.5*(error/width)**2)
    posterior=prior*likelihood
    if posterior.sum()>0: prior=.1*prior+.9*posterior/posterior.sum()
    mean=prior@pool[ids]; correction=float(prior@error)*np.exp(-np.maximum(np.arange(144)-t,0)/12.)
    return mean,correction


def feedback(q,load,pv,e,price,t,expected_net,alpha,terminal_curve,terminal=False):
    """Same controller for simulated and real execution; returns physical actions."""
    e=np.asarray(e,float); b=np.asarray(q)+np.asarray(pv)-np.asarray(load)
    future=np.asarray(expected_net)[...,t+1:]
    expensive=price[t+1:]>price[t]+1e-10
    reserve=alpha*np.maximum((np.maximum(future,0)*expensive).sum(axis=-1)/.9-.9*np.maximum(-future,0).sum(axis=-1),0)
    # Opportunity cost of retaining energy for tomorrow, measured in internal kWh.
    target=GRID[np.argmin(5*price[t]*.9*(GRID-1200)+terminal_curve)]-1200
    reserve=np.clip(np.maximum(reserve,target),0,9600)
    c=np.minimum(np.minimum(np.maximum(b,0),CFG.cap),np.maximum(10800-e,0)/.9)
    d=np.minimum(np.minimum(np.maximum(-b,0),CFG.cap),np.maximum(e-1200-reserve,0)*.9)
    end=e+.9*c-d/.9
    if terminal: end=np.minimum(end,1200+(143-t)*CFG.cap/.9)
    c=np.maximum(end-e,0)/.9; d=np.maximum(e-end,0)*.9
    r=np.maximum(-b+c-d,0); s=np.maximum(b-c+d,0)
    return dict(c=c,d=d,r=r,s=s,end=end)


def prepared(tree):
    """Forecast features shared by all candidates; only revealed node membership."""
    pool=tree['loads']-tree['pvs']; count=len(pool); start=tree['start']
    means=np.zeros((count,144-start,144)); corrections=np.zeros_like(means); node_at=np.zeros((count,144-start),int)
    for node in tree['nodes']:
        for j in node['members']:
            for t in range(node['start'],node['end']):
                node_at[j,t-start]=node['id']
                means[j,t-start],corrections[j,t-start]=forecast(pool,tree['weights'],pool[j,:t+1],start,t,node['members'])
    return means,corrections,node_at


def rollout(tree,contracts,e0,price,curve,alpha,gain,cache,previous=None,terminal=False,return_details=False):
    means,corrections,node_at=cache; start=tree['start']; n=len(tree['weights']); e=np.full(n,e0,dtype=float); cost=np.zeros(n)
    initial=contracts[0]
    if previous is None: cost+=float(price[start:]@initial)
    else: cost+=float(price[start:]@(1.5*np.maximum(initial-previous,0)-.5*np.maximum(previous-initial,0)))
    for node in tree['nodes'][1:]:
        s=node['start']; parent=tree['nodes'][node['parent']]; diff=contracts[node['id']]-contracts[parent['id']][s-parent['start']:]
        cost[node['members']]+=float(price[s:]@(1.5*np.maximum(diff,0)-.5*np.maximum(-diff,0)))
    emergency=np.zeros(n); unused=np.zeros(n)
    for t in range(start,144):
        ids=node_at[:,t-start]
        q=np.array([contracts[i][t-tree['nodes'][i]['start']] for i in ids])
        forecast_net=means[:,t-start]+gain*corrections[:,t-start]
        future_q=np.array([np.pad(contracts[i],(tree['nodes'][i]['start'],0)) for i in ids])
        a=feedback(q,tree['loads'][:,t],tree['pvs'][:,t],e,price,t,forecast_net-future_q,alpha,curve,terminal)
        cost+=5*price[t]*a['r']; emergency+=a['r']; unused+=a['s']; e=a['end']
    value=float(tree['weights']@(cost+np.interp(e,GRID,curve)))
    if return_details: return value,dict(cost=cost,end=e,emergency=emergency,unused=unused)
    return value


def search(tree,tree_contracts,baseline_q,e0,price,curve,previous=None,terminal=False):
    cache=prepared(tree); logs=[]; best=None
    def evaluate(contracts,alpha,gain,label):
        nonlocal best
        value=rollout(tree,contracts,e0,price,curve,alpha,gain,cache,previous,terminal)
        logs.append(dict(candidate=label,alpha=alpha,gain=gain,value=value))
        if best is None or value<best[0]-1e-7: best=(value,[q.copy() for q in contracts],alpha,gain,label)
    # Common future contract perturbation preserves a feasible decision at each node.
    for blend in (0.,.5,1.):
        difference=(baseline_q-tree_contracts[0])*(1-blend)
        contracts=[np.maximum(q+difference[node['start']-tree['start']:],0) for q,node in zip(tree_contracts,tree['nodes'])]
        for alpha in (0.,.5,1.):
            for gain in (0.,.75): evaluate(contracts,alpha,gain,f'seed:{blend}')
    for step in (50.,20.):
        for block in range(tree['start']//36,4):
            current=best
            for sign in (-1,1):
                contracts=[q.copy() for q in current[1]]
                for q,node in zip(contracts,tree['nodes']):
                    left=max(36*block,node['start']); right=36*(block+1)
                    if left<right: q[left-node['start']:right-node['start']]=np.maximum(q[left-node['start']:right-node['start']]+sign*step,0)
                evaluate(contracts,current[2],current[3],f'block:{block}:{sign*step}')
    # Refit control after contract moves: all changes judged by total closed-loop cost.
    contracts=best[1]
    for alpha in (0.,.25,.5,.75,1.):
        for gain in (0.,.75): evaluate(contracts,alpha,gain,'policy_refine')
    return best,logs


def terminal_value(data,pred,day,time_limit=5.):
    """Next-day costs using past-only persistence PV, no future forecast releases."""
    if day==364: return np.zeros(9),dict(annual_terminal=True,solves=[])
    history=np.arange(max(8,day-28),day); lh=data['load'][day-6]; vh=data['pv'][day-1]
    le=data['load'][history]-pred[history,0]; ve=data['pv'][history]-data['pv'][history-1]
    order=np.argsort((le-ve).mean(axis=1),kind='stable'); groups=np.array_split(order,7)
    chosen=[int(g[len(g)//2]) for g in groups]; weights=np.array([len(g)/len(history) for g in groups])
    loads=np.maximum(lh+le[chosen],0); pvs=np.maximum(vh+ve[chosen],0)
    tr=dict(nodes=[dict(id=0,parent=None,start=0,end=144,members=list(range(7)),mass=1.,children=[])],loads=loads,pvs=pvs,weights=weights,start=0)
    cache=prepared(tr); price=data['fixed_price']; tail=-ref.old.LAMBDA*(GRID-1200); curve=[]; solves=[]
    for e in GRID:
        q,meta=ref.opt.solve(price,lh,vh,list(zip(loads,pvs)),weights,float(e),terminal_target=1200 if day==363 else None,time_limit=time_limit,terminal_rate=ref.old.LAMBDA)
        scores=[rollout(tr,[q],e,price,tail,alpha,.75,cache,terminal=day==363) for alpha in (0.,.5,1.)]
        curve.append(min(scores)); solves.append(meta)
    curve=np.array(curve); curve-=curve[0]
    return curve,dict(history=history[chosen].tolist(),solves=solves,approximation='fixed virtual next-day contract, same feedback family; no future adjustments')
