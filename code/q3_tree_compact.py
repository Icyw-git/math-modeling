"""Seven-path tree: same representative trajectories as the adaptive baseline."""
import numpy as np
import q3_scenario_tree as dense
ref=dense.ref
optimize=dense.optimize


def build(data,pred,day,hour,branches=3):
    # Use the exact baseline representative paths and masses, not 28 new paths.
    lh=pred[day,0]; pv=ref.old.pv_forecast_from_issue(data,day,hour)
    paths,weights,diagnostics=baseline_paths(data,pred,day,lh,pv,hour,7,True)
    history=diagnostics['history']; loads=np.array([p[0] for p in paths]); pvs=np.array([p[1] for p in paths]); start=hour*6
    historical_pv=np.array([ref.old.pv_forecast_from_issue(data,h,hour) for h in history])
    # The realized historical observations, not unreleased current-day data.
    le=np.array([data['load'][h]-pred[h,0] for h in history]); ve=data['pv'][history]-historical_pv
    nodes=[dict(id=0,parent=None,start=start,end=start+36,members=list(range(len(history))),mass=1.,children=[])]
    level=[0]
    for boundary in range(start+36,144,36):
        revision=np.array([ref.old.pv_forecast_from_issue(data,h,boundary//6) for h in history])-historical_pv
        features=np.column_stack([le[:,boundary-36:boundary].mean(axis=1),ve[:,boundary-36:boundary].mean(axis=1),revision[:,boundary:boundary+36].mean(axis=1),revision[:,boundary:].mean(axis=1)])
        following=[]
        for parent in level:
            for members in dense.cluster(features,weights,nodes[parent]['members'],branches):
                node=dict(id=len(nodes),parent=parent,start=boundary,end=boundary+36,members=members,mass=float(weights[members].sum()),children=[])
                nodes[parent]['children'].append(node['id']); nodes.append(node); following.append(node['id'])
        level=following
    return dict(nodes=nodes,loads=loads,pvs=pvs,weights=weights,history=history,start=start,load_hat=lh,pv_hat=pv)


baseline_paths=ref.conditioned_paths


def control_value(tree,solution,price,terminal=False,grid_size=97):
    """Exact same grid recurrence as dense, evaluating only reachable neighbors."""
    cfg=ref.old.CFG; grid=np.linspace(cfg.emin,cfg.emax,grid_size)
    reach=int(np.ceil(cfg.cap/cfg.eta/(grid[1]-grid[0])))
    indices=np.arange(grid_size)[:,None]+np.arange(-reach,reach+1)[None,:]
    valid=(indices>=0)&(indices<grid_size); indices=np.clip(indices,0,grid_size-1)
    delta=grid[indices]-grid[:,None]; c=np.maximum(delta,0)/cfg.eta; d=np.maximum(-delta,0)*cfg.eta
    physical=valid&(c<=cfg.cap+1e-8)&(d<=cfg.cap+1e-8); tables={}
    for node in reversed(tree['nodes']):
        s=node['start']; m=node['end']-s; vals=np.empty((m+1,grid_size))
        if node['children']: vals[m]=sum(tree['nodes'][j]['mass']/node['mass']*tables[j][0] for j in node['children'])
        elif terminal: vals[m]=1e12; vals[m,0]=0.
        else: vals[m]=-ref.old.LAMBDA*(grid-cfg.emin)
        ids=np.asarray(node['members']); weights=tree['weights'][ids]/node['mass']; q=solution[node['id']]['q']
        for t in range(m-1,-1,-1):
            expected=np.zeros(grid_size)
            for j,w in zip(ids,weights):
                b=q[t]+tree['pvs'][j,s+t]-tree['loads'][j,s+t]
                allowed=physical&((c<=1e-8)|(c<=b+1e-8)); r=np.maximum(-b+c-d,0)
                expected+=w*np.where(allowed,5*price[s+t]*r+vals[t+1][indices],1e12).min(axis=1)
            vals[t]=expected
        tables[node['id']]=vals
    return grid,tables[0]
