"""Independent Q3 experiment; reuse the checked ledger/settlement runner.

Adapters are installed only in this process and always restored. No old source
or result is edited. Future node contracts are virtual until reoptimization.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import q3_scenario_tree as tree
ref=tree.ref


def run(data,out,end_day=365,grid_size=97,time_limit=15.,branches=3,allow_future=True):
    out=Path(out)
    if (out/'summary.json').exists(): raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    original_paths=ref.conditioned_paths; original_solve=ref.opt.solve; original_value=ref.old.value_table
    context={}
    with (out/'trees.jsonl').open('w',encoding='utf8') as archive:
        def paths(data,pred,day,lh,pv,hour,count=7,adaptive=True):
            tr=tree.build(data,pred,day,hour,branches)
            context.update(tree=tr,day=day,hour=hour,solution=None)
            return list(zip(tr['loads'],tr['pvs'])),tr['weights'],dict(history=tr['history'])

        def solve(price,lh,pv,scenarios,weights,e0,previous_q=None,terminal_target=None,time_limit=time_limit,**kwargs):
            begun=time.perf_counter(); tr=context['tree']
            try:
                solution,meta=tree.optimize(tr,data['fixed_price'],e0,previous_q,terminal_target is not None,time_limit,allow_future)
                context['solution']=solution
                q=solution[0]['q'].copy()
            except RuntimeError as exc:
                q,meta=original_solve(price,lh,pv,scenarios,weights,e0,previous_q,terminal_target=terminal_target,time_limit=time_limit,terminal_rate=ref.old.LAMBDA)
                meta.update(fallback=True,tree_failure=str(exc),seconds=time.perf_counter()-begun)
            archive.write(json.dumps(dict(day=context['day'],hour=context['hour'],history=tr['history'],weights=tr['weights'].tolist(),nodes=tr['nodes'],solution=None if context['solution'] is None else [{k:v.tolist() if hasattr(v,'tolist') else v for k,v in node.items()} for node in context['solution']],meta=meta))+'\n')
            archive.flush()
            return q,meta

        def value(q,price,scenarios,weights,terminal=False,grid_size=97):
            if context['solution'] is None: return original_value(q,price,scenarios,weights,terminal,grid_size)
            return tree.control_value(context['tree'],context['solution'],data['fixed_price'],terminal,grid_size)

        ref.conditioned_paths=paths; ref.opt.solve=solve; ref.old.value_table=value
        try:
            ref.run(data,out,'adaptive',end_day,grid_size,time_limit)
        finally:
            ref.conditioned_paths=original_paths; ref.opt.solve=original_solve; ref.old.value_table=original_value
    cfg=json.loads((out/'config.json').read_text())
    cfg.update(mode='four_stage_tree',branches=branches,allow_future_adjustments=allow_future,scenario_count='all preceding 28 completed days (23 at first day)',eta_charge=.9,eta_discharge=.9,terminal_rate=ref.old.LAMBDA,information_hours=[0,6,12,18],battery_planning='node-shared scheduled actions; conservative six-hour approximation',control='node-conditioned Bellman; actual current interval response',annual_initial_soc=1200,annual_terminal_soc=1200,january='shared previous warmup ending at 1200 kWh')
    for path in [Path(__file__),Path(tree.__file__)]: cfg['hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    from verify_c_q3_refined import verify
    verify(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--end-day',type=int,default=365); ap.add_argument('--time-limit',type=float,default=15.); ap.add_argument('--branches',type=int,default=3); ap.add_argument('--grid-size',type=int,default=97); ap.add_argument('--freeze-future',action='store_true'); a=ap.parse_args()
    run(ref.old.load_data(),a.output,a.end_day,a.grid_size,a.time_limit,a.branches,not a.freeze_future)
