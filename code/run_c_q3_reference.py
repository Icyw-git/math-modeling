"""Legacy budget comparison with explicit contract numerical validation."""
import hashlib
import json
import numpy as np
import solve_c_q3_refined as m
original=m.old.contract_milp
def checked(*args,**kwargs):
    q,meta=original(*args,**kwargs)
    minimum=float(np.min(q)); meta['raw_min_contract']=minimum
    if not np.isfinite(q).all() or minimum < -1e-6:
        previous=args[6] if len(args)>6 else kwargs.get('previous_q')
        q=np.maximum(np.asarray(args[1])-args[2],0) if previous is None else previous.copy()
        meta.update(fallback=True,status='Rejected nonfinite/negative incumbent: '+meta['status'])
    else: q=np.maximum(q,0)
    return q,meta

if __name__=='__main__':
    m.old.contract_milp=checked
    out=m.ROOT/'results/q3_refined/reference'
    m.run(m.old.load_data(),out,mode='reference')
    config=json.loads((out/'config.json').read_text())
    config['hashes'][str(m.Path(__file__).resolve())]=hashlib.sha256(m.Path(__file__).read_bytes()).hexdigest()
    config['reference_guard']='Reject nonfinite or below -1e-6; clip only numerical negatives'
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
