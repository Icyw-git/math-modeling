"""Formal seven-path experiment; dense 28-path pilot remains reproducible."""
import argparse
import hashlib
import json
from pathlib import Path
import solve_c_q3_tree as runner
import q3_tree_compact as compact


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--end-day',type=int,default=365); ap.add_argument('--time-limit',type=float,default=15.); ap.add_argument('--freeze-future',action='store_true'); a=ap.parse_args()
    runner.tree=compact
    runner.run(compact.ref.old.load_data(),a.output,a.end_day,97,a.time_limit,3,not a.freeze_future)
    cfg=json.loads((a.output/'config.json').read_text()); cfg['scenario_count']=7
    for p in [Path(__file__),Path(compact.dense.__file__)]: cfg['hashes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (a.output/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
