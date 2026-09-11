"""Same joint policy with independently cached, past-only terminal curves."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import run_c_q3_joint as runner

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--cache',type=Path,required=True); a=ap.parse_args()
    manifest=json.loads((a.cache/'manifest.json').read_text())
    for path,digest in manifest['hashes'].items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    used={}
    def cached(data,pred,day,time_limit=5):
        path=a.cache/f'{day:03}.json'; begun=time.perf_counter()
        while True:
            try: row=json.loads(path.read_text()); break
            except (FileNotFoundError,json.JSONDecodeError):
                if time.perf_counter()-begun>600: raise RuntimeError(f'Terminal cache unavailable: {day}')
                time.sleep(.2)
        assert row['day']==day and all(h<day for h in row['meta'].get('history',[]))
        used[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        return np.asarray(row['values']),row['meta']
    runner.joint.terminal_value=cached
    runner.run(runner.ref.old.load_data(),a.output)
    cfg=json.loads((a.output/'config.json').read_text()); cfg['terminal_cache_hashes']=used; cfg['terminal_cache_manifest']=manifest
    cfg['hashes'][str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (a.output/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
