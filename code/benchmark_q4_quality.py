"""Paired frozen-window benchmark, not a new annual policy backtest."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json
import hashlib
from pathlib import Path
import numpy as np
import q4_model as old
import q4_solver_quality as new

def main():
    out=old.ROOT/'results/q4_quality'; out.mkdir(exist_ok=True)
    results=[]
    for variant in ('q2_neutral','q2_cvar','q3_neutral','q3_cvar'):
        mode=variant[:2]; rho=.2 if variant.endswith('cvar') else 0.
        data=old.load_data(mode); arc=old.archive(data)
        pool=[]
        for f in sorted((old.ROOT/'results/q4_cvar'/variant).glob('day_*.json')):
            day=json.loads(f.read_text(encoding='utf8'))
            for w in day['windows']:
                if w['status']!='contract_frozen': pool.append((f,day,w))
        selected=[('fallback',next(x for x in pool if x[2]['fallback'])),
                  ('high_gap',max((x for x in pool if not x[2]['fallback']),key=lambda x:x[2].get('gap') or 0)),
                  ('ordinary',next(x for x in pool if x[2]['status']=='Optimal' and (mode=='q2' or x[2]['start']>0)))]
        for category,(f,day,window) in selected:
            t=window['start']; di=day['day']; ledger=day['ledger']
            l,v,p,w,meta=old.scenarios(data,arc,di,t,mode)
            assert hashlib.sha256(p.tobytes()).hexdigest()==window['price_scenario_sha256']
            q=np.array([r['q0'] for r in ledger]); units=q.copy()
            for change in day['changes']:
                if change['start']>=t: break
                st=change['start']; q[st:]=change['new']
                units[st:]+=1.5*np.array(change['up'])-.5*np.array(change['down'])
            cash=sum(r['price']*(r['charged_units']+5*r['emergency_kwh']) for r in ledger[:t])
            args=(l,v,p,w,ledger[t]['soc_start_kwh'],meta['load_point'],meta['pv_point'])
            kwargs=dict(previous=q[t:] if t else None,charged=units[t:] if t else None,past_cash=cash,rho=rho,terminal=di==364,rate=float(arc['price'][di].min()/.9),limit=30.)
            item=dict(variant=variant,category=category,date=day['daily']['date'],start=t,input_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            for label,solver in (('old',old.contract),('new',new.contract)):
                _,stats=solver(*args,**kwargs); item[label]=stats
                print(variant,category,label,stats['status'],'fallback=',stats['fallback'],'seconds=',round(stats['seconds'],2),flush=True)
            if not item['old']['fallback'] and not item['new']['fallback']:
                item['objective_difference']=item['new']['objective_rebuilt']-item['old']['objective_rebuilt']
            results.append(item)
            (out/'benchmark.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf8')
    lines=['# 第四问求解器小规模对照','',
           '固定原窗口SOC、历史场景、合同和已实现费用；两求解器同为30秒预算。每策略选择首个回退、最大已报告gap、首个普通最优窗口，属于有意挑选的诊断样本，不代表全年分布。未重跑全年，不能据此宣称全年节省。',
           '', '| 策略 | 类别 | 日期/区间 | 旧回退 | 新回退 | 旧耗时/s | 新耗时/s | 新减旧目标/元 |',
           '| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
    for r in results:
        lines.append(f"| {r['variant']} | {r['category']} | {r['date']}/{r['start']} | {r['old']['fallback']} | {r['new']['fallback']} | {r['old']['seconds']:.2f} | {r['new']['seconds']:.2f} | {r.get('objective_difference','不可比较')} |")
    lines+=['','目标含终端估值及（适用时）风险项，不是实际账单。旧版失败窗口没有合格场景目标，不能拿预测缺口回退的账单直接比较优化目标。新模型完整设置模式整数变量，提交经过全部线性约束和边界校验的可行初解；无求解器候选时保留初解，并标记seed_used。旧年度代码及检查点未修改。']
    lines+=['','## 结论与采用范围','',
            '12个定点样本中旧版回退4次、完整整数新版0次；但8个双方都有合格目标的样本，新版目标全部更高。完整整数模型不宜直接替代原算法。统计属于诊断样本，不是随机样本或全年性能估计。',
            '', '已提供q4_solver_adaptive.contract选择性补救接口：旧版合格解原样保留，仅在旧版失败后追加完整整数求解；补救也失败则保留安全回退。非最优LP不再标注gap=0。接口路由经过单元测试，但未作新全年回测；回退样本得到完整可行解不等于实际电费更低。',
            '', '完整整数求解器通过原有10项测试及2项新增测试（极短预算、年末可行初解）；选择性接口通过10项原有测试及3项路由测试。没有改变28条场景、风险权重、SOC物理约束和原年度结果。',
            '', '30秒是求解器设置而非严格墙钟限制，两版均出现超时。新版本总墙钟耗时约为旧版2倍，不宣称提速。原始每窗统计见results/q4_quality/benchmark.json。dual_bound为求解器原目标口径，比较objective_rebuilt时应补上终端常数偏移（非年末为rate*1200），不能直接相减。']
    (out/'source_hashes.json').write_text(json.dumps({str(p.relative_to(old.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(old.__file__),Path(new.__file__))},indent=2),encoding='utf8')
    (old.ROOT/'reports/Q4_SOLVER_QUALITY.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__': main()
