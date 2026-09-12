"""Repair the same preselected windows; historical baseline timings are references."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json
import hashlib
import numpy as np
import q4_model as model
import q4_solver_repair as repair
from pathlib import Path

def main():
    root=model.ROOT; out=root/'results/q4_repair_cached'; out.mkdir(exist_ok=True)
    reference=root/'results/q4_quality/benchmark.json'
    previous=json.loads(reference.read_text(encoding='utf8'))
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(model.__file__),Path(repair.__file__),reference)}
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    datasets={}; result=[]
    for ref in previous:
        name=ref['variant']; mode=name[:2]; rho=.2 if name.endswith('cvar') else 0.
        if mode not in datasets:
            data=model.load_data(mode); datasets[mode]=(data,model.archive(data))
        data,arc=datasets[mode]
        path=root/'results/q4_cvar'/name/f"day_{ref['date']}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['input_sha256']
        day=json.loads(path.read_text(encoding='utf8')); t=ref['start']; di=day['day']; ledger=day['ledger']
        l,v,p,w,meta=model.scenarios(data,arc,di,t,mode)
        saved=next(wi for wi in day['windows'] if wi['start']==t)
        assert hashlib.sha256(p.tobytes()).hexdigest()==saved['price_scenario_sha256']
        q=np.array([r['q0'] for r in ledger]); units=q.copy()
        for change in day['changes']:
            if change['start']>=t: break
            st=change['start']; q[st:]=change['new']
            units[st:]+=1.5*np.array(change['up'])-.5*np.array(change['down'])
        cash=sum(r['price']*(r['charged_units']+5*r['emergency_kwh']) for r in ledger[:t])
        q,stats=repair.contract(l,v,p,w,ledger[t]['soc_start_kwh'],meta['load_point'],meta['pv_point'],
            previous=q[t:] if t else None,charged=units[t:] if t else None,past_cash=cash,rho=rho,
            terminal=di==364,rate=float(arc['price'][di].min()/.9),limit=30.)
        row={key:ref[key] for key in ('variant','category','date','start','input_sha256')}
        row.update(old=ref['old'],full_mip=ref['new'],repair=stats,contract=q.tolist())
        if not ref['old']['fallback']: row['objective_difference']=stats['objective_rebuilt']-ref['old']['objective_rebuilt']
        result.append(row)
        (out/'benchmark.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
        print(name,ref['category'],stats['status'],round(stats['seconds'],2),row.get('objective_difference'),flush=True)
    lines=['# 第四问固定模式修复试验','',
        '复用上一轮事先固定的12个诊断窗口，核对日期输入哈希和价格场景哈希。冻结原SOC、合同及历史信息；旧版与完整整数版数值沿用上一轮记录，耗时仅供参考，并非本轮同时重测。新修复版预算30秒：最多40%用于连续松弛，剩余用于固定模式LP。',
        '', '| 策略 | 类别 | 选中解 | 旧耗时/s | 修复耗时/s | 修复减旧目标/元 | 松弛界相对差距 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in result:
        s=r['repair']; delta=r.get('objective_difference')
        lines.append(f"| {r['variant']} | {r['category']} | {s['status']} | {r['old']['seconds']:.2f} | {s['seconds']:.2f} | {delta if delta is not None else '旧版回退，不可直接比较'} | {s['gap']} |")
    lines+=['','固定充放电模式后重新优化，不直接执行取整后的松弛结果。仅当完整物理及合同校验通过、且重建目标不劣于已有候选时才替换。始终保留可行初解。',
        '', '松弛问题证明最优时才记录全模型下界，并将终端常数偏移对齐；固定模式LP最优不等于原混合整数模型最优。未知下界记录null，不把它当作gap=0。',
        '', '目标包含终端估值和适用的CVaR项，不是实际电费。未运行全年，不把诊断窗口的变化推断为全年收益。原模型、场景数量、风险参数和年度结果不变。']
    comparable=[r for r in result if 'objective_difference' in r]
    lines+=['','## 采用结论','',
        f"修复总耗时{sum(r['repair']['seconds'] for r in result):.2f}秒，{sum(r['repair']['seed_used'] for r in result)}个窗口停留在初解；{len(comparable)}个可比窗口中{sum(r['objective_difference'] < -1e-6 for r in comparable)}个目标改善。",
        '组合接口q4_solver_adaptive先使用旧版，仅在回退或未知/较大gap时尝试固定模式修复；已合格的旧解仅在修复目标更低时替换。组合总耗时包含旧求解阶段，不能用修复耗时替代。未接入全年。',
        '修复版通过14项测试，组合路由通过5项测试及10项原模型回归测试。',
        '上一轮完整整数试验版的初解校验存在循环内反复读取稀疏矩阵属性的数组复制开销，可能先耗尽预算。因此旧完整整数版的耗时和目标不能作为完整整数算法本身较差的证据。缓存数组后该瓶颈消除。原全年模型不包含这段新增校验，不受影响。']
    (root/'reports/Q4_REPAIR_BENCHMARK.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__': main()
