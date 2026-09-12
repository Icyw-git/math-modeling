"""Summarize four predetermined seasonal weeks after independent verification."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT/'tmp/q2_plot_deps'))
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/q4_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEASONS={'winter':('冬季','2025-02-01','2025-02-07'),
         'spring':('春季','2025-05-15','2025-05-21'),
         'summer':('夏季','2025-08-15','2025-08-21'),
         'autumn':('秋季','2025-11-15','2025-11-21')}
NAMES=('q2_neutral','q2_cvar','q3_neutral','q3_cvar')

def aggregate(paths):
    days=[json.loads(p.read_text(encoding='utf8')) for p in paths]
    daily=[d['daily'] for d in days]
    ledger=[r for d in days for r in d['ledger']]
    windows=[w for d in days for w in d['windows'] if w['status']!='contract_frozen']
    return dict(fee=sum(d['total_fee'] for d in daily),emergency_fee=sum(d['emergency_fee'] for d in daily),
        emergency_kwh=sum(d['emergency_kwh'] for d in daily),unused_kwh=sum(d['unused_kwh'] for d in daily),
        start_soc=ledger[0]['soc_start_kwh'],end_soc=ledger[-1]['soc_end_kwh'],
        max_daily=max(d['total_fee'] for d in daily),fallbacks=sum(w['fallback'] for w in windows),
        attempts=sum(len(w.get('stages',[]))==2 for w in windows),accepted=sum(w.get('route')=='lp_repair' for w in windows),
        window_seconds=sum(w['seconds'] for w in windows),reference_rate=min(r['price_forecast_0'] for r in days[-1]['ledger'])/.9)

def main():
    dest=ROOT/'figures/q4_adaptive_seasons'; dest.mkdir(parents=True,exist_ok=True)
    records=[]; hashes={}
    for season,(cn,start,end) in SEASONS.items():
        for name in NAMES:
            new_dir=ROOT/'results/q4_adaptive_seasons'/season/name
            verification=json.loads((new_dir/'verification.json').read_text())
            assert verification['passed'] and verification['days']==7 and verification['prefix_days']==7
            new_paths=sorted(new_dir.glob('day_*.json')); old_paths=[ROOT/'results/q4_cvar'/name/p.name for p in new_paths]
            assert len(new_paths)==7 and all(p.exists() for p in old_paths)
            assert new_paths[0].name==f'day_{start}.json' and new_paths[-1].name==f'day_{end}.json'
            config=json.loads((new_dir/'config.json').read_text(encoding='utf8'))
            for relative,digest in {**config['source_hashes'],**config['input_hashes']}.items():
                assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
            old=aggregate(old_paths); new=aggregate(new_paths)
            assert abs(old['start_soc']-new['start_soc'])<1e-6
            assert abs(old['reference_rate']-new['reference_rate'])<1e-12
            for p in new_paths+old_paths+[new_dir/'config.json',new_dir/'verification.json']:
                hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
            delta=new['fee']-old['fee']; delta_soc=new['end_soc']-old['end_soc']
            records.append(dict(season=season,season_cn=cn,start=start,end=end,variant=name,old=old,new=new,
                fee_difference=delta,fee_change_percent=100*delta/old['fee'],soc_difference=delta_soc,
                value_adjusted_difference=delta-new['reference_rate']*delta_soc,verification=verification))
    hashes[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (dest/'comparison.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    (dest/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'pdf.fonttype':42,'font.size':9})
    fig,axes=plt.subplots(1,2,figsize=(11,4.2)); x=np.arange(4); width=.19
    colors=['#4c78a8','#f58518','#54a24b','#b279a2']
    for i,name in enumerate(NAMES):
        rows=[r for season in SEASONS for r in records if r['season']==season and r['variant']==name]
        label=('问题2' if name.startswith('q2') else '问题3')+(' CVaR' if name.endswith('cvar') else ' 风险中性')
        axes[0].bar(x+(i-1.5)*width,[r['fee_change_percent'] for r in rows],width,label=label,color=colors[i])
        axes[1].bar(x+(i-1.5)*width,[r['new']['fallbacks']-r['old']['fallbacks'] for r in rows],width,label=name,color=colors[i])
    labels=[SEASONS[s][0] for s in SEASONS]
    for ax in axes: ax.axhline(0,color='black',linewidth=.7); ax.set_xticks(x,labels)
    axes[0].set_ylabel('新策略实际费用变化 / %'); axes[1].set_ylabel('最终回退次数变化')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False); fig.tight_layout(rect=(0,0,1,.91))
    fig.savefig(dest/'seasonal_comparison.pdf',bbox_inches='tight'); plt.close(fig)
    lines=['# 第四问多模式择优修复：四季连续回测','',
        '四个自然季节各预先固定一个连续7天窗口，四种策略共16组、112策略日。日期不是根据结果挑选。每组新策略与原年度策略使用相同起始SOC、历史信息、费用规则和终端价值口径；日间SOC连续。',
        '', '| 季节 | 策略 | 费用变化/元 | 变化率 | 期末SOC差/kWh | 估值调整差/元 | 原/新回退 | 尝试/接受 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        lines.append(f"| {r['season_cn']} | {r['variant']} | {r['fee_difference']:+.2f} | {r['fee_change_percent']:+.2f}% | {r['soc_difference']:+.2f} | {r['value_adjusted_difference']:+.2f} | {r['old']['fallbacks']} / {r['new']['fallbacks']} | {r['new']['attempts']} / {r['new']['accepted']} |")
    improved=sum(r['fee_difference']<-.01 for r in records); adjusted=sum(r['value_adjusted_difference']<-.01 for r in records)
    worse=sum(r['fee_difference']>.01 for r in records); unchanged=len(records)-improved-worse
    old_fb=sum(r['old']['fallbacks'] for r in records); new_fb=sum(r['new']['fallbacks'] for r in records)
    emergency_increases=sum(r['new']['emergency_fee']-r['old']['emergency_fee']>.01 for r in records)
    lines+=['','## 汇总结论','',
        f'- 16组中，未经期末估值调整有{improved}组费用下降、{worse}组上升、{unchanged}组基本不变（0.01元容差）；按统一线性储能参考价调整后有{adjusted}组差额为负。',
        f'- 最终回退合计由{old_fb}次变为{new_fb}次。修复尝试与接受分开记录，未接受的候选不会覆盖原合格解。',
        '- 四季短期验证仍不是全年收益证明，也不能稳定估计90%尾部风险；没有据此修改CVaR权重或场景数量。',
        '- 未利用电量、应急电量、窗口耗时和最大日费用均保存在comparison.json，结论不能只依据总费用。',
        f'- {emergency_increases}组应急费用增加超过0.01元；总费用下降不等于所有供电风险指标改善。',
        '', '## 供电、弃用与期末口径','',
        '| 季节 | 策略 | 应急费用变化/元 | 应急电量变化/kWh | 未利用电量变化/kWh | 期末参考价/(元/kWh) |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in records:
        a,b=r['old'],r['new']
        lines.append(f"| {r['season_cn']} | {r['variant']} | {b['emergency_fee']-a['emergency_fee']:+.2f} | {b['emergency_kwh']-a['emergency_kwh']:+.2f} | {b['unused_kwh']-a['unused_kwh']:+.2f} | {b['reference_rate']:.6f} |")
    lines+=['','期末估值调整差=实际费用差−期末SOC差×参考价。参考价取该周最后一天0时预测最低电价/0.9，两个版本共用；它不是电费或真实延续价值保证。四种策略只分别与自己的原版本比较，跨季节起始SOC并不相同。',
        '', '冬季窗口此前已参与开发验证，其余三季提供扩展验证。四季之间不连续，本次不是全年回测。季节日志的旧days字段自2月累计，实际完成天数以summary/verification和7个日期文件为准。',
        '', '## 验收','']
    for r in records:
        v=r['verification']; lines.append(f"- {r['season_cn']} {r['variant']}：7天1008段独立验收通过，最大物理残差{v['max_physical_error']:.3g} kWh；与原结果比较覆盖{v['prefix_days']}天。")
    lines+=['','## 下一阶段判断','',
        '四季验证在物理可行性、回退减少和成本表现上支持进入固定参数的全年连续验证。当前阶段已完成循环消除、多模式候选、四季验证和这一判断；尚未启动新的全年运行。',
        '全年验证应保留四种策略、同样的28条场景及风险权重，初末SOC均1200 kWh，从2月1日连续运行至12月31日。不得拼接这四个季节结果，也不得以每周原SOC重置。以原年度四组作为各自对照，比较实际费用、日费用CVaR、应急费用、弃用和回退。候选更好只保证同窗口模型目标，不保证全年费用或风险下降。',
        'Benders和动态CVaR仍后置：先处理非凸补救下有效割的条件，再决定分解是否合适；不与候选搜索的全年验证同时改变模型。',
        '', '图表见[季节费用与回退比较](../figures/q4_adaptive_seasons/seasonal_comparison.pdf)，原始指标和来源哈希在同目录JSON。未重跑全年、未生成正式Excel、未提交Git。']
    (ROOT/'reports/Q4_SEASONAL_VALIDATION.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__': main()
