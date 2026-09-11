"""Produce a data-derived comparison report and vector figures after verification."""
import hashlib
import json
import os
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/q2_plot_deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/q2_mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    out=ROOT/'results/q2_terminal_value'; figdir=ROOT/'figures/q2_terminal_value'; figdir.mkdir(parents=True,exist_ok=True)
    assert json.loads((out/'verification.json').read_text())['passed']
    new=pd.read_csv(out/'ledger.csv'); old=pd.read_csv(ROOT/'results/q2_enhanced_grid193/enhanced/ledger.csv')
    summary=json.loads((out/'summary.json').read_text()); assert len(new)==len(old)==48096
    rows=[]
    for name,frame in [('固定线性估值',old),('下一日成本曲线',new)]:
        rows.append(dict(model=name,cost=float((frame.plan_cost_yuan+frame.emergency_cost_yuan).sum()),emergency=float(frame.emergency_kwh.sum()),unused=float(frame.unused_kwh.sum()),loss=float((.1*frame.charge_kwh+(1/.9-1)*frame.discharge_kwh).sum())))
    comparison=pd.DataFrame(rows); comparison.to_csv(out/'comparison.csv',index=False)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key,label in zip(axes,['cost','emergency','unused'],['实际电费 / 万元','应急电量 / 万kWh','未利用电量 / 万kWh']):
        ax.bar(comparison.model,comparison[key]/1e4,color=['#376795','#D78843']); ax.set_ylabel(label)
    fig.savefig(figdir/'comparison.pdf'); fig.savefig(figdir/'comparison.png',dpi=150); plt.close(fig)
    selected=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']; fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for date,ax in zip(selected,axes.flat):
        curve=pd.read_csv(out/f'curve_{date}.csv'); ax.plot(curve.energy,curve.value,label='下一日成本曲线')
        ax.plot(curve.energy,-old.price_yuan_per_kwh.min()/.9*(curve.energy-1200),'--',label='固定线性估值')
        ax.set_xlabel(f'{date} 日末储能 / kWh'); ax.set_ylabel('相对后续成本 / 元'); ax.legend()
    fig.savefig(figdir/'terminal_curves.pdf'); fig.savefig(figdir/'terminal_curves.png',dpi=150); plt.close(fig)
    files=[out/'ledger.csv',out/'comparison.csv']+[out/f'curve_{d}.csv' for d in selected]
    (figdir/'sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf8')
    text='# 第二问：固定购电的终端估值对照\n\n'
    text+='普通购电逐段固定为原50 kWh增强模型，仅替换日末价值函数。25个400 kWh间隔储能点的下一日情景成本经线性插值进入193点控制网格。情景独立补救为偏乐观两阶段近似，不是场景树。末端值不计入真实电费。\n\n'
    text+='| 模型 | 实际费用/元 | 应急量/kWh | 未利用量/kWh | 转换损耗/kWh |\n|---|---:|---:|---:|---:|\n'
    for r in rows: text+=f'| {r["model"]} | {r["cost"]:.2f} | {r["emergency"]:.2f} | {r["unused"]:.2f} | {r["loss"]:.2f} |\n'
    text+=f'\n相对对照节省 {rows[0]["cost"]-rows[1]["cost"]:.2f} 元；未利用量减少 {rows[0]["unused"]-rows[1]["unused"]:.2f} kWh。负值表示恶化，不预设新方法更优。\n'
    text+=f'\n共334天48096段，期初/期末储能均1200 kWh；曲线回退 {summary["fallback_days"]} 天，运行耗时 {summary["seconds"]:.1f} 秒。物理与费用校验见 results/q2_terminal_value/verification.json。\n'
    text+='\n一月使用原基线预热；本实验没有重算购电，因此不能解释成完整日前策略改进。求解容差0.001可能使成本曲线带有数值噪声，未强行单调化。\n\n四个指定日期结果：\n\n'
    daily=pd.read_csv(out/'daily.csv')
    for r in daily[daily.date.isin(selected)].to_dict('records'): text+=f'- {r["date"]}：实际费用 {r["total_cost_yuan"]:.2f} 元，未利用量 {r["unused_kwh"]:.2f} kWh。\n'
    text+='\n图表：figures/q2_terminal_value/comparison.pdf、terminal_curves.pdf；数值源及哈希随图保存。图表已生成，人工视觉检查需另外确认。\n'
    (ROOT/'reports/Q2_TERMINAL_VALUE_REPORT.md').write_text(text,encoding='utf8')

if __name__=='__main__': main()
