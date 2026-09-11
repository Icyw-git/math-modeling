"""Report verified Q3 experiments; preserve unfavorable comparisons."""
import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'tmp/q2_plot_deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/q3_mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=ROOT/'results/q3_refined'; figs=ROOT/'figures/q3_refined'; figs.mkdir(parents=True,exist_ok=True)
    modes=['reference','solver','adaptive']; names=['原预算对照','改进求解器','条件负载场景']; summaries=[]; daily={}; ledgers={}
    for mode in modes:
        assert json.loads((root/mode/'verification.json').read_text())['passed']
        s=json.loads((root/mode/'summary.json').read_text()); summaries.append(dict(mode=mode,**s))
        daily[mode]=pd.read_csv(root/mode/'daily.csv'); ledgers[mode]=pd.read_csv(root/mode/'ledger.csv')
    table=pd.DataFrame(summaries); table['saving_yuan']=table.total_fee_yuan.iloc[0]-table.total_fee_yuan; table['saving_percent']=100*table.saving_yuan/table.total_fee_yuan.iloc[0]; table.to_csv(root/'comparison.csv',index=False)
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,col,label in zip(axes,['total_fee_yuan','emergency_kwh','unused_kwh'],['真实费用 / 万元','应急电量 / 万kWh','未利用量 / 万kWh']):
        ax.bar(names,table[col]/1e4,color=['#777777','#376795','#D78843']); ax.set_ylabel(label); ax.tick_params(axis='x',labelrotation=15)
    fig.savefig(figs/'comparison.pdf'); fig.savefig(figs/'comparison.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained'); dates=pd.to_datetime(daily['reference'].date)
    for mode,name in zip(modes[1:],names[1:]): ax.plot(dates,(daily['reference'].total_fee_yuan-daily[mode].total_fee_yuan).cumsum()/1e4,label=name)
    ax.axhline(0,color='gray',lw=.7); ax.set_ylabel('累计节省 / 万元'); ax.legend(); fig.savefig(figs/'cumulative.pdf'); fig.savefig(figs/'cumulative.png',dpi=160); plt.close(fig)
    selected=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']; fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    colors={'reference':'#777777','solver':'#376795','adaptive':'#D78843'}
    for day,ax in zip(selected,axes.flat):
        for mode,name in zip(modes,names):
            g=ledgers[mode]; g=g[g.date==day]; ax.plot((g.interval+1)/6,g.soc_end_kwh,label=name,color=colors[mode])
        ax.set_xlabel(f'{day} 时刻 / h'); ax.set_ylabel('储能 / kWh')
    axes[0,0].legend(); fig.savefig(figs/'selected_soc.pdf'); fig.savefig(figs/'selected_soc.png',dpi=160); plt.close(fig)
    datafiles=[root/'comparison.csv']+[root/m/'daily.csv' for m in modes]+[root/m/'ledger.csv' for m in modes]
    (figs/'sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in datafiles},indent=2),encoding='utf8')
    text='# 第三问精进结果\n\n## 实验口径\n\n旧报告对应的全年台账在当前本地缺失，故重建预算对照。三组均7场景、97点（100 kWh）控制网格、2月1日到12月31日334天，期初期末1200 kWh。旧源码不改；所有组修正执行时的年末可达性。对照是本机0.5秒预算复跑，不保证复现旧报告金额，短限时受运行环境影响。\n\n'
    text+='求解器组保留旧预测场景，使用选择性整数约束、10秒上限和逐窗口物理回代。条件场景组另加入已发生日内负载误差的收缩预测与历史相似度加权，不读取未来实测；参数事先固定，未按全年成绩调参。\n\n'
    text+='## 全年实际结果\n\n| 方案 | 费用/元 | 较对照节省/元 | 节省比例 | 应急量/kWh | 未利用量/kWh |\n|---|---:|---:|---:|---:|---:|\n'
    for name,r in zip(names,table.to_dict('records')): text+=f'| {name} | {r["total_fee_yuan"]:.2f} | {r["saving_yuan"]:.2f} | {r["saving_percent"]:.3f}% | {r["emergency_kwh"]:.2f} | {r["unused_kwh"]:.2f} |\n'
    gain=table.total_fee_yuan.iloc[1]-table.total_fee_yuan.iloc[2]
    text+=f'\n条件负载场景相对仅改求解器的费用变化：节省 {gain:.2f} 元（负数为恶化）。不把较低预测误差直接等同于经济收益，不将全年算法比较当作全局最优证明。\n\n## 求解质量与验收\n\n'
    a,c=table.iloc[0],table.iloc[2]
    text+=f'重要权衡：条件场景相对预算对照，未利用量减少 {100*(1-c.unused_kwh/a.unused_kwh):.2f}%，但应急电量增加 {100*(c.emergency_kwh/a.emergency_kwh-1):.2f}%，应急费用增加 {100*(c.emergency_fee_yuan/a.emergency_fee_yuan-1):.2f}%，转换损耗增加 {100*(c.conversion_loss_kwh/a.conversion_loss_kwh-1):.2f}%。未利用量与转换损耗之和仍减少 {100*(1-(c.unused_kwh+c.conversion_loss_kwh)/(a.unused_kwh+a.conversion_loss_kwh)):.2f}%。因此本版改善真实总电费，不是所有指标同时改善。\n\n'
    for name,r in zip(names,table.to_dict('records')): text+=f'- {name}：回退 {int(r["fallbacks"])} 次，最大报告gap {r["max_gap"]}，墙钟时间 {r["seconds"]:.1f} 秒。\n'
    for mode,name in zip(modes,names):
        log=pd.read_csv(root/mode/'solves.csv'); text+=f'- {name}求解状态：{log.status.value_counts().to_dict()}。\n'
        if 'seconds' in log: text+=f'  单窗口平均 {log.seconds.mean():.3f} 秒，最长 {log.seconds.max():.3f} 秒。\n'
    text+='\n全部48096段通过非负、供需平衡、SOC递推与跨日连续、功率、互斥、应急不充电及年度端点校验。逐次合同从0时原合同重建，并用最终合同费用加半价总变动量独立校验账单；原附件和模型依赖哈希未变。gap在目标接近零时可能放大，须结合状态和实际回代误差阅读。\n\n原版与新版合计12项测试通过；条件场景独立7天试跑与全年前7天的购电、调整、充放电、应急和SOC逐值一致。没有声称已完成全年重复运行。\n\n'
    text+='细网格短期敏感性：100 kWh网格7天费用332591.28元，50 kWh网格332717.11元，期末SOC均1908.63 kWh。差额125.83元（约0.038%），不是单调改善，也不是全年网格收敛证明。正式三组使用统一100 kWh网格。\n\n'
    text+='## 四个指定日期\n\n| 日期 | 对照费用/元 | 求解器费用/元 | 条件场景费用/元 |\n|---|---:|---:|---:|\n'
    for day in selected: text+='| '+day+' | '+' | '.join(f'{daily[m].set_index("date").loc[day,"total_fee_yuan"]:.2f}' for m in modes)+' |\n'
    text+='\n## 限制与复现\n\n沿用取消退原价并罚50%、增购150%的工作解释；不通过改账单追求低费用。各情景未来储能独立补救仍偏乐观，忽略未来更新的期权价值；固定线性末端值仍是启发式。一个年度回测不保证未来收益。\n\n对照使用 code/run_c_q3_reference.py（仅修正微小负合同数值处理，较大异常明确回退）；改进组运行 code/solve_c_q3_refined.py --mode solver 或 adaptive --output 对应新目录；再运行 code/verify_c_q3_refined.py 目录，最后运行 code/report_c_q3_refined.py。项目局部依赖在tmp/q3_deps与tmp/c_solver_deps，未修改系统Python。\n\n图表及可追溯源：figures/q3_refined/comparison.pdf、cumulative.pdf、selected_soc.pdf、sources.json。\n'
    (ROOT/'reports/Q3_REFINED_RESULTS.md').write_text(text,encoding='utf8'); print(table.to_string(index=False))

if __name__=='__main__': main()
