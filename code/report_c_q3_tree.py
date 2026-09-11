"""Compare real bills, not planning objectives; retain adverse findings."""
import os
import sys
import json
import hashlib
import platform
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'tmp/q2_plot_deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/q3_tree_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=ROOT/'results/q3_tree'; figs=ROOT/'figures/q3_tree'; figs.mkdir(parents=True,exist_ok=True)
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__}
    for dist in importlib.metadata.distributions(path=[str(ROOT/'tmp/c_solver_deps'),str(ROOT/'tmp/q3_deps')]):
        if dist.metadata['Name'].lower() in ('highspy','scipy'): versions[dist.metadata['Name']]=dist.version
    (root/'environment.json').write_text(json.dumps(dict(versions=versions,solver_threads=1,random_seed=0,mip_relative_gap=.001,time_limit_seconds=15,nominal_grid_kwh=100),indent=2),encoding='utf8')
    paths={'adaptive':ROOT/'results/q3_refined/adaptive','frozen':root/'frozen','rolling':root/'rolling'}
    names={'adaptive':'旧条件场景方案','frozen':'场景树：冻结未来调整','rolling':'场景树：预期未来调整'}
    daily={}; ledgers={}; records=[]
    assert json.loads((root/'test_results.json').read_text())['passed']
    for mode,path in paths.items():
        assert json.loads((path/'verification.json').read_text())['passed']
        if mode!='adaptive': assert json.loads((path/'tree_verification.json').read_text())['passed']
        daily[mode]=pd.read_csv(path/'daily.csv'); ledgers[mode]=pd.read_csv(path/'ledger.csv')
        record=dict(mode=mode,**json.loads((path/'summary.json').read_text()))
        record['delivered_contract_fee_yuan']=float(daily[mode].final_contract_fee_yuan.sum())
        record['adjustment_friction_yuan']=record['plan_fee_yuan']-record['delivered_contract_fee_yuan']
        records.append(record)
    table=pd.DataFrame(records).set_index('mode'); old=table.loc['adaptive']; new=table.loc['rolling']; frozen=table.loc['frozen']
    table['cost_change_vs_adaptive_yuan']=table.total_fee_yuan-old.total_fee_yuan
    table['cost_change_percent']=100*table.cost_change_vs_adaptive_yuan/old.total_fee_yuan
    table.to_csv(root/'comparison.csv')
    # Repeated first-week execution, including actual decisions and contracts.
    pilot=pd.read_csv(root/'compact_pilot/ledger.csv'); full=ledgers['rolling'].iloc[:len(pilot)]
    numeric=pilot.select_dtypes(include='number').columns
    maximum=float(np.max(abs(pilot[numeric].to_numpy()-full[numeric].to_numpy())))
    repeat=dict(intervals=len(pilot),max_numeric_difference=maximum,passed=maximum<=1e-6)
    (root/'prefix_reproducibility.json').write_text(json.dumps(repeat,indent=2),encoding='utf8')
    selected=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
    selected_rows=[]
    for date in selected:
        for mode in paths: selected_rows.append(dict(mode=mode,**daily[mode].set_index('date').loc[date].to_dict(),date=date))
    pd.DataFrame(selected_rows).to_csv(root/'selected_dates.csv',index=False)
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    labels=[names[m] for m in paths]; colors=['#6b7280','#d99243','#237eab']
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
    for ax,col,label in zip(axes,['total_fee_yuan','emergency_kwh','unused_kwh'],['实际电费 / 万元','应急电量 / 万kWh','未利用电量 / 万kWh']):
        bars=ax.bar(range(3),table[col]/1e4,color=colors); ax.set_xticks(range(3),['旧方案','树：冻结调整','树：预期调整'],rotation=15); ax.set_ylabel(label)
        ax.bar_label(bars,fmt='%.2f',padding=3); ax.margins(y=.15)
    fig.savefig(figs/'comparison.pdf'); fig.savefig(figs/'comparison.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    dates=pd.to_datetime(daily['adaptive'].date)
    for mode,color in zip(['frozen','rolling'],colors[1:]): axes[0].plot(dates,(daily[mode].total_fee_yuan-daily['adaptive'].total_fee_yuan).cumsum()/1e4,label=names[mode],color=color)
    axes[0].axhline(0,color='gray',lw=.7); axes[0].set_ylabel('相对旧方案累计多支出 / 万元'); axes[0].legend(fontsize=8)
    fee_cols=['delivered_contract_fee_yuan','adjustment_friction_yuan','emergency_fee_yuan']
    changes=[new[c]-old[c] for c in fee_cols]
    bars=axes[1].bar(['交付合同电费','调整摩擦成本','应急电费'],np.array(changes)/1e4,color=['#d99243' if c>0 else '#237eab' for c in changes]); axes[1].bar_label(bars,fmt='%+.2f'); axes[1].axhline(0,color='gray',lw=.7); axes[1].set_ylabel('预期调整树 − 旧方案 / 万元'); axes[1].margins(y=.2)
    fig.savefig(figs/'cost_diagnosis.pdf'); fig.savefig(figs/'cost_diagnosis.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for date,ax in zip(selected,axes.flat):
        for mode,color in zip(paths,colors):
            g=ledgers[mode]; g=g[g.date==date]; ax.plot((g.interval+1)/6,g.soc_end_kwh,label=names[mode],color=color)
        for hour in (6,12,18): ax.axvline(hour,color='gray',lw=.5,alpha=.4)
        ax.set_xlabel(f'{date} 时刻 / h'); ax.set_ylabel('SOC / kWh')
    axes[0,0].legend(fontsize=8); fig.savefig(figs/'selected_soc.pdf'); fig.savefig(figs/'selected_soc.png',dpi=160); plt.close(fig)
    sources=[root/'comparison.csv']+[p/f for p in paths.values() for f in ['daily.csv','ledger.csv','summary.json']]
    (figs/'sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf8')
    diff=new.total_fee_yuan-old.total_fee_yuan; ablation=frozen.total_fee_yuan-new.total_fee_yuan
    report='# 第三问四阶段场景树：实测对照结果\n\n'
    report+=f'已完成0/6/12/18四阶段场景树与滚动控制。相对此前条件场景方案，全年实际电费变化为 **{diff:+,.2f}元（{100*diff/old.total_fee_yuan:+.3f}%）**。金额变正表示恶化，不把更复杂模型自动视为更优。\n\n'
    report+='## 可比口径\n\n三组均统计2025年2月1日至12月31日334天、48096个10分钟时段，期初期末1200 kWh，沿用共同一月预热。效率、容量、功率、免费弃用、不卖电、应急只补负载、逐次调整结算及100 kWh控制网格一致。树正式版复用旧版7条条件代表路径；仅密集初试使用全部历史路径。原模型的末端线性估值保留，不计入实际电费。未填写时间标签待澄清的Excel，不涉及第四问。\n\n'
    report+='## 全年对比\n\n| 指标 | 旧条件场景方案 | 树：冻结未来调整 | 树：预期未来调整 |\n|---|---:|---:|---:|\n'
    for field,label in [('total_fee_yuan','真实总电费/元'),('plan_fee_yuan','普通购电及调整结算/元'),('delivered_contract_fee_yuan','最终交付合同电费/元'),('adjustment_friction_yuan','各次变动的半价摩擦成本/元'),('emergency_fee_yuan','应急费用/元'),('emergency_kwh','应急电量/kWh'),('adjustment_abs_kwh','合同总变动量/kWh'),('unused_kwh','未利用量/kWh'),('conversion_loss_kwh','转换损耗/kWh')]: report+='| '+label+' | '+' | '.join(f'{table.loc[m,field]:,.2f}' for m in paths)+' |\n'
    report+=f'\n同一场景树内部，允许规划未来调整相对冻结组节省 **{ablation:,.2f}元**（负数表示更贵）。这项对照比直接与旧方案比较更接近“未来调整机会”的作用，但并非严格的单因素全局最优比较。实际到达6/12/18时，两组均可重新调整；冻结组仅在当前规划中不预期这种机会。\n\n'
    report+='## 差额从何而来\n\n'
    for col,label in zip(fee_cols,['最终交付合同电费','合同调整摩擦成本','应急电费']): report+=f'- {label}：新树比旧方案变化 {new[col]-old[col]:+,.2f} 元。\n'
    report+=f'\n三项之和严格等于总费用变化。本版应急费用下降，但未利用量增加{new.unused_kwh-old.unused_kwh:,.2f} kWh（{100*(new.unused_kwh/old.unused_kwh-1):.2f}%）。单看应急减少不能判断降费成功。不能把有剩余SOC时的全部应急费用视为可消除浪费，因为放电功率和后续需求也可能限制决策。\n\n'
    report+='机制解释属于模型分析而非已证明的唯一因果：树内6小时共享计划储能动作，比旧版情景独立补救保守；滚动执行又使用节点混合的Bellman延续值，计划与执行并非完全一致。有限代表路径分支到单条路径后还可能过度自信。未来调价机会有价值，但此实现的近似误差可能更大。\n\n'
    report+='## 验证与计算\n\n'
    for mode in ['frozen','rolling']:
        s=table.loc[mode]; log=pd.read_csv(paths[mode]/'solves.csv'); v=json.loads((paths[mode]/'tree_verification.json').read_text())
        report+=f'- {names[mode]}：{int(s.fallbacks)}次回退；总耗时{s.seconds:.1f}秒；最大报告gap={s.max_gap:.6g}；状态{log.status.value_counts().to_dict()}；最多{v["max_leaves"]}个叶子。\n'
        for _,failure in log[log.fallback==True].iterrows(): report+=f'  回退窗口：{failure.date} {int(failure.hour)}时；原因：{failure.get("tree_failure", "见日志")}。该窗口使用已校验的两阶段备用规划，因此消融比较不是每个窗口都严格同构。\n'
    tests=json.loads((root/'test_results.json').read_text())
    report+=f'\n{tests["tests"]}项第三问回归测试通过，覆盖未来实测与未发布预报隔离、节点分组与概率、冻结合同、年度端点、快速递推与完整递推一致、注入失败后的已校验回退等。初次直接发现测试因旧模块未预先加入局部SciPy路径而加载失败；使用独立测试入口完成全套运行，未改旧源码。\n\n'
    report+=f'正式两组供需平衡、SOC递推、跨日连续、功率、互斥、应急不充电、年度端点及逐次合同账单重建全部通过10⁻⁶ kWh容差。虚拟节点SOC链接、概率分区及当前根合同与实际版本也独立核对。7天试跑与全年前1008段最大数值差为{maximum:.3g}，重复前缀校验={repeat["passed"]}；不声称重复运行了两次全年。求解限时可使跨机器结果略有差异。\n\n'
    report+='主树求解预算15秒，相对gap目标0.001；若失败，备用规划另给15秒，因此整个窗口可超过15秒。限时可行解必须回代通过，gap未达到目标仍如实记录；没有将所有结果声称为最优。28条历史路径初试7天发生4次回退，随后正式配置恢复为与旧版一致的7代表路径；该选择基于计算与可比性，不是按全年费用挑选参数。短期末SOC不同的费用不用于收益判断。两组正式实验固定参数，未根据全年结果重新调参。\n\n'
    report+='## 四个指定日期\n\n| 日期 | 旧方案费用/元 | 冻结树费用/元 | 滚动树费用/元 |\n|---|---:|---:|---:|\n'
    for date in selected: report+='| '+date+' | '+' | '.join(f'{daily[m].set_index("date").loc[date,"total_fee_yuan"]:,.2f}' for m in paths)+' |\n'
    report+='\n每日起止SOC、应急量与未利用量详见 results/q3_tree/selected_dates.csv，不能将某天费用孤立地解释为长期收益。\n\n## 结论与复现\n\n'
    report+=('本版未胜过此前条件场景方案，不建议替换此前最低费用结果。下一步应先解决计划与实时控制的一致性、6小时内储能保守性，并用时间顺序验证集选参数，而不是直接继续加深场景树或用全年金额反复调参。\n\n' if diff>=0 else '本版在该年度下降实际费用，但仍是近似策略，需要独立年份验证，不声称全年全局最优。\n\n')
    report+='正式入口：`python code/run_c_q3_tree.py --output 新目录`；消融增加`--freeze-future`。验证：`python code/verify_c_q3_tree.py 结果目录`。测试：`python code/run_c_q3_tree_tests.py`。出报告和图：`python code/report_c_q3_tree.py`。密集试验入口为`code/solve_c_q3_tree.py`，正式7路径入口为`code/run_c_q3_tree.py`，不要混淆。\n\n'
    report+='依赖沿用项目局部NumPy/Pandas/SciPy/HiGHS及Matplotlib；配置记录输入与求解源码SHA256，图源记录数据SHA256。独立文件夹results/q3_tree保存台账、每日费用、transactions.csv、trees.jsonl（所有信息节点及虚拟合同）、solves.csv和验证报告；figures/q3_tree含comparison.pdf、cost_diagnosis.pdf、selected_soc.pdf及PNG预览。原结果不覆盖，不提交或推送Git。\n'
    (ROOT/'reports/Q3_TREE_RESULTS.md').write_text(report,encoding='utf8')
    print(table.to_string()); print('prefix',repeat)


if __name__=='__main__': main()
