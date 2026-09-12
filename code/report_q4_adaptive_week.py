"""Compare seven-day continuous runs to the original annual prefix."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'tmp/q2_plot_deps'))
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/q4_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    dest=ROOT/'figures/q4_adaptive_week'; dest.mkdir(parents=True,exist_ok=True)
    names=('q2_neutral','q2_cvar','q3_neutral','q3_cvar')
    results=[]; source={}; hashes={}
    for name in names:
        new_dir=ROOT/'results/q4_adaptive_week'/name; old_dir=ROOT/'results/q4_cvar'/name
        verification=json.loads((new_dir/'verification.json').read_text())
        assert verification['passed'] and verification['days']==7 and verification['intervals']==1008
        old_ver=json.loads((old_dir/'verification.json').read_text()); assert old_ver['passed'] and old_ver['days']==334
        new_files=sorted(new_dir.glob('day_*.json')); assert len(new_files)==7
        series={}; stat={}
        for kind,paths in (('old',[old_dir/p.name for p in new_files]),('new',new_files)):
            rows=[json.loads(p.read_text(encoding='utf8')) for p in paths]
            for p in paths: hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
            assert [r['day'] for r in rows]==list(range(31,38))
            assert abs(rows[0]['ledger'][0]['soc_start_kwh']-1200)<1e-6
            daily=[r['daily'] for r in rows]; ledger=[r for d in rows for r in d['ledger']]
            windows=[w for d in rows for w in d['windows'] if w['status']!='contract_frozen']
            fees=[d['total_fee'] for d in daily]
            series[kind]=dict(dates=[d['date'] for d in daily],daily_fees=fees,soc=[r['soc_start_kwh'] for r in ledger]+[ledger[-1]['soc_end_kwh']])
            stat[kind]=dict(fee=sum(fees),emergency_fee=sum(d['emergency_fee'] for d in daily),
                emergency_kwh=sum(d['emergency_kwh'] for d in daily),unused_kwh=sum(d['unused_kwh'] for d in daily),
                end_soc=ledger[-1]['soc_end_kwh'],fallbacks=sum(w['fallback'] for w in windows),
                windows=len(windows),window_seconds=sum(w['seconds'] for w in windows),
                repairs_attempted=sum(len(w.get('stages',[]))==2 for w in windows),
                repairs_accepted=sum(w.get('route')=='lp_repair' for w in windows),
                max_daily_fee=max(fees))
            rate=min(r['price_forecast_0'] for r in rows[-1]['ledger'])/.9
            stat[kind]['reference_rate']=rate
        assert abs(stat['old']['reference_rate']-stat['new']['reference_rate'])<1e-12
        delta=stat['new']['fee']-stat['old']['fee']; delta_e=stat['new']['end_soc']-stat['old']['end_soc']
        results.append(dict(variant=name,**stat,fee_difference=delta,soc_difference=delta_e,
            value_adjusted_difference=delta-rate*delta_e,verification=verification))
        source[name]=series
    hashes[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (dest/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    (dest/'plot_source.json').write_text(json.dumps(source,indent=2),encoding='utf8')
    (dest/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'pdf.fonttype':42,'font.size':9})
    fig,axes=plt.subplots(4,2,figsize=(12,11))
    for row,name in enumerate(names):
        label=('问题2' if name.startswith('q2') else '问题3')+(' / CVaR' if name.endswith('cvar') else ' / 风险中性')
        for kind,color,style,title in (('old','#4878a8','-','原策略'),('new','#d5823e','--','择优修复')):
            s=source[name][kind]
            axes[row,0].plot(np.arange(1,8),np.array(s['daily_fees'])/1e4,color=color,linestyle=style,marker='o',markersize=3,label=title)
            axes[row,1].plot(np.arange(1009)/144,np.array(s['soc'])/1000,color=color,linestyle=style,label=title,linewidth=.9)
        axes[row,0].set_xlabel(label+'；2月日期'); axes[row,0].set_ylabel('实际日费用 / 万元'); axes[row,0].set_xticks(range(1,8))
        axes[row,1].set_xlabel(label+'；自2月1日0时起 / 天'); axes[row,1].set_ylabel('储能 / MWh'); axes[row,1].set_ylim(0,12)
    handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(dest/'cost_soc.pdf',bbox_inches='tight'); plt.close(fig)
    lines=['# 第四问择优修复：7天连续回测','',
        '2025年2月1日至7日，四种策略各1008段。新策略接入择优LP修复；原策略使用已独立验收的年度前7天。双方初始SOC均1200 kWh，使用相同历史信息、费用规则和终端价值设置，日间SOC连续。原求解器及年度结果未修改。',
        '', '## 实际费用与期末储能','',
        '| 策略 | 原费用/元 | 新费用/元 | 新减旧/元 | 原期末SOC/kWh | 新期末SOC/kWh |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        a,b=r['old'],r['new']; lines.append(f"| {r['variant']} | {a['fee']:.2f} | {b['fee']:.2f} | {r['fee_difference']:+.2f} | {a['end_soc']:.2f} | {b['end_soc']:.2f} |")
    lines+=['','## 应急、未利用电量及求解质量','',
        '| 策略 | 原/新应急电量/kWh | 原/新应急费/元 | 原/新未利用电量/kWh | 原/新最终回退 | 修复尝试/接受 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        a,b=r['old'],r['new']; lines.append(f"| {r['variant']} | {a['emergency_kwh']:.2f} / {b['emergency_kwh']:.2f} | {a['emergency_fee']:.2f} / {b['emergency_fee']:.2f} | {a['unused_kwh']:.2f} / {b['unused_kwh']:.2f} | {a['fallbacks']} / {b['fallbacks']} | {b['repairs_attempted']} / {b['repairs_accepted']} |")
    lines+=['','## 期末口径与结论边界','',
        '这不是全年终点试验，2月7日末不强制回到1200 kWh。为保持与原全年前缀一致，没有人为日末放空。不能将费用差直接解读为共同末SOC条件下的节省。',
        '辅助指标为费用差减去统一储能参考价乘期末SOC差；参考价取2月7日0时预测整日最低价除以0.9，与现有日末价值口径一致。它不是实际电费，也不是真实延续价值的证明。']
    for r in results: lines.append(f"- {r['variant']}：参考价{r['new']['reference_rate']:.6f}元/kWh，估值调整后差额{r['value_adjusted_difference']:+.2f}元。")
    lines+=['','7天不足以评价全年收益或尾部风险稳定性；本轮不据此调风险权重。新策略每个窗口可能改变后续SOC，属于连续重算，不是把单窗费用拼入旧结果。',
        '', '## 独立验收与复现','']
    for r in results: lines.append(f"- {r['variant']}：独立原附件、账单双公式、供需平衡、SOC递推、容量功率、互斥与无应急充电检查通过，最大物理残差{r['verification']['max_physical_error']:.3g} kWh。")
    lines+=['','运行 `python code/batch_q4_adaptive_week.py`，四组完成后运行 `python code/report_q4_adaptive_week.py`。默认只跑7天；配置及代码哈希一致时复用检查点。',
        '', '图表见[日费用与SOC](../figures/q4_adaptive_week/cost_soc.pdf)；同目录comparison.json保存原/新窗口耗时、修复次数和全部对比指标，plot_source.json及source_hashes.json保存图表源数据与来源。时间为不同运行的参考，不是严格同机同时配对的性能测试。',
        '', '未重跑全年、未生成正式Excel、未提交Git。']
    (ROOT/'reports/Q4_ADAPTIVE_WEEK.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__': main()
