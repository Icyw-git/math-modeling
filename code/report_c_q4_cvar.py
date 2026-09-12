"""Generate four data-driven PDF figures and result report after verification."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
sys.path.append(str(ROOT/'tmp/q2_plot_deps'))
sys.path.append(str(ROOT/'tmp/q3_deps'))
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/q4_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json
from run_c_q4_cvar import VARIANTS


def main():
    dest=ROOT/'figures/q4_cvar'; dest.mkdir(parents=True,exist_ok=True)
    records=[]; frames={}; summaries={}; quality=[]
    for name in ['q2_neutral','q2_cvar','q3_neutral','q3_cvar']:
        p=ROOT/'results/q4_cvar'/name
        s=json.loads((p/'summary.json').read_text()); v=json.loads((p/'verification.json').read_text())
        assert s['complete'] and v['passed'] and s['days']==334
        records.append(dict(variant=name,**s)); summaries[name]=s
        frames[name]=pd.read_csv(p/'ledger.csv')
        windows=[w for f in sorted(p.glob('day_*.json')) for w in json.loads(f.read_text(encoding='utf8'))['windows'] if w['status']!='contract_frozen']
        quality.append(dict(variant=name,contract_windows=len(windows),fallbacks=sum(w['fallback'] for w in windows),max_reported_gap=max((w['gap'] for w in windows if w.get('gap') is not None),default=None),max_window_seconds=max(w['seconds'] for w in windows),status_counts={status:sum(w['status']==status for w in windows) for status in sorted(set(w['status'] for w in windows))},verification=v))
    summary=pd.DataFrame(records); summary.to_csv(dest/'comparison.csv',index=False)
    (dest/'quality.json').write_text(json.dumps(quality,indent=2,ensure_ascii=False),encoding='utf8')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42})
    labels=['问题2\n期望费用','问题2\nCVaR','问题3\n期望费用','问题3\nCVaR']
    def save(fig,name):
        fig.savefig(dest/f'{name}.pdf',bbox_inches='tight'); fig.savefig(dest/f'{name}.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4)); x=np.arange(4)
    normal=summary.initial_plan_fee+summary.increase_fee-summary.refund+summary.cancel_penalty
    ax.bar(x,normal/1e6,label='普通合同与调整结算',color='#4878a8'); ax.bar(x,summary.emergency_fee/1e6,bottom=normal/1e6,label='应急费用',color='#da8853')
    ax.set_xticks(x,labels); ax.set_ylabel('全年费用 / 百万元'); ax.legend(frameon=False); save(fig,'cost_components')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,mode in zip(axes,['q2','q3']):
        for kind,label in [('neutral','期望费用'),('cvar','CVaR')]:
            d=pd.read_csv(ROOT/'results/q4_cvar'/f'{mode}_{kind}'/'daily.csv')
            y=np.sort(d.total_fee.to_numpy()); ranks=(np.arange(len(y))+.5)/len(y)
            ax.plot(ranks,y/1e4,label=label)
        ax.set_xlim(.8,1); ax.set_xlabel('实际日费用经验分位'); ax.set_ylabel(f'{"问题2" if mode=="q2" else "问题3"} 日费用 / 万元'); ax.legend(frameon=False)
    save(fig,'daily_tail')
    chosen=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
    fig,axes=plt.subplots(2,2,figsize=(10,6)); source=[]
    for ax,date in zip(axes.flat,chosen):
        d=frames['q3_cvar'].query('date == @date'); source.append(d)
        ax.plot(d.interval/6,d.price,label='实际价格',color='#4878a8'); ax.plot(d.interval/6,d.price_forecast_0,label='0时预测',color='#da8853',linestyle='--')
        ax.set_xlabel(f'{date} 时刻 / h'); ax.set_ylabel('价格 / (元/kWh)'); ax.legend(frameon=False)
    pd.concat(source).to_csv(dest/'specified_days_source.csv',index=False); fig.tight_layout(); save(fig,'price_forecast')
    fig,axes=plt.subplots(4,2,figsize=(12,11))
    for row,date in enumerate(chosen):
        for col,mode in enumerate(['q2','q3']):
            d=frames[mode+'_cvar'].query('date == @date'); ax=axes[row,col]; twin=ax.twinx()
            ax.plot(d.interval/6,d.q*6,label='普通购电',color='#4878a8'); ax.plot(d.interval/6,d.emergency_kwh*6,label='应急购电',color='#da8853')
            twin.plot(d.interval/6,d.soc_start_kwh/1000,label='SOC',color='#548f63',linestyle='--'); twin.set_ylim(0,12); twin.set_ylabel('储能 / MWh')
            ax.set_ylabel('功率 / kW'); ax.set_xlabel(f'{date} {"问题2" if col==0 else "问题3"} 时刻 / h')
            if row==0: ax.legend(loc='upper left',frameon=False); twin.legend(loc='upper right',frameon=False)
    fig.tight_layout(); save(fig,'dispatch_soc')
    tables=[]; four=[]; emergencies=[]
    for mode in ('q2','q3'):
        for date in chosen:
            d=frames[mode+'_cvar'].query('date == @date')
            for t in (60,72,84,96,108,120):
                r=d[d.interval==t].iloc[0]; tables.append(dict(mode=mode,date=date,interval=int(t),ordinary_kwh=r.q,emergency_kwh=r.emergency_kwh))
            for start in range(0,144,24):
                b=d[(d.interval>=start)&(d.interval<start+24)]; four.append(dict(mode=mode,date=date,start_hour=start/6,charge_kwh=b.charge_kwh.sum(),discharge_kwh=b.discharge_kwh.sum(),start_soc=b.soc_start_kwh.iloc[0],end_soc=b.soc_end_kwh.iloc[-1]))
            active=None; energy=0.
            for r in d.itertuples():
                if r.emergency_kwh>1e-6:
                    if active is None: active=r.interval
                    energy+=r.emergency_kwh
                elif active is not None:
                    emergencies.append(dict(mode=mode,date=date,start_interval=active,end_interval=r.interval,energy_kwh=energy)); active=None; energy=0.
            if active is not None: emergencies.append(dict(mode=mode,date=date,start_interval=active,end_interval=144,energy_kwh=energy))
    pd.DataFrame(tables).to_csv(dest/'specified_purchases.csv',index=False); pd.DataFrame(four).to_csv(dest/'specified_storage.csv',index=False); pd.DataFrame(emergencies).to_csv(dest/'specified_emergencies.csv',index=False)
    lines=['# 第四问：联合价格场景与CVaR全年结果','',
           '本轮优先四组分别优化并独立验收，正式期334天、48096段，初末SOC均1200 kWh。全部未来价未知，不使用未来实际负载或PV；已知价格两组后置，其未完成检查点不纳入结果。',
           '', '| 策略 | 总费用/元 | 平均日费用/元 | 最大日费用/元 | 实际日费用CVaR90/元 | 应急费用/元 | 回退窗口 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n,s in summaries.items():
        daily=pd.read_csv(ROOT/'results/q4_cvar'/n/'daily.csv')
        lines.append(f'| {n} | {s["total_fee"]:.2f} | {daily.total_fee.mean():.2f} | {daily.total_fee.max():.2f} | {s["realized_daily_cvar90"]:.2f} | {s["emergency_fee"]:.2f} | {int(s["fallbacks"])} |')
    lines+=['','## 风险控制的实际效果','']
    for mode in ('q2','q3'):
        a=summaries[mode+'_neutral']; b=summaries[mode+'_cvar']
        lines.append(f'- {mode}：CVaR方案相对风险中性，总费变化{b["total_fee"]-a["total_fee"]:+.2f}元；实际日CVaR90变化{b["realized_daily_cvar90"]-a["realized_daily_cvar90"]:+.2f}元。正值表示更贵或尾部更高，负值表示改善。')
    lines+=['','## 独立验收和求解质量','']
    for item in quality:
        v=item['verification']; lines.append(f'- {item["variant"]}：{item["contract_windows"]}个合同窗口，回退{item["fallbacks"]}次，最大报告gap={item["max_reported_gap"]}，最大物理残差={v["max_physical_error"]:.3g} kWh，年末SOC={v["final_soc"]:.6f} kWh。')
        if v.get('prefix_days'): lines.append(f'  独立试跑前缀{v["prefix_days"]}天，最大差={v["max_prefix_difference"]}，一致={v["prefix_match"]}。')
    lines+=['','## 模型与限制','',
    '合同目标为0.8期望场景账单+0.2场景CVaR90，固定预测终端价值另计；实时控制仍是共同期望延续值。场景独立储能补救偏乐观，滚动CVaR不是嵌套动态风险。28条等权历史路径的尾部估计较粗。窗口场景CVaR与此表实际日费用CVaR是不同指标。已知价为算法信息对照，不称严格下界。',
    '正价格采用对数残差场景；先前未交付合同的场景价格风险保留。实际费用不含终端估值或风险惩罚。互斥不合格或无可行解时使用明示回退，限时解的gap单独保存。30秒为求解器时间预算，不是严格进程墙钟上限，预处理等可能导致超时；详细状态和最大窗口耗时见quality.json。LP松弛只有通过物理互斥检查才使用，不把全年策略宣称为全局最优。',
    '', '## 文件与复现','',
    '逐日合同版本、窗口CVaR、预测来源、执行台账在results/q4_cvar各策略目录；verification.json是独立验收。figures/q4_cvar包含comparison.csv、四个指定日的购电/储能/应急表及四类PDF。',
    '图表：price_forecast.pdf、cost_components.pdf、daily_tail.pdf、dispatch_soc.pdf。PNG仅供视觉检查。',
    '运行：python code/run_c_q4_cvar.py --variant q2_cvar（其余名称见上表）；随后python code/verify_c_q4_cvar.py results/q4_cvar/q2_cvar；全部验收后python code/report_c_q4_cvar.py。四次回测可用python code/run_c_q4_batch.py统一执行，已有运行时不要重复启动。先固定OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1，并确保本地highspy可读。',
    '正式Excel及论文正文未生成，未自动提交Git。']
    (ROOT/'reports/Q4_RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__': main()
