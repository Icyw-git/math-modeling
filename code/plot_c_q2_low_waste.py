"""Figures keep actual electricity fees separate from design penalties."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp'/'q2_plot_deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp'/'q2_mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    previous=ROOT/'results'/'q2_enhanced_grid193'/'enhanced'
    current=ROOT/'results'/'q2_low_waste'; out=ROOT/'figures'/'q2_low_waste'; out.mkdir(parents=True,exist_ok=True)
    a=pd.read_csv(previous/'ledger.csv'); b=pd.read_csv(current/'ledger.csv')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    f,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    normal=[a.plan_cost_yuan.sum()/1e4,b.plan_cost_yuan.sum()/1e4]
    emergency=[a.emergency_cost_yuan.sum()/1e4,b.emergency_cost_yuan.sum()/1e4]
    axes[0].bar([0,1],normal,color='#376795',label='计划电费')
    axes[0].bar([0,1],emergency,bottom=normal,color='#D78843',hatch='//',label='应急电费')
    axes[0].set_ylabel('实际费用 / 万元')
    for i,total in enumerate(np.array(normal)+emergency): axes[0].text(i,total+10,f'{total:.2f}',ha='center')
    spill=np.array([a.unused_kwh.sum(),b.unused_kwh.sum()])/1e4
    loss=np.array([(.1*a.charge_kwh+(1/.9-1)*a.discharge_kwh).sum(),b.conversion_loss_kwh.sum()])/1e4
    axes[1].bar([0,1],spill,color='#5C9F82',label='未利用电量')
    axes[1].bar([0,1],loss,bottom=spill,color='#B7C4CB',hatch='..',label='充放电损耗')
    axes[1].set_ylabel('电量 / 万kWh')
    for i,total in enumerate(spill+loss): axes[1].text(i,total+2,f'{total:.2f}',ha='center')
    for ax in axes:
        ax.set_xticks([0,1],['上一精细版','低未利用量版']); ax.legend(frameon=False,loc='upper right')
        ax.set_ylim(0,ax.get_ylim()[1]*1.16)
    f.savefig(out/'cost_and_energy.pdf'); f.savefig(out/'cost_and_energy.png',dpi=160); plt.close(f)
    a['month']=pd.to_datetime(a.date).dt.month; b['month']=pd.to_datetime(b.date).dt.month
    monthly=pd.DataFrame({'previous_unused_kwh':a.groupby('month').unused_kwh.sum(),'new_unused_kwh':b.groupby('month').unused_kwh.sum(),
                          'previous_cost_yuan':a.groupby('month').plan_cost_yuan.sum()+a.groupby('month').emergency_cost_yuan.sum(),
                          'new_cost_yuan':b.groupby('month').plan_cost_yuan.sum()+b.groupby('month').emergency_cost_yuan.sum()})
    f,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    axes[0].plot(monthly.index,monthly.previous_unused_kwh/1e4,'o-',label='上一精细版',color='#D78843')
    axes[0].plot(monthly.index,monthly.new_unused_kwh/1e4,'s-',label='低未利用量版',color='#376795')
    axes[0].set(xlabel='月份',ylabel='未利用量 / 万kWh',xticks=range(2,13)); axes[0].legend(frameon=False)
    axes[1].bar(monthly.index,(monthly.previous_cost_yuan-monthly.new_cost_yuan)/1e4,color='#376795')
    axes[1].axhline(0,color='gray',lw=.7); axes[1].set(xlabel='月份',ylabel='相比上一版节省 / 万元',xticks=range(2,13))
    f.savefig(out/'monthly_comparison.pdf'); f.savefig(out/'monthly_comparison.png',dpi=160); plt.close(f)
    monthly.to_csv(out/'monthly_source.csv',encoding='utf-8-sig')
    source=[previous/'ledger.csv',current/'ledger.csv']
    (out/'sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},indent=2),encoding='utf-8')
    print(str(out))


if __name__=='__main__': main()
