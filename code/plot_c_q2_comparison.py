"""Paper-ready Chinese vector figures, from verified saved ledgers only."""
from pathlib import Path
import hashlib
import json
import os
import sys
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp'/'q2_plot_deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp'/'q2_mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    folder=ROOT/'results'/'q2_enhanced'
    out=ROOT/'figures'/'q2_comparison'; out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    table=pd.read_csv(folder/'comparison.csv')
    baseline=json.loads((ROOT/'results'/'q2_baseline'/'summary.json').read_text(encoding='utf-8'))
    labels={'enhanced':'改进主模型','no_margin':'无安全余量','one_day':'24小时前瞻','alpha65':'65%分位数','alpha90':'90%分位数'}
    costs=pd.concat([pd.DataFrame([dict(variant='baseline',plan_cost_yuan=baseline['plan_cost_yuan'],
                                      emergency_cost_yuan=baseline['emergency_cost_yuan'])]),table],ignore_index=True)
    names=['原基线']+[labels[x] for x in table.variant]
    fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained')
    x=np.arange(len(costs)); normal=costs.plan_cost_yuan/1e4; emergency=costs.emergency_cost_yuan/1e4
    ax.bar(x,normal,label='计划电费',color='#376795')
    ax.bar(x,emergency,bottom=normal,label='应急电费',color='#D78843',hatch='//')
    ax.set_xticks(x,names); ax.set_ylabel('正式期费用 / 万元'); ax.legend(frameon=False,ncols=2)
    for i,total in enumerate(normal+emergency): ax.text(i,total+8,f'{total:.1f}',ha='center',fontsize=9)
    ax.set_ylim(0,float((normal+emergency).max())*1.13)
    fig.savefig(out/'cost_comparison.pdf'); fig.savefig(out/'cost_comparison.png',dpi=160); plt.close(fig)
    base_daily=pd.read_csv(ROOT/'results'/'q2_baseline'/'daily.csv')
    new_daily=pd.read_csv(folder/'enhanced'/'daily.csv')
    paired=new_daily[['date','total_cost_yuan']].merge(base_daily[['date','total_cost_yuan']],on='date',suffixes=('_new','_baseline'))
    paired['month']=pd.to_datetime(paired.date).dt.month
    monthly=paired.groupby('month')[['total_cost_yuan_new','total_cost_yuan_baseline']].sum()
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    axes[0].plot(monthly.index,monthly.total_cost_yuan_baseline/1e4,'o-',label='原基线',color='#D78843')
    axes[0].plot(monthly.index,monthly.total_cost_yuan_new/1e4,'s-',label='改进主模型',color='#376795')
    axes[0].set(xlabel='月份',ylabel='月度费用 / 万元',xticks=range(2,13)); axes[0].legend(frameon=False)
    axes[1].plot(pd.to_datetime(paired.date),(paired.total_cost_yuan_baseline-paired.total_cost_yuan_new).cumsum()/1e4,color='#376795')
    axes[1].axhline(0,color='gray',lw=.8); axes[1].set(xlabel='日期',ylabel='累计节省费用 / 万元')
    fig.savefig(out/'monthly_and_cumulative.pdf'); fig.savefig(out/'monthly_and_cumulative.png',dpi=160); plt.close(fig)
    f=pd.read_csv(folder/'enhanced'/'ledger.csv'); fb=pd.read_csv(ROOT/'results'/'q2_baseline'/'ledger.csv')
    selected=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
    fig,axes=plt.subplots(4,2,figsize=(11,10),layout='constrained')
    for i,date in enumerate(selected):
        a=f[f.date==date]; b=fb[fb.date==date]; t=np.arange(144)/6
        axes[i,0].plot(t,a.load_kwh*6,label='负载',color='#303030',lw=1)
        axes[i,0].plot(t,a.pv_kwh*6,label='光伏',color='#63A778',lw=1)
        axes[i,0].plot(t,a.purchase_kwh*6,label='普通购电',color='#376795',lw=1)
        axes[i,0].plot(t,a.emergency_kwh*6,label='应急购电',color='#D78843',lw=1)
        axes[i,0].set_ylabel(f'{date[5:]}\n功率 / kW')
        axes[i,1].plot(np.arange(145)/6,np.r_[b.soc_start_kwh.iloc[0],b.soc_end_kwh],label='原基线',color='#D78843',lw=1)
        axes[i,1].plot(np.arange(145)/6,np.r_[a.soc_start_kwh.iloc[0],a.soc_end_kwh],label='改进主模型',color='#376795',lw=1)
        axes[i,1].axhline(1200,color='gray',ls=':',lw=.7); axes[i,1].axhline(10800,color='gray',ls=':',lw=.7)
        axes[i,1].set_ylabel('储能量 / kWh')
        for ax in axes[i]: ax.set(xlim=(0,24),xticks=[0,6,12,18,24],xlabel='时刻 / h')
    axes[0,0].legend(frameon=False,ncols=4,fontsize=8); axes[0,1].legend(frameon=False,ncols=2)
    fig.savefig(out/'selected_days.pdf'); fig.savefig(out/'selected_days.png',dpi=150); plt.close(fig)
    monthly.to_csv(out/'monthly_source.csv',encoding='utf-8-sig')
    sources=[folder/'comparison.csv',folder/'enhanced'/'ledger.csv',folder/'enhanced'/'daily.csv',
             ROOT/'results'/'q2_baseline'/'ledger.csv',ROOT/'results'/'q2_baseline'/'daily.csv']
    (out/'sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    refined=ROOT/'results'/'q2_enhanced_grid193'/'enhanced'
    if (refined/'summary.json').exists():
        sr=json.loads((refined/'summary.json').read_text(encoding='utf-8'))
        fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
        normal=np.array([baseline['plan_cost_yuan'],sr['plan_cost_yuan']])/1e4
        emergency=np.array([baseline['emergency_cost_yuan'],sr['emergency_cost_yuan']])/1e4
        axes[0].bar([0,1],normal,color='#376795',label='计划电费')
        axes[0].bar([0,1],emergency,bottom=normal,color='#D78843',hatch='//',label='应急电费')
        axes[0].set(xticks=[0,1],xticklabels=['原基线','改进版（50 kWh网格）'],ylabel='费用 / 万元',ylim=(0,1800))
        axes[0].legend(frameon=False)
        for i,total in enumerate(normal+emergency): axes[0].text(i,total+15,f'{total:.2f}',ha='center')
        grids=[]
        for size,source in [(200,folder/'enhanced'),(100,ROOT/'results'/'q2_enhanced_grid97'/'enhanced'),(50,refined)]:
            data=json.loads((source/'summary.json').read_text(encoding='utf-8'))
            grids.append(dict(grid_kwh=size,total_cost_yuan=data['total_cost_yuan'],emergency_cost_yuan=data['emergency_cost_yuan']))
        g=pd.DataFrame(grids)
        axes[1].plot(np.arange(3),g.total_cost_yuan/1e4,'o-',color='#376795')
        axes[1].set(xticks=[0,1,2],xticklabels=['200','100','50'],xlabel='储能网格间距 / kWh',ylabel='改进模型总费用 / 万元')
        for i,cost in enumerate(g.total_cost_yuan/1e4): axes[1].annotate(f'{cost:.2f}',(i,cost),xytext=(0,8),textcoords='offset points',ha='center')
        axes[1].margins(x=.15,y=.2)
        fig.savefig(out/'refined_comparison.pdf'); fig.savefig(out/'refined_comparison.png',dpi=160); plt.close(fig)
        g.to_csv(out/'grid_source.csv',index=False,encoding='utf-8-sig')
        refined_sources=[refined/'summary.json',ROOT/'results'/'q2_enhanced_grid97'/'enhanced'/'summary.json',folder/'enhanced'/'summary.json']
        (out/'refined_sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in refined_sources},indent=2),encoding='utf-8')
    print(str(out))


if __name__=='__main__': main()
