# 第四问CVaR首版执行计划

仅新增第四问对应问题2/3的独立代码、结果与图表，不覆盖前三问，不生成正式Excel，不自动提交Git。

## 项目目录结构

- code/q4_model.py：预测、联合场景、均值-CVaR合同与期望控制。
- code/run_c_q4_cvar.py：六组运行、逐日检查点与安全续跑。
- code/test_c_q4_cvar.py、code/verify_c_q4_cvar.py：单元与独立验收。
- results/q4_cvar/：各实验独立目录。
- figures/q4_cvar/：四类图表及源数据。
- reports/Q4_MODELING_REPORT.md、第四问结果报告：定义、实际结果、限制与复现。

## 顺序

1. 单元检查与7天试跑，检查泄露、CVaR和账单。
2. 优先四组全年实验：q2/q3各自比较neutral与cvar。两组known按用户要求后置，保留检查点；保留回退、实际gap、完整状态与来源。
3. 独立重建账单和物理约束，比较短期与全年前缀。
4. 汇总真实费用、实际日CVaR及图表；如实记录风险控制的反例。
