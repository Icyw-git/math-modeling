# 数学建模竞赛仓库

面向大学生数学建模竞赛（国赛 CUMCM / 美赛 MCM-ICM / 华为杯 / 华中杯等）的一站式工作仓库，包含 AI 辅助建模工作流（Skills）、竞赛论文 LaTeX 模板与配套资源。

## 仓库结构

```
.
├── skills/                                      # 建模竞赛工作流 Skills
│   ├── 1start-mathmodel/                        # 工作流总控入口：生成 plan.md / todo.md
│   ├── 2analysis-modeling/                      # 赛题分析与模型设计
│   ├── 3coding-visual/                          # 编程求解与数据图表
│   ├── 4drawio/                                 # 流程图与模型结构图
│   ├── 5writing/                                # 论文写作（含 Typst/LaTeX 模板）
│   ├── 6verity/                                 # 结果与论文验收
│   ├── doctor/                                  # 依赖检查与安装向导
│   ├── mathmodel-figure-templates/              # 科研绘图模板
│   ├── typst-author/                            # Typst 写作与离线参考文档
│   └── _references/                             # 共享数学建模规范
└── 基于2026年全国大学生数学建模LaTeX模版的项目/   # CUMCM LaTeX 论文模板
    ├── cumcmthesis.cls / cumcm2026.sty          # 模板类与样式文件
    ├── example.tex / example.pdf                # 示例论文与编译效果
    └── figures/                                 # 模板所需图片资源
```

## 工作流程

以 `1start-mathmodel` 为入口，按以下阶段推进：

1. **赛题分析与建模设计**（`2analysis-modeling`）：审题、确定模型方向与子问题拆分
2. **编程实现和图表生成**（`3coding-visual`）：模型求解代码与数据可视化
3. **流程与架构图绘制**（`4drawio`）：模型结构图、算法流程图
4. **竞赛论文撰写**（`5writing`）：按所选竞赛类型调用对应模板成文
5. **验证和验收**（`6verity`）：检查结果正确性与论文完整性

开始前可运行 `doctor` 检查 Python、排版引擎（Typst / XeLaTeX）、DrawIO 等依赖是否就绪。

## 使用方式

- **Skills 工作流**：将 `skills/` 下需要的 Skill 文件夹放入 AI 编码工具的个人 Skills 目录（保留子目录结构），从 `1start-mathmodel` 启动。各阶段共同使用 `_references` 中的建模规范。
- **论文模板**：进入 `基于2026年全国大学生数学建模LaTeX模版的项目/`，参考 `example.tex` 编写论文，使用 `xelatex` 编译（需跑两遍以解决交叉引用）。模板含 2026 年更新的 AI 使用声明书。

## 依赖说明

运行绘图、编译论文和导出流程图需要 Python、排版引擎（Typst / XeLaTeX）、DrawIO 等本地程序；安装 Skill 本身不安装这些依赖，可通过 `doctor` 获取安装指引。
