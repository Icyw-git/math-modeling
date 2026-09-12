# 论文工程

本目录为 C 题论文的 LaTeX 工程，采用 2026 年全国大学生数学建模竞赛 `cumcmthesis` 模板。

## 编译与提交设置

- 入口文件：`main.tex`
- 编译引擎：XeLaTeX，建议连续编译两次
- 电子版设置：`\documentclass[withoutpreface]{cumcmthesis}`，不生成封面与承诺页
- AI 工具声明：位于参考文献之前；使用 AI 时，提交包还需另附真实的《AI工具使用详情.pdf》
- 图表：正文引用 `../figures/` 下的 PDF 图件，技术路线图位于 `assets/`

`paper_before_revision.pdf` 是修改前的成稿快照，不是当前源文件的编译结果。当前环境未发现 XeLaTeX，需在安装完整 TeX 环境后编译并检查最终 PDF。
