# HMML 分层建模方法库设计

## 设计依据

MM-Agent 论文提出的 HMML 按“领域、子领域、方法节点”组织建模知识。论文中的系统先分析并拆分问题，再检索候选方法；作者的检索代码沿层级给节点评分，将候选方法排序后交给建模阶段。

本项目借鉴这个结构和检索顺序，为排班优化实现一个小型版本。知识卡片由本项目自行编写，范围先覆盖排班、资源分配、匹配、回归和随机模拟。

## 目录结构

- data/hmml.json：分层知识数据。
- src/math_modeling_agent/hmml.py：Pydantic 知识节点和 JSON 加载。
- src/math_modeling_agent/retriever.py：问题与目标双路关键词检索。
- src/math_modeling_agent/agent.py：先检索方法，再调用求解器和验证器。

## 三层节点

1. 领域：例如运筹优化、统计与预测、仿真与不确定性。
2. 子领域：例如排班与资源分配、匹配与网络优化。
3. 方法节点：记录核心思路、适用条件、假设、局限、求解器以及实现状态。

实现状态区分“已实现”和“知识候选”。目前只有 CP-SAT 与项目里的排班求解器对应；其他方法仅供 Agent 比较和解释，不能被误认为已经可执行。

## 检索过程

检索器分别接收问题描述和期望目标，沿每个方法节点的祖先路径累计关键词命中分：

- 领域层命中权重为 1。
- 子领域层命中权重为 2。
- 方法层命中权重为 3。
- 最终分数由问题匹配分数的 70% 和目标匹配分数的 30% 组成。

较具体的方法词对排名影响更大。结果附带各层路径、分项分数、适用条件和实现状态，方便追踪推荐依据。

这些权重是本项目的第一版工程参数，并非 MM-Agent 论文的原始参数。当前检索使用可解释的关键词命中，没有使用嵌入模型或 LLM 重排。

## 运行流程

一次排班建模运行的顺序为：

问题结构 → HMML 候选检索 → CP-SAT 求解 → 独立约束验证

后续接入自然语言理解后，问题分析阶段可以产生问题描述、子任务和目标，再把它们传给 HMML 检索器。

## 参考

- 论文：MM-Agent: LLM as Agents for Real-world Mathematical Modeling Problem，https://arxiv.org/abs/2505.14148
- 作者检索实现：https://github.com/usail-hkust/LLM-MM-Agent/blob/main/MMAgent/agent/retrieve_method.py
- 作者中文项目说明：https://github.com/usail-hkust/LLM-MM-Agent/blob/main/README_zh.md
