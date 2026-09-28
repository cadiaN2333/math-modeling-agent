# 线性规划领域适配器设计

## 目标

在保留现有员工排班能力的基础上，为运筹优化 Agent 增加第二个真实求解领域：连续变量、单目标线性规划。以生产计划问题作为端到端案例，完成自然语言澄清、结构化建模、方法推荐、求解和独立验证。

## 本期范围

- 支持连续决策变量、线性最大化/最小化目标，以及 `<=`、`>=`、`==` 线性约束。
- 变量上下界作为显式线性约束表示，例如非负变量写作 `x >= 0`，不增加可空上下界字段。
- 使用 OR-Tools `pywraplp` 的 GLOP 求解连续 LP。官方 Python 示例通过 `pywraplp.Solver.CreateSolver("GLOP")` 创建求解器，并读取求解状态、目标值和变量值。[官方 LP 示例](https://developers.google.com/optimization/lp/lp_example)
- 生产计划示例：最大化 `40A + 30B`，约束为 `2A + B <= 100`、`A + B <= 80`、`A >= 0`、`B >= 0`；预期 `A=20`、`B=60`、最优目标值为 `2600`。
- 暂不支持整数/二进制变量、非线性、多目标、不确定/随机规划、运输/网络流。用户要求必须取整的产量时，本期标记为 unsupported，不对连续解自行取整。

## 接入方案比较

1. **单独另建 LP 分析和 CLI 管线：** 改动少，但与排班重复编排、澄清和错误处理，后续难以统一。
2. **立即抽象覆盖所有优化领域的万能模型：** 长期灵活，但只有两个领域实例前容易过度设计，且一次改动面过大。
3. **共享分析状态 + 固定领域草稿 + 领域求解器分流：** 本期采用。保留现有排班适配器，新增 LP 草稿与 LP 求解器；等运输/网络流再落地后，依据真实共性抽取更稳定的公共接口。

## 结构化分析格式

扩展 `ProblemAnalysis`，加入必填 `problem_family`（`employee_scheduling`、`linear_programming`、`other`）。保留必填 `scheduling_draft`，新增必填 `linear_program_draft`；两个字段均不得是 Optional/Union：

- family 为排班且 ready 时，只填排班草稿，LP 草稿使用固定空值形状。
- family 为 LP 且 ready 时，只填 LP 草稿，排班草稿三个列表为空。
- `needs_clarification` 或 `unsupported` 时两个草稿均为空；已识别领域可保留在 `problem_family`，其他领域使用 `other`。

LP 草稿由变量列表、目标方向、目标项列表和约束列表构成。目标方向使用 `maximize`、`minimize` 或 `not_applicable`；约束关系使用 `<=`、`>=`、`==` 或 `not_applicable`。非活动 LP 草稿以空变量/约束列表、`not_applicable` 和空目标项表示，以维持 DeepSeek Structured Outputs 的固定 JSON Schema，不引入 `anyOf`。

## 模块与数据流

- `analysis_agent.py`：按 `problem_family` 分析；对整数规划、非线性、多目标和运输/网络流明确返回 unsupported；增加 `to_linear_program_problem()` 转换及交叉引用校验。
- `models.py`：加入 LP 领域输入模型，检查变量唯一、目标/约束的变量引用有效、系数和 RHS 有限。
- 新增 `linear_solver.py`：使用 GLOP 创建连续变量、线性约束和目标，统一输出 `OPTIMAL`、`INFEASIBLE`、`UNBOUNDED` 等状态。
- 新增 `linear_validator.py`：独立重算每条约束左侧和目标值；按 `1e-6` 容差验证关系及报告变量值。
- `cli.py`：根据 `problem_family` 分流到原排班编排或 LP 编排，保持共同顶层响应字段 `analysis`、`method_recommendations`、`modeling_run`。
- `data/hmml.json`：新增连续线性规划/GLOP 方法卡，并标记为已实现；不要把整数规划或网络流误标为已实现。
- Evals：把生产计划案例从 unsupported 边界升级为 ready/optimal；新增一个 LP 无解案例；运输/网络流继续作为 unsupported 边界。

## 验证与失败处理

- 单元测试覆盖 LP 结构校验、最优生产计划、无解、无界、非最优状态不读取变量解，以及独立 validator 的可行/不可行报告。
- CLI/Evals 测试使用假分析输出或直接构造 `ProblemAnalysis`，普通 pytest 不访问 DeepSeek。
- 只有用户显式运行 `--live` 才调用 DeepSeek；真实响应必须通过 LP schema、求解器和独立 validator。
- GLOP 无法创建或输入格式无效时返回明确的 solver 状态/错误，不生成伪造解。
