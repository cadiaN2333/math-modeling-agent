# 线性规划领域适配器实施计划

> **执行说明：** 在 `D:\Agent` 当前仓库逐任务实施；每个生产行为先写失败测试，再写实现。

**目标：** 支持从自然语言提取连续 LP、使用 OR-Tools GLOP 求解并独立验证，同时保留排班 CP-SAT 流程。

**架构：** `ProblemAnalysis` 增加非可空 `problem_family`、`scheduling_draft` 和 `linear_program_draft` 固定字段，避免生成 DeepSeek 不支持的 `anyOf`。CLI 按问题族分流至排班或 LP 适配器；LP Evals 从 unsupported 升级为可求解，并增加无解案例。

**技术栈：** Python、Pydantic、OR-Tools `pywraplp` GLOP、pytest。

---

## 文件职责

- `src/math_modeling_agent/models.py`：LP 变量、线性项、目标、约束和问题模型及引用校验。
- `src/math_modeling_agent/analysis_agent.py`：新增 LP Structured Output 草稿、问题族校验和 `to_linear_program_problem`。
- `src/math_modeling_agent/linear_solver.py`：调用 GLOP，返回统一状态、目标值和变量值。
- `src/math_modeling_agent/linear_validator.py`：独立重算 LP 约束与目标值。
- `src/math_modeling_agent/linear_agent.py`：LP 求解、HMML 推荐与 validator 编排。
- `src/math_modeling_agent/cli.py`：按 `problem_family` 分流求解。
- `data/hmml.json`：加入标记“已实现”的连续线性规划/GLOP 方法卡。
- `src/math_modeling_agent/evals.py`、`evals/cases.json`：增加 LP draft 规范化、变量/目标评分及 LP 案例。
- `tests/test_models.py`、`tests/test_solver.py`、`tests/test_validator.py`、`tests/test_analysis_agent.py`、`tests/test_cli.py`、`tests/test_evals.py`：覆盖新模型端到端行为并更新 `ProblemAnalysis` 构造。
- `docs/本地运行与密钥配置.md`：说明新 LP 示例和评测命令。

## 任务一：内部 LP 模型与校验

- [x] 在 `tests/test_models.py` 先写测试，定义生产计划问题：最大化 `40A+30B`，约束 `2A+B<=100`、`A+B<=80`、`A>=0`、`B>=0`；并测试重复变量名、未声明变量引用被拒绝。
- [x] 运行对应测试，预期因 `LinearProgramProblem` 类型不存在而失败。
- [x] 在 `models.py` 增加 `LinearVariable(name, unit)`、`LinearTerm(variable, coefficient)`、`LinearConstraint(name, terms, relation, rhs)`、`LinearObjective(direction, terms)`、`LinearProgramProblem(variables, objective, constraints)`；关系为 `<=`、`>=`、`==`，目标为 maximize/minimize。
- [x] 用 Pydantic validator 检查变量名唯一、目标/约束引用都已声明，表达式不重复引用变量，系数和 RHS 为有限数。
- [x] 重跑模型测试，预期通过。

## 任务二：GLOP 求解和独立 validator

- [x] 在 `tests/test_solver.py` 写失败测试：生产计划应得到 `OPTIMAL`、`A=20`、`B=60`、目标值 `2600`；再加 `x>=10` 与 `x<=5` 的 INFEASIBLE 测试，确保无解时变量值为空。
- [x] 在 `tests/test_validator.py` 写失败测试：正确解通过；篡改一个变量值使约束违反后报告对应错误。
- [x] 新增 `linear_solver.py`：通过 `pywraplp.Solver.CreateSolver("GLOP")` 建模，非 OPTIMAL/FEASIBLE 不读取变量解；记录 GLOP 不可创建时的明确状态。
- [x] 新增 `linear_validator.py`：独立计算每个线性表达式和目标值，使用 `1e-6` 容差检查 `<=`/`>=`/`==` 和目标值一致性。
- [x] 运行两组聚焦测试，预期最优、无解和错误解均被正确区分。

## 任务三：固定的分析输出 Schema 与转换

- [x] 在 `tests/test_analysis_agent.py` 先增加测试：JSON Schema 仍不含 `anyOf`；ready LP 分析有 `problem_family="linear_programming"` 并可转换；非活动领域 draft 为空；整数/非线性问题为 unsupported。
- [x] 新增 Draft 类型：变量名/单位、线性项、约束、目标方向；`linear_program_draft` 在未使用时固定为 `variables=[]`、`objective_direction="not_applicable"`、`objective_terms=[]`、`constraints=[]`。
- [x] 在 `ProblemAnalysis` 中加入 `problem_family: Literal["employee_scheduling", "linear_programming", "other"]` 和必填 `linear_program_draft`。ready 时要求对应家族的 draft 完整、另一家族 draft 为空；非 ready 时两份 draft 都必须为空。
- [x] 新增 `to_linear_program_problem`，将 Draft 转成内部 LP 模型；更新现有所有分析器测试构造，补上问题族和 inactive LP draft。
- [x] 更新提示词：支持员工排班和连续单目标 LP；整数/二进制、非线性、多目标、运输网络流标 unsupported。变量界限用显式线性约束表示。
- [x] 运行分析器测试，预期 schema 无 `anyOf`，旧排班测试仍通过，新 LP 转换测试通过。

## 任务四：HMML、编排与 CLI 分流

- [x] 在 `data/hmml.json` 增加 `continuous_linear_programming` 方法卡，solver 为 OR-Tools GLOP，状态“已实现”，限制注明只支持连续线性目标与约束。
- [x] 在 `tests/test_cli.py` 写 LP ready 请求的假分析器测试，断言走 LP solver、返回最优状态并通过 LP validator；另测 unsupported/needs_clarification 不会启动任何求解器。
- [x] 新增 `linear_agent.py`，返回与排班相同顶层字段 `solver_result`、`validation_report`、`method_recommendations`。
- [x] 修改 `cli.py`：按 `problem_family` 路由 `employee_scheduling` 到原有 `run_modeling`，`linear_programming` 到新 LP 编排；`other` 不得求解。
- [x] 运行 CLI 和 HMML 测试，预期排班原有响应保持兼容，LP 生产计划输出 `OPTIMAL` 且验证通过。

## 任务五：LP Evals 升级

- [x] 在 `tests/test_evals.py` 先添加失败测试：`linear_programming` 案例使用 `linear_program_draft` 比较；期望变量值和 objective value 按 `1e-6` 容差评分。
- [x] 修改 `evals.py` 的 live dispatcher，按 family 调用对应 converter/runner；给 LP draft 添加顺序无关规范化。
- [x] 将 `evals/cases.json` 的生产计划 LP 案例改为 ready/OPTIMAL，期望 A=20、B=60、目标 2600；再加入 LP 无解案例。运输/网络流仍 unsupported。
- [x] 更新数据集测试：仍无重复 ID、问题族覆盖 scheduling/linear_programming/transportation，LP ready 案例必须声明 expected model、solver、method 和 validator。
- [x] 用假的 analyzer 测试 LP live 流程和评分，不发起网络请求。

## 任务六：文档、回归与提交

- [x] 更新本地运行文档，解释 LP 连续变量范围、整数决策仍不支持，并展示 offline Evals 与单案例 `--live` 命令。
- [x] 用虚拟环境内唯一 `--basetemp` 运行全套 PowerShell pytest；预期排班与 LP 测试均通过。
- [x] 执行 `python -m math_modeling_agent.evals`，确认 LP 案例结构有效且默认 `api_calls=0`；另经用户授权执行完整 `--live` 在线评测。
- [x] 检查 `.env` 未暂存、`git diff --cached --check` 通过后，以中文提交。
