# 混合整数线性规划扩展实施计划

> **For agentic workers:** 在 `feature/min-cost-network-flow` 隔离工作树内执行；遵循测试先行，每个求解行为先确认测试按预期失败，再实现并复跑。

**目标：** 扩展现有线性规划领域，使自然语言和结构化输入都能表达连续、整数、二进制变量的单目标线性模型，并用适当的 OR-Tools 后端求解和独立验证。

**架构：** 连续 LP 和 MILP 共用现有 `LinearProgramProblem`、Draft、CLI 领域路由及 validator。每个变量声明连续/整数/二进制域；全连续模型仍走 GLOP，含离散变量的线性模型走 SCIP。HMML 与 Evals 标识实际方法和能力边界；不支持非线性、多目标和非线性整数模型。

**技术栈：** Python 3.11、Pydantic、OR-Tools `pywraplp`（GLOP / SCIP）、DeepSeek Responses 结构化输出、pytest。

---

## 文件职责

- 修改 `src/math_modeling_agent/models.py`：为内部线性变量增加 `domain`，默认连续以兼容直接构造的旧 LP 输入。
- 修改 `src/math_modeling_agent/analysis_agent.py`：在线性变量 Draft 中加入必填域字段，更新提示词、integer unsupported 处理和 Draft 转换。
- 修改 `src/math_modeling_agent/linear_solver.py`：按变量域选择 GLOP/SCIP，映射 `NumVar`、`IntVar`、`BoolVar`，结果记录后端。
- 修改 `src/math_modeling_agent/linear_validator.py`：独立检查整数性与二进制 `[0,1]` 边界。
- 修改 `src/math_modeling_agent/linear_agent.py`、`data/hmml.json`：推荐并标记已实现的连续 LP 或 MILP 方法。
- 修改 `src/math_modeling_agent/evals.py`、`evals/cases.json`：为整数域建模评分，并加入一条整数规划案例。
- 扩展 `tests/test_models.py`、`tests/test_analysis_agent.py`、`tests/test_linear_solver.py`、`tests/test_linear_validator.py`、`tests/test_linear_agent.py`、`tests/test_retrieval.py`、`tests/test_evals.py` 和 `tests/test_cli.py`。
- 修改 `README.md`：列出连续、整数、二进制线性模型的变量域与求解器边界。

## 任务一：变量域的数据契约

**步骤 1：写失败测试。** 验证 `LinearVariable(name="x", unit="件", domain="integer")` 保留整数域；省略 `domain` 时仍为连续；未知域被 Pydantic 拒绝。

**步骤 2：运行红灯。**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_models.py -k linear_variable_domain -q
```

预期：当前 `LinearVariable` 没有 `domain` 字段，整数域测试失败。

**步骤 3：实现。** 在 `models.py` 增加 `domain: Literal["continuous", "integer", "binary"] = "continuous"`，保持旧 LP JSON 可按连续变量解释。

**步骤 4：复跑模型测试。**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_models.py -q
```

## 任务二：GLOP/SCIP 求解和变量域校验

**步骤 1：写失败测试。** 新建整数问题：最大化 `x`，约束 `2x <= 5` 和 `x >= 0`，`x` 为整数，期望 `x=2`、目标值 `2`、求解器为 SCIP。再建混合问题：最大化 `3x + 5y`，约束 `x + 2y <= 4`、`4x + 2y <= 12`、`x >= 0`、`y` 为二进制，期望 `x=2`、`y=1`、目标值 `11`。连续变量案例仍须使用 GLOP。

**步骤 2：运行红灯。**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_linear_solver.py -k "integer or binary or mixed" -q
```

预期：现有构建器把所有变量建为 GLOP `NumVar`，整数最优值测试失败。

**步骤 3：实现后端分流。** 变量域全为 continuous 时使用 GLOP；任一变量为 integer/binary 时使用 SCIP。continuous 使用 `NumVar(-inf, inf)`，integer 使用 `IntVar(-inf, inf)`，binary 使用 `BoolVar(name)`。后端不可用时返回 `SOLVER_UNAVAILABLE`，不要伪造变量值。`LinearSolverResult` 增加 `solver_name`。

**步骤 4：先独立测试整数校验。** validator 对 integer 检查 `abs(value - round(value)) <= tolerance`；binary 同时检查与 0 或 1 的距离不超过 tolerance。错误解必须报告具体变量名。

**步骤 5：复跑求解器和 validator 测试。**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_linear_solver.py tests\test_linear_validator.py -q
```

## 任务三：自然语言模型草稿和方法推荐

**步骤 1：写失败测试。** 分析 Draft Schema 必须要求每个变量 `domain` 为 `continuous`、`integer` 或 `binary`，并仍满足没有 `anyOf` 的结构化输出约束；`to_linear_program_problem` 必须保留域。

**步骤 2：运行红灯。**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_analysis_agent.py -k "integer or variable_domain" -q
```

**步骤 3：更新分析器。** `LinearVariableDraft` 增加必填 `domain` 字段；提示词明确用户的“整数个/0-1/是否启用”等措辞分别对应 integer/binary，变量域不清楚且影响解时追问。移除把整数/二进制线性模型列为 unsupported 的规则；非线性、多目标仍 unsupported。转换时复制 Draft 域到内部 `LinearVariable.domain`。

**步骤 4：更新 HMML。** 升级 `integer_programming` 方法卡，说明 SCIP、连续/整数/二进制混合变量及其线性单目标限制；连续问题仍优先推荐 `continuous_linear_programming`。Agent 的问题描述和目标描述必须根据变量域区分两种方法，并过滤不兼容的方法推荐。

**步骤 5：复跑分析器、转换和 HMML 检索测试。** 确认排班、连续 LP 和网络流原有 Schema 与路由不变。

## 任务四：Evals、文档和全量验收

**步骤 1：新增离线整数案例。** `evals/cases.json` 增加最大化整数生产量案例，含明确单位的题干、`expected_model`、变量域、`OPTIMAL`、已实现方法 `integer_programming`、期望解和 validator 状态。旧 Evals 案例缺少 `domain` 时只按 continuous 兼容归一化。

**步骤 2：测试端到端。** 使用 ready 的 MIP `ProblemAnalysis`，经既有 `--solve-draft` 路由，断言选择 SCIP、结果状态为 `OPTIMAL`、validator 通过；needs_clarification/unsupported 不得启动求解器。

**步骤 3：更新 README。** 说明变量域、GLOP/SCIP 自动分流、未实现的非线性和多目标边界，并给出结构化 Draft 的 PowerShell 调用方式。

**步骤 4：全量回归。**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp 'D:\Agent\.worktrees\min-cost-network-flow\.pytest-basetemp-milp' -q
```

预期：连续 LP 原测试、MILP 新测试、排班/最小费用流以及 Q1—Q5 能源园区测试全部通过；仅允许已知 OR-Tools/SWIG 弃用警告。

**步骤 5：检查并提交。** `git diff --check` 通过，工作树只包含本计划文件清单内改动，以中文提交说明提交；不合并或推送分支。

## 执行记录

- [x] 内部模型保留连续变量兼容，新增 integer/binary 域；GLOP/SCIP 自动分流并记录实际后端。
- [x] Validator 独立检查整数性、0/1 域、线性约束和目标值；SCIP 不可用时不返回伪解。
- [x] DeepSeek Draft Schema、旧 Draft 兼容、HMML 方法能力过滤、Evals 整数案例和已确认 Draft CLI 均已接通。
- [x] DeepSeek 单案例在线 Evals：`mixed_integer_linear_programming_integer_product` 通过 1/1；题干明确单位，SCIP 解为 `x=2`、目标值 2，validator 通过。
- [x] 全量测试通过：`149 passed`，仅有 3 条已知 OR-Tools/SWIG 弃用警告。
- [ ] 以中文提交本轮改动；不合并或推送分支。
