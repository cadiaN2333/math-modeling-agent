# 线性模型情景重算与差异分析实施计划

> **For agentic workers:** 在 `feature/min-cost-network-flow` 隔离工作树内执行，按测试先行实现并验证；用户已授权在隔离分支直接实施并推送 GitHub。

**目标：** 对同一线性模型的基准方案和多个约束情景分别求解、独立验证，并给出可解释的目标值与变量变化。

**架构：** 采用一个基准 `LinearProgramProblem` 加多个命名情景，每个情景保存完整结构化模型。情景必须保留基准变量名称、单位和 domain；约束与目标系数可变化。只有目标方向和目标系数完全相同时才计算目标值差异；否则照常求解并说明目标值不可直接比较。所有求解复用现有 GLOP/SCIP 适配器与 validator。

**技术栈：** Python、Pydantic、OR-Tools GLOP/SCIP、现有线性模型求解与校验、pytest。

---

## 文件职责

- 新建 `src/math_modeling_agent/scenario_models.py`：情景输入结构及变量签名校验。
- 新建 `src/math_modeling_agent/scenario_analysis.py`：运行基准与情景模型、计算安全的差异工件。
- 修改 `src/math_modeling_agent/cli.py`：新增 `--scenario-file` JSON 输入入口。
- 新建 `tests/test_scenario_analysis.py`、扩展 `tests/test_cli.py`。
- 修改 `README.md`：解释情景模型的可比条件、状态和 PowerShell 用法。

## 任务一：情景输入契约

- [x] **步骤 1：写失败测试。** 一个请求包含 `base_problem` 和 1—20 个 `{scenario_id, description, problem}` 情景；重复 ID、变量名/单位/domain 与基准不一致都必须被拒绝；允许情景目标系数与约束变化。
- [x] **步骤 2：运行红灯。**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_scenario_analysis.py -k request_model -q
```

预期：`scenario_models.py` 和情景请求类型尚不存在。

- [x] **步骤 3：实现输入模型。**

```python
class LinearScenario(BaseModel):
    scenario_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    problem: LinearProgramProblem


class ScenarioAnalysisRequest(BaseModel):
    base_problem: LinearProgramProblem
    scenarios: list[LinearScenario] = Field(min_length=1, max_length=20)
```

在 `model_validator(mode="after")` 中检查 scenario_id 唯一，且每个情景与基准的 `(name, unit, domain)` 列表相同；不要要求目标系数相同。

- [x] **步骤 4：复跑请求模型测试。**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_scenario_analysis.py -k request_model -q
```

## 任务二：情景求解与差异工件

- [x] **步骤 1：写失败测试。** 基准模型 `max 3x`、`0<=x<=10` 得到 `x=10`；场景 S1 将容量改成 `x<=8`，得到 `x=8`、目标差 `-6`、变量差 `-2`。S2 设 `x>=10` 且 `x<=4`，状态为 `INFEASIBLE`，不输出伪造的变量/目标差。S3 改变目标系数，仍求解并报告 `objective_delta=None` 与不可比原因，但变量差可计算。
- [x] **步骤 2：运行红灯。**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_scenario_analysis.py -k scenario_delta -q
```

预期：`run_scenario_analysis` 尚不存在。

- [x] **步骤 3：实现计算流程。** 使用 `run_linear_modeling` 分别求解基准和每个情景。仅当两边状态为 `OPTIMAL`/`FEASIBLE` 且 validator 通过时计算变量差；仅当目标方向与按变量名归一化后的目标系数相同，且两个解都有效时计算 `scenario_objective - base_objective`。相对差定义为 `100 * delta / abs(base_objective)`；基准目标为零时填 `None`。任一解不可用时，差值填 `None` 并写明原因。

结果工件至少包含基准求解结果、每个 scenario_id/description、solver_result、validation_report、variable_deltas、objective_delta、relative_objective_delta_percent 和 objective_comparison_note。

- [x] **步骤 4：复跑情景求解测试。**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_scenario_analysis.py -q
```

## 任务三：CLI 场景文件入口

- [x] **步骤 1：写失败测试。** CLI 读取含连续模型和容量情景的 JSON，输出基准/场景状态与差值；非法 JSON、变量签名不一致返回退出码 `2`；不可行情景显示 `INFEASIBLE` 但不生成数值差。
- [x] **步骤 2：实现 `--scenario-file <路径>`。** 将参数加入现有互斥输入组；用 `ScenarioAnalysisRequest.model_validate_json` 或 `model_validate(json.loads(...))` 复核输入；调用 `run_scenario_analysis` 并用统一 JSON 输出函数写结果。求解器不可用或可行解未通过 validator 时返回 `1`；输入文件错误返回 `2`；合法无解情景本身不算 CLI 错误。
- [x] **步骤 3：运行 CLI 定向测试。**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_cli.py -k scenario_file -q
```

## 任务四：文档、全量验证与推送

- [x] **步骤 1：README 说明。** 记录完整模型输入格式、变量签名/目标可比规则、不可行情景如何报告和运行命令：

```powershell
python -m math_modeling_agent.cli --scenario-file .\scenarios.json
```

- [x] **步骤 2：全量回归并检查离线 Evals。**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp 'D:\Agent\.worktrees\min-cost-network-flow\.pytest-basetemp-scenario' -q
& D:\Agent\.venv\Scripts\python.exe -m math_modeling_agent.evals
```

全量测试必须通过；离线 Evals 保持 `api_calls=0`，现有 12 个案例继续有效。

- [x] **步骤 3：提交并推送。** `git diff --check` 通过后，用中文提交说明提交；推送当前 `feature/min-cost-network-flow` 到 `origin`，不合并到 main、不强推。

## 执行记录

- [x] 情景请求校验变量签名、scenario_id 唯一和情景数量上限。
- [x] 基准与情景分别求解并调用独立 validator；只对同目标函数的可验证解计算目标差，不可行情景不生成伪变量差。
- [x] CLI `--scenario-file`、README 情景 JSON 示例和错误处理已接通。
- [x] 全量测试 `162 passed`；离线 Evals 为 12 个案例、`api_calls=0`。
- [x] 提交并推送 GitHub（提交 `90294d3`）。
- [x] 复核修正：拒绝线性模型未知字段；目标系数必须精确相同才比较目标值；仅基准目标严格为零时省略相对百分比；无效 UTF-8 情景文件按输入错误返回退出码 `2`。
- [x] 情景请求、情景和线性模型各层级均拒绝未知字段；复核计划中类型名与实现一致。
