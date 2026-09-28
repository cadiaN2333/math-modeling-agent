# 运筹优化 Agent 分层 Evals 实施计划

> **执行说明：** 按任务逐项实施，复选框跟踪进度；所有生产代码都先有失败测试。

**目标：** 为当前排班 MVP 建立 8 个可重复案例、离线评分测试和显式在线评测入口，并将线性规划与运输/网络流作为当前不支持边界。

**架构：** `evals/cases.json` 保存带领域标签的请求和期望；`math_modeling_agent.evals` 提供纯离线评分、数据集校验和可选在线执行。没有 `--live` 时不调用分析器、不读取 `.env`；在线模式只在指定后调用 DeepSeek，ready 排班再由现有求解器和 validator 处理。

**技术栈：** Python 标准库、Pydantic 结果模型、OR-Tools 现有排班求解器、pytest、PowerShell。

---

## 文件职责

- 修改 `src/math_modeling_agent/analysis_agent.py`：明确当前只支持员工排班，非排班优化问题需返回 unsupported。
- 新增 `evals/cases.json`：8 个排班、能力边界和求解案例。
- 新增 `src/math_modeling_agent/evals.py`：数据集加载/校验、单案例评分、离线预检和 `--live` 在线评测入口。
- 新增 `tests/test_evals.py`：评分器、案例定义、在线/离线分支和报告统计测试；所有模型调用均用假分析器替代。
- 修改 `docs/本地运行与密钥配置.md`：记录 PowerShell 离线命令和需显式授权的在线命令。

## 任务一：测试当前能力边界

**文件：** 修改 `tests/test_analysis_agent.py`。

- [ ] **步骤 1：先添加失败测试。**

```python
def test_analysis_instructions_limit_current_domain_to_scheduling() -> None:
    from math_modeling_agent.analysis_agent import ANALYSIS_INSTRUCTIONS

    assert "只支持员工排班" in ANALYSIS_INSTRUCTIONS
    assert "运输/网络流" in ANALYSIS_INSTRUCTIONS
    assert "unsupported" in ANALYSIS_INSTRUCTIONS
```

- [ ] **步骤 2：验证 RED。**

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider .\tests\test_analysis_agent.py::test_analysis_instructions_limit_current_domain_to_scheduling -q
```

预期：失败，因为提示词还没有清楚声明非排班模型类型超出当前能力。

- [ ] **步骤 3：在 `ANALYSIS_INSTRUCTIONS` 增加规则。** 明确生产计划线性规划、运输/网络流及其他非员工排班问题必须返回 unsupported，不得转换为员工排班草稿。
- [ ] **步骤 4：重跑同一测试。** 预期通过。

## 任务二：先测试纯函数评分器

**文件：** 新增 `tests/test_evals.py`，新增 `src/math_modeling_agent/evals.py`。

- [ ] **步骤 1：先写状态误判测试。**

```python
def test_eval_case_fails_when_analysis_status_is_wrong() -> None:
    from math_modeling_agent.evals import evaluate_case

    case = {
        "id": "lp_not_supported",
        "expected_analysis_status": "unsupported",
    }
    payload = {
        "analysis": {"status": "ready"},
        "modeling_run": None,
    }

    result = evaluate_case(case, payload)

    assert result["passed"] is False
    assert result["checks"][0]["name"] == "analysis_status"
```

- [ ] **步骤 2：验证 RED。**

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider .\tests\test_evals.py::test_eval_case_fails_when_analysis_status_is_wrong -q
```

预期：因 `evaluate_case` 尚不存在而失败。

- [ ] **步骤 3：实现评分器接口。**

```python
def evaluate_case(case: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """根据案例期望对分析结果及可选建模结果逐项评分。"""
    analysis = payload["analysis"]
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, expected: Any, actual: Any) -> None:
        checks.append(
            {"name": name, "passed": passed, "expected": expected, "actual": actual}
        )

    actual_status = analysis.get("status")
    expected_status = case["expected_analysis_status"]
    record("analysis_status", actual_status == expected_status, expected_status, actual_status)

    for index, group in enumerate(case.get("required_text_groups", [])):
        text_fields = []
        for field_name in group["fields"]:
            value = analysis.get(field_name, "")
            text_fields.extend(value if isinstance(value, list) else [value])
        searchable_text = " ".join(str(item) for item in text_fields).casefold()
        alternatives = group["any_of"]
        matched = any(term.casefold() in searchable_text for term in alternatives)
        record(f"required_text_{index + 1}", matched, alternatives, matched)

    expected_model = case.get("expected_model")
    if expected_model is not None:
        actual_model = analysis.get("scheduling_draft")
        record("structured_model", actual_model == expected_model, expected_model, actual_model)

    run = payload.get("modeling_run")
    expected_solver_statuses = case.get("expected_solver_statuses")
    if expected_solver_statuses is None:
        record("solver_not_run", run is None, "not_run", "run" if run else "not_run")
    else:
        solver = (run or {}).get("solver_result") or {}
        actual_solver_status = solver.get("status")
        record(
            "solver_status",
            actual_solver_status in expected_solver_statuses,
            expected_solver_statuses,
            actual_solver_status,
        )

        expected_method = case.get("expected_method_id")
        if expected_method:
            method_ids = {
                item.get("method_id")
                for item in (run or {}).get("method_recommendations", [])
            }
            record("implemented_method", expected_method in method_ids, expected_method, sorted(method_ids))

        validation_expectation = case.get("expected_validation")
        if validation_expectation == "valid":
            report = (run or {}).get("validation_report")
            record("validator", bool(report and report.get("is_valid")), True, report)
        elif validation_expectation == "not_run":
            report = (run or {}).get("validation_report")
            record("validator", report is None, "not_run", report)

    return {
        "case_id": case["id"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
```

- [ ] **步骤 4：测试评分器边界。** 关键词组必须声明要搜索的结果字段（例如 `unsupported_reasons`），避免模型仅把用户原文复述到 `known_facts` 就错误通过。为 ready 可行、ready 无解、追问缺失项、unsupported 非排班请求分别添加测试；不要求模型的自由文本逐字相同。
- [ ] **步骤 5：运行 `tests/test_evals.py`。** 预期所有评分器单测通过，不连接网络。

## 任务三：添加 8 个版本化案例

**文件：** 新增 `evals/cases.json`，修改 `tests/test_evals.py`。

- [ ] **步骤 1：添加案例集结构测试。** 断言 JSON 有版本号、案例 ID 唯一、恰有 8 个首版案例，所有状态值有效；ready 案例都有 solver 期望，非 ready 案例不要求求解。
- [ ] **步骤 2：新增 8 个案例：** 缺少最大工时需追问；完整三人两班可行；无人具备急救技能导致无解；8 小时班次但员工最多工作 7 小时导致无解；班次冲突要求 unsupported；工资/公平性要求 unsupported；生产计划 LP unsupported；运输/网络流 unsupported。
- [ ] **步骤 3：为非排班案例加入关键词组。** 至少要求结果说明该问题不属于当前员工排班支持范围；不能仅因 status 为 unsupported 就视为通过。完整 ready 案例使用 `expected_model` 保存期望员工、技能、工时、班次和覆盖数据，由评分器比较结构化字段，而不比较解释文字。
- [ ] **步骤 4：运行案例集测试。** 预期 8 个案例均可加载，ID 无重复，结构符合评分器所需字段。

## 任务四：添加显式在线评测入口

**文件：** 修改 `src/math_modeling_agent/evals.py` 和 `tests/test_evals.py`。

- [ ] **步骤 1：增加命令行入口。** 无参数时只校验案例文件并报告案例数量；不调用 `analyze_problem`，不读取 `.env`。`--live` 才逐案例调用 DeepSeek；可用 `--case-id <id>` 限定单个案例。
- [ ] **步骤 2：逐例执行。** 在线模式调用 `analyze_problem(request)`；若 ready，则调用 `to_scheduling_problem` 和 `run_modeling`；其他状态不启动求解。每例异常只记录案例 ID 与异常类型，禁止打印密钥或原始响应。
- [ ] **步骤 3：汇总报告。** 输出 JSON，包含逐案例状态、评分检查、错误数、总通过率、分析状态正确率、求解类别正确率、已实现方法命中率及 validator 通过率；任一案例失败时返回退出码 1。
- [ ] **步骤 4：用假分析器测试两种模式。** 无 `--live` 时断言分析器未被调用；指定 `--live --case-id` 时用 mock 返回固定分析结果，不访问真实网络，并检查 CLI 汇总。
- [ ] **步骤 5：运行在线入口单测。** 预期默认模式零 API 调用；live 分支由假分析器覆盖。

## 任务五：文档、全量测试与提交

**文件：** 修改 `docs/本地运行与密钥配置.md`。

- [ ] **步骤 1：写明 PowerShell 用法。** 普通离线校验命令 `python -m math_modeling_agent.evals` 不调用 API；只有显式 `python -m math_modeling_agent.evals --live --case-id <id>` 才发起模型请求。明确批量 `--live` 会按案例数消耗 API 配额。
- [ ] **步骤 2：运行完整测试。** 在 `%TEMP%` 权限受限时为 pytest 指定虚拟环境内的唯一临时目录；预期完整测试通过且无网络请求。
- [ ] **步骤 3：检查 Git。** 确认 `.env` 未暂存，再提交代码、案例和文档；提交说明使用中文。
