# 最小费用网络流适配器实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在保留员工排班和连续 LP 能力的基础上，增加整数单位的单商品最小费用网络流建模、求解与独立校验。

**架构：** 沿用 `ProblemAnalysis` 固定领域草稿与 `problem_family` 分流架构，新增 `minimum_cost_flow` 草稿、`MinCostFlowProblem` 内部模型及 OR-Tools `SimpleMinCostFlow` 适配器。自然语言分析只抽取显式节点供需和路线容量/成本；所有可行解必须经过独立 validator。

**技术栈：** Python 3.11+、Pydantic 2、OR-Tools Graph `SimpleMinCostFlow`、pytest、DeepSeek Structured Outputs（仅在线 Evals）。

---

## 文件职责

- `src/math_modeling_agent/models.py`：内部 `FlowNode`、`FlowArc`、`MinCostFlowProblem` 及引用/数值校验。
- `src/math_modeling_agent/analysis_agent.py`：固定 `MinCostFlowDraft` Schema、`problem_family` 扩展、领域互斥校验和 `to_minimum_cost_flow_problem()`。
- `src/math_modeling_agent/min_cost_flow_solver.py`：把内部节点/路线映射到 OR-Tools SimpleMinCostFlow，输出稳定状态、边流量和总费用。
- `src/math_modeling_agent/min_cost_flow_validator.py`：独立检查每条边的流量/容量、节点流守恒和总费用。
- `src/math_modeling_agent/min_cost_flow_agent.py`：HMML 方法检索、求解和 validator 编排。
- `src/math_modeling_agent/cli.py`：依据 `problem_family` 分流，保持既有顶层响应字段不变。
- `src/math_modeling_agent/evals.py`、`evals/cases.json`：对网络流草稿归一化，评分路线解和总费用，增加可行与不可行案例。
- `data/hmml.json`：增加已实现的单商品最小费用流方法卡。
- `tests/min_cost_flow_fixtures.py`：共享固定运输图与 ready 分析草稿，避免不同测试各自构造不一致数据。
- `tests/test_models.py`、`tests/test_analysis_agent.py`、`tests/test_cli.py`、`tests/test_evals.py`：扩展现有测试；新增 `tests/test_min_cost_flow_solver.py`、`tests/test_min_cost_flow_validator.py`、`tests/test_min_cost_flow_agent.py`。
- `docs/本地运行与密钥配置.md`：更新支持范围、PowerShell 测试与在线 Evals 命令。

### 共享测试数据

在 `tests/min_cost_flow_fixtures.py` 实现下面两个构造器和常量，供单元、Agent、分析与评分测试复用。pytest 测试目录按默认 import mode 可直接以 `from min_cost_flow_fixtures import ...` 导入。

```python
EXPECTED_ARC_FLOWS = {
    "W1_S1": 20,
    "W1_S2": 0,
    "W2_S1": 5,
    "W2_S2": 25,
}


def make_transport_problem():
    from math_modeling_agent.models import FlowArc, FlowNode, MinCostFlowProblem

    return MinCostFlowProblem(
        nodes=[
            FlowNode(node_id="W1", name="仓库一", supply=20),
            FlowNode(node_id="W2", name="仓库二", supply=30),
            FlowNode(node_id="S1", name="门店一", supply=-25),
            FlowNode(node_id="S2", name="门店二", supply=-25),
        ],
        arcs=[
            FlowArc(arc_id="W1_S1", from_node="W1", to_node="S1", capacity=20, unit_cost=2),
            FlowArc(arc_id="W1_S2", from_node="W1", to_node="S2", capacity=20, unit_cost=4),
            FlowArc(arc_id="W2_S1", from_node="W2", to_node="S1", capacity=25, unit_cost=3),
            FlowArc(arc_id="W2_S2", from_node="W2", to_node="S2", capacity=30, unit_cost=1),
        ],
        flow_unit="箱",
        cost_unit="元/箱",
    )


def make_ready_analysis():
    from math_modeling_agent.analysis_agent import (
        AnalysisSubtask,
        LinearProgramDraft,
        MinCostFlowArcDraft,
        MinCostFlowDraft,
        MinCostFlowNodeDraft,
        ProblemAnalysis,
        SchedulingDraft,
    )

    return ProblemAnalysis(
        status="ready",
        problem_family="minimum_cost_flow",
        summary="以最低费用将两仓货物送至两家门店。",
        known_facts=["供给、需求、路线容量与单位费用均已给出。"],
        missing_information=[],
        clarifying_questions=[],
        unsupported_reasons=[],
        subtasks=[
            AnalysisSubtask(
                task_id="T1",
                description="建立仓库到门店的最小费用流模型。",
                objective="满足所有供需并最小化运输总费用。",
                data_needed=[],
                depends_on=[],
                hmml_problem_query="仓库供给、门店需求、路线容量与运输费用",
                hmml_goal_query="满足全部需求并最小化总运输费用",
            )
        ],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
        linear_program_draft=LinearProgramDraft(
            variables=[],
            objective_direction="not_applicable",
            objective_terms=[],
            constraints=[],
        ),
        minimum_cost_flow_draft=MinCostFlowDraft(
            nodes=[
                MinCostFlowNodeDraft(node_id="W1", name="仓库一", supply=20),
                MinCostFlowNodeDraft(node_id="W2", name="仓库二", supply=30),
                MinCostFlowNodeDraft(node_id="S1", name="门店一", supply=-25),
                MinCostFlowNodeDraft(node_id="S2", name="门店二", supply=-25),
            ],
            arcs=[
                MinCostFlowArcDraft(arc_id="W1_S1", from_node="W1", to_node="S1", capacity=20, unit_cost=2),
                MinCostFlowArcDraft(arc_id="W1_S2", from_node="W1", to_node="S2", capacity=20, unit_cost=4),
                MinCostFlowArcDraft(arc_id="W2_S1", from_node="W2", to_node="S1", capacity=25, unit_cost=3),
                MinCostFlowArcDraft(arc_id="W2_S2", from_node="W2", to_node="S2", capacity=30, unit_cost=1),
            ],
            flow_unit="箱",
            cost_unit="元/箱",
        ),
    )


def make_transport_eval_payload():
    return {
        "analysis": make_ready_analysis().model_dump(mode="json"),
        "modeling_run": {
            "solver_result": {
                "status": "OPTIMAL",
                "arc_flows": EXPECTED_ARC_FLOWS,
                "total_cost": 80,
            },
            "validation_report": {
                "is_valid": True,
                "errors": [],
                "recomputed_total_cost": 80,
            },
            "method_recommendations": [
                {
                    "method_id": "minimum_cost_flow",
                    "implementation_status": "已实现",
                }
            ],
        },
    }
```

## 任务一：内部图模型与引用校验

**文件：** `src/math_modeling_agent/models.py`、`tests/test_models.py`

- [x] **步骤 1：先写会失败的模型测试。** 在 `tests/min_cost_flow_fixtures.py` 按“共享测试数据”一节添加固定运输问题构造器和 ready 分析构造器；在 `tests/test_models.py` 测试 W1/W2 供给 20/30、S1/S2 净需求 -25/-25 及四条路线可正确构造；再测试重复节点/边 ID、未知端点、负容量、小数容量或费用被拒绝。供需不平衡应允许进入模型，以便由求解器报告 `INFEASIBLE`。

```python
def test_network_flow_rejects_unknown_arc_endpoint() -> None:
    from pydantic import ValidationError
    import pytest
    from math_modeling_agent.models import MinCostFlowProblem
    from min_cost_flow_fixtures import make_transport_problem

    raw = make_transport_problem().model_dump()
    raw["arcs"][0]["to_node"] = "UNKNOWN"
    with pytest.raises(ValidationError):
        MinCostFlowProblem.model_validate(raw)
```
- [x] **步骤 2：运行红灯测试。**

```powershell
$testTemp = Join-Path (Resolve-Path .\.venv).Path ('pytest-tmp-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testTemp | Out-Null
& .\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $testTemp tests\test_models.py -q
```

预期：新增测试因 `MinCostFlowProblem` 尚不存在而失败。

- [x] **步骤 3：实现最小内部模型。** 添加下列数据结构，并在 `MinCostFlowProblem` 的 `model_validator` 中检查 node/arc 编号唯一、边端点存在且不能自连。容量必须是非负严格整数；净供给和单位费用必须是严格整数；并行边允许但必须有不同 `arc_id`。

```python
class FlowNode(BaseModel):
    node_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    supply: int = Field(strict=True)


class FlowArc(BaseModel):
    arc_id: str = Field(min_length=1)
    from_node: str = Field(min_length=1)
    to_node: str = Field(min_length=1)
    capacity: int = Field(ge=0, strict=True)
    unit_cost: int = Field(strict=True)


class MinCostFlowProblem(BaseModel):
    nodes: list[FlowNode] = Field(min_length=2)
    arcs: list[FlowArc] = Field(min_length=1)
    flow_unit: str = Field(min_length=1)
    cost_unit: str = Field(min_length=1)
```

- [x] **步骤 4：运行模型测试确认通过。** 重用步骤 2 命令；预期全部 `tests/test_models.py` 通过。
- [x] **步骤 5：提交模型层。** 使用中文提交说明 `增加最小费用流内部模型`。

## 任务二：SimpleMinCostFlow 求解器

**文件：** 新建 `src/math_modeling_agent/min_cost_flow_solver.py`、`tests/test_min_cost_flow_solver.py`

- [x] **步骤 1：先写求解器失败测试。** 使用设计文档中的四节点例子，断言状态为 `OPTIMAL`、路线流量为 W1→S1=20、W2→S1=5、W2→S2=25、W1→S2=0，且总费用为 80。再加总供给不平衡与容量不足两个用例，断言 `INFEASIBLE`、路线流量为空、总费用为 `None`。

```python
def test_solver_finds_minimum_cost_warehouse_delivery() -> None:
    from math_modeling_agent.min_cost_flow_solver import solve_min_cost_flow
    from min_cost_flow_fixtures import EXPECTED_ARC_FLOWS, make_transport_problem

    result = solve_min_cost_flow(make_transport_problem())

    assert result.status == "OPTIMAL"
    assert result.arc_flows == EXPECTED_ARC_FLOWS
    assert result.total_cost == 80
```
- [x] **步骤 2：运行新测试确认红灯。**

```powershell
$testTemp = Join-Path (Resolve-Path .\.venv).Path ('pytest-tmp-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testTemp | Out-Null
& .\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $testTemp tests\test_min_cost_flow_solver.py -q
```

预期：因 `min_cost_flow_solver.py` 尚不存在而失败。

- [x] **步骤 3：实现求解器。** 使用 `from ortools.graph.python import min_cost_flow`、`SimpleMinCostFlow()`、`add_arc_with_capacity_and_unit_cost()`、`set_node_supply()` 和 `solve()`。以输入节点顺序建立 node_id→index 映射，以输入边顺序保留 arc_id→solver-index 映射。只在 `OPTIMAL` 状态读取 flows 与 optimal_cost；`INFEASIBLE`、`BAD_COST_RANGE`、`BAD_CAPACITY_RANGE`、solver unavailable 和未知状态均不返回伪解。
- [x] **步骤 4：重跑求解器测试。** 重用步骤 2 命令；预期最优与不可行状态均正确。
- [x] **步骤 5：提交求解器层。** 使用中文提交说明 `实现 SimpleMinCostFlow 求解器`。

## 任务三：独立网络流 validator

**文件：** 新建 `src/math_modeling_agent/min_cost_flow_validator.py`、`tests/test_min_cost_flow_validator.py`

- [x] **步骤 1：先写 validator 测试。** 正确路线流量必须通过；分别篡改超容量、节点供需守恒、负流量、缺失边/多余边和总费用时，断言报告包含具体错误。

```python
def test_validator_accepts_the_optimal_transport_plan() -> None:
    from math_modeling_agent.min_cost_flow_validator import validate_min_cost_flow_solution
    from min_cost_flow_fixtures import EXPECTED_ARC_FLOWS, make_transport_problem

    report = validate_min_cost_flow_solution(
        make_transport_problem(), EXPECTED_ARC_FLOWS, reported_total_cost=80
    )

    assert report.is_valid is True
    assert report.errors == []
    assert report.recomputed_total_cost == 80
```
- [x] **步骤 2：运行新测试确认红灯。**

```powershell
$testTemp = Join-Path (Resolve-Path .\.venv).Path ('pytest-tmp-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testTemp | Out-Null
& .\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $testTemp tests\test_min_cost_flow_validator.py -q
```

预期：因 validator 模块尚不存在而失败。

- [x] **步骤 3：实现报告和校验函数。** 定义 `MinCostFlowValidationReport(is_valid, errors, recomputed_total_cost)` 及 `validate_min_cost_flow_solution(problem, arc_flows, reported_total_cost)`。对每条边检查 flow 为非负整数且不超过 capacity；累计 `outgoing - incoming`，与节点的 signed supply 比较；重算 `sum(flow * unit_cost)` 并与 solver 总费用核对。
- [x] **步骤 4：运行 validator 测试确认通过。** 重用步骤 2 命令；预期所有合法/篡改用例均通过断言。
- [x] **步骤 5：提交 validator 层。** 使用中文提交说明 `增加最小费用流独立校验器`。

## 任务四：固定分析草稿 Schema 和转换

**文件：** `src/math_modeling_agent/analysis_agent.py`、`tests/test_analysis_agent.py`

- [x] **步骤 1：先写 Schema 与转换测试。** 断言 JSON Schema 仍不含 `anyOf`；`ProblemAnalysis` 有必填 `problem_family` 与固定 `minimum_cost_flow_draft`；ready 网络流可转换为 `MinCostFlowProblem`；inactive draft 为空；排班/LP ready 与网络流 draft 互斥。

```python
def test_network_flow_analysis_schema_is_fixed_and_converts() -> None:
    import json
    from math_modeling_agent.analysis_agent import ProblemAnalysis, to_minimum_cost_flow_problem
    from min_cost_flow_fixtures import make_ready_analysis

    schema = ProblemAnalysis.model_json_schema()
    assert "anyOf" not in json.dumps(schema)
    assert "minimum_cost_flow_draft" in schema["required"]

    problem = to_minimum_cost_flow_problem(make_ready_analysis())
    assert len(problem.nodes) == 4
    assert len(problem.arcs) == 4
    assert problem.flow_unit == "箱"
```
- [x] **步骤 2：运行红灯测试。**

```powershell
$testTemp = Join-Path (Resolve-Path .\.venv).Path ('pytest-tmp-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testTemp | Out-Null
& .\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $testTemp tests\test_analysis_agent.py -q
```

预期：新测试因缺少 `minimum_cost_flow` family 和 draft 类型而失败。

- [x] **步骤 3：实现固定草稿和转换。** 新增 `MinCostFlowNodeDraft(node_id, name, supply)`、`MinCostFlowArcDraft(arc_id, from_node, to_node, capacity, unit_cost)`、`MinCostFlowDraft(nodes, arcs, flow_unit, cost_unit)`。inactive draft 使用 `nodes=[]`、`arcs=[]`、两个单位字段 `not_applicable`。在 `ProblemAnalysis` 中加入 `minimum_cost_flow` family；ready 时只允许对应领域 draft 非空；非 ready 时所有领域 draft 为空。新增 `to_minimum_cost_flow_problem()` 并更新全部测试构造。
- [x] **步骤 4：更新 DeepSeek 分析提示词。** 只支持整数单商品最小费用流；明确正 supply、负 demand、0 中转节点；不推断节点、路线、容量或费用；费用小数必须精确换算为已说明的最小费用单位，不能精确换算时追问；max-flow、部分供需、多商品和车辆路径返回 unsupported。
- [x] **步骤 5：运行分析器测试确认通过。** 重用步骤 2 命令；预期 schema 无 `anyOf` 且既有排班/LP 用例不回归。
- [x] **步骤 6：提交分析 Schema。** 使用中文提交说明 `扩展最小费用流结构化分析`。

## 任务五：领域 Agent、HMML 与 CLI 路由

**文件：** 新建 `src/math_modeling_agent/min_cost_flow_agent.py`、修改 `data/hmml.json`、`src/math_modeling_agent/cli.py`、`tests/test_min_cost_flow_agent.py`、`tests/test_cli.py`

- [x] **步骤 1：先写 Agent 与 CLI 测试。** Agent 测试断言生产计划最小费用流为 OPTIMAL 且 validator 有效；CLI 注入 ready 分析时按 `problem_family="minimum_cost_flow"` 进入新适配器，clarification/unsupported 不调用任何 solver。

```python
def test_min_cost_flow_agent_solves_and_validates_transport_plan() -> None:
    from math_modeling_agent.min_cost_flow_agent import run_min_cost_flow_modeling
    from min_cost_flow_fixtures import make_transport_problem

    result = run_min_cost_flow_modeling(make_transport_problem())

    assert result.solver_result.status == "OPTIMAL"
    assert result.solver_result.total_cost == 80
    assert result.validation_report is not None
    assert result.validation_report.is_valid is True
    assert any(
        item.method_id == "minimum_cost_flow"
        and item.implementation_status == "已实现"
        for item in result.method_recommendations
    )
```
- [x] **步骤 2：运行新测试确认红灯。**

```powershell
$testTemp = Join-Path (Resolve-Path .\.venv).Path ('pytest-tmp-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testTemp | Out-Null
& .\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $testTemp tests\test_min_cost_flow_agent.py tests\test_cli.py -q
```

预期：因 Agent 和 CLI 新路由尚不存在而失败。

- [x] **步骤 3：增加 HMML 方法卡。** 在 `operations_research` 的网络流子领域新增 `minimum_cost_flow` 方法，`implementation_status="已实现"`、`solver="OR-Tools SimpleMinCostFlow"`，关键词覆盖运输、供给、需求、路线容量、单位费用、最小总费用。
- [x] **步骤 4：实现 `run_min_cost_flow_modeling(problem)`。** 建立与其他 Agent 一致的 dataclass，先用 `HMMLRetriever` 推荐方法，再调用 solver；只有 `OPTIMAL` 才调用 validator。响应包含 `solver_result`、`validation_report`、`method_recommendations`。
- [x] **步骤 5：修改 CLI 分流。** 为 ready 分析增加 `minimum_cost_flow` 分支：调用 `to_minimum_cost_flow_problem()` 和 `run_min_cost_flow_modeling()`；现有排班和 LP 分支保持原样。
- [x] **步骤 6：运行 Agent/CLI/HMML 回归测试。** 重用步骤 2 命令；预期新领域与旧领域路由均通过。
- [x] **步骤 7：提交领域路由。** 使用中文提交说明 `接入最小费用流领域 Agent`。

## 任务六：Evals 与数据集升级

**文件：** `src/math_modeling_agent/evals.py`、`evals/cases.json`、`tests/test_evals.py`

- [x] **步骤 1：先写评分器测试。** 网络流 draft 比较不受 nodes/arcs 列表顺序和生成的 `arc_id` 顺序影响；测试最优状态、已实现方法、validator、每条期望路线流量与总费用；错误路线或错误总费用必须失败。

```python
def test_eval_scores_expected_network_routes_and_total_cost() -> None:
    from math_modeling_agent.evals import evaluate_case

    case = {
        "id": "minimum_cost_flow_warehouse_delivery",
        "family": "transportation",
        "expected_analysis_status": "ready",
        "expected_solver_statuses": ["OPTIMAL"],
        "expected_method_id": "minimum_cost_flow",
        "expected_validation": "valid",
        "expected_route_flows": [
            ["W1", "S1", 20], ["W1", "S2", 0],
            ["W2", "S1", 5], ["W2", "S2", 25]
        ],
        "expected_total_cost": 80,
    }
    from min_cost_flow_fixtures import make_transport_eval_payload
    payload = make_transport_eval_payload()

    result = evaluate_case(case, payload)

    assert result["passed"] is True
```
- [x] **步骤 2：运行红灯测试。**

```powershell
$testTemp = Join-Path (Resolve-Path .\.venv).Path ('pytest-tmp-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testTemp | Out-Null
& .\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $testTemp tests\test_evals.py -q
```

预期：新检查因评分器尚不识别网络流 draft/route flows 而失败。

- [x] **步骤 3：实现 draft 归一化和 route-flow 评分。** 忽略任意生成的 `arc_id`，但保留节点编号、端点、容量、单位费用和单位；通过分析 draft 的 arc ID→端点映射把 solver flows 转成 `(from_node,to_node)` 路线并比较；整数费用与整数流量必须精确匹配。
- [x] **步骤 4：升级 Evals 案例。** 将 `unsupported_transportation_network_flow` 替换为 ready 案例 `minimum_cost_flow_warehouse_delivery`，expected_model 包含四个节点、四条边、`flow_unit="箱"`、`cost_unit="元/箱"`；期望 `OPTIMAL`、总费用 80、四条路线流量 W1→S1=20、W1→S2=0、W2→S1=5、W2→S2=25。增加 `minimum_cost_flow_infeasible_capacity`，使可用出边容量小于总需求，期望 `INFEASIBLE` 和 validator `not_run`；再增加 `unsupported_multicommodity_flow`，确保多商品流仍返回 unsupported。案例总数更新为 11。
- [x] **步骤 5：扩展在线 Evals dispatcher。** ready 网络流分析转换为 `MinCostFlowProblem` 并调用 `run_min_cost_flow_modeling`；离线默认仍不调用 DeepSeek。
- [x] **步骤 6：运行 Evals 测试。** 重用步骤 2 命令；预期 dataset 为 11 个唯一案例且 ready 领域用例声明模型、状态、方法和 validator 期望。
- [ ] **步骤 7：提交 Evals。** 使用中文提交说明 `增加最小费用流在线与离线评测`。

## 任务七：文档、完整测试与发布

**文件：** `docs/本地运行与密钥配置.md`，全项目测试及 Git

- [x] **步骤 1：更新中文使用文档。** 说明排班、连续 LP、整数最小费用网络流三个已实现领域和各自边界；提供 PowerShell 的离线 Evals、完整 pytest 和单案例 `--live` 命令；解释费用单位换算。
- [x] **步骤 2：运行完整 PowerShell pytest。**

```powershell
$testTemp = Join-Path (Resolve-Path .\.venv).Path ('pytest-tmp-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testTemp | Out-Null
& .\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $testTemp -q
```

预期：所有排班、LP 与网络流测试通过；允许报告已知的 OR-Tools SWIG deprecation warnings，但不得有失败。

- [x] **步骤 3：验证离线和在线 Evals。** 在隔离 worktree 源码上运行离线 Evals，确认 11 个案例有效且 `api_calls=0`；经用户已授权的 DeepSeek 测试分别运行仓库配送可行、容量不足和多商品 unsupported 案例，报告不得含密钥或原始 API 响应。
- [x] **步骤 4：检查并提交文档/回归。** 运行 `git diff --check`，确认 `.env` 不在 `git ls-files`、`git check-ignore -v .env` 命中；用中文提交说明 `完善最小费用流文档与回归`。
- [ ] **步骤 5：完成分支集成。** 全量验证通过后，按 `finishing-a-development-branch` 流程检测 worktree 与 base branch，并请用户选择合并到 main、本地保留 feature 分支或其他集成方式；依选择执行后再次验证并同步 GitHub。
