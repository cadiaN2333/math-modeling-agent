# 统一优化 IR、RAG 与 LangChain 实施计划

> **执行说明：** 在隔离分支 `feature/min-cost-network-flow` 中按测试先行逐项实现；各阶段独立提交，最后普通推送，不合并主分支。

**目标：** 建立覆盖当前优化问题族的求解器中立 IR、可追溯的混合 RAG 知识服务和 LangChain 编排适配层，不增加新的求解问题类型。

**架构：** 领域模型先由编译器转换为带来源和单位的 `OptimizationIR`，能力注册表再选择现有 GLOP、SCIP、CP-SAT 或最小费用流后端。公共应用服务管理草稿、用户确认、求解与验证；CLI 和 LangChain 共用该服务。RAG 检索 HMML 与经过审核的本地知识卡，只提供带来源的建议，不自动写入未经确认的模型。

**技术栈：** Python、Pydantic、OR-Tools、DeepSeek OpenAI 兼容接口、LangChain `create_agent`、LangChain Chroma、可配置的中文嵌入模型、pytest。

---

## 文件职责

| 路径 | 职责 |
| --- | --- |
| `src/math_modeling_agent/optimization_ir.py` | IR 变量、线性 formulation、网络流 formulation、证据引用及结构校验。 |
| `src/math_modeling_agent/optimization_compilers.py` | 排班、LP/MILP、最小费用流和能源园区领域模型到 IR 的编译及解映射。 |
| `src/math_modeling_agent/optimization_registry.py` | 后端能力声明、后端选择、通用求解结果结构和不支持能力的错误。 |
| `src/math_modeling_agent/optimization_backends.py` | 通过 IR 调用现有 GLOP/SCIP、CP-SAT 和 OR-Tools SimpleMinCostFlow。 |
| `src/math_modeling_agent/modeling_service.py` | 公共分析、检索、草稿确认、IR 编译、求解、验证和报告服务。 |
| `src/math_modeling_agent/knowledge_models.py` | 知识文档、分块和带来源检索证据的结构。 |
| `src/math_modeling_agent/knowledge_service.py` | 知识清单加载、索引构建、关键词/向量混合检索与证据排序。 |
| `scripts/build_knowledge_index.py` | 显式构建或重建本地 RAG 索引；模型权重只在执行该命令时按配置加载。 |
| `.gitignore` | 排除本地 Chroma 索引、缓存和下载到项目目录的模型文件。 |
| `src/math_modeling_agent/langchain_adapter.py` | 可选 LangChain `create_agent` 工具与核心服务的薄适配。 |
| `data/knowledge/manifest.json`、`data/knowledge/cards/*.md` | 经过审核的运筹知识卡和来源清单；不得放入用户原始题目附件或密钥。 |
| `tests/test_optimization_ir.py` | IR 字段、引用、边界和 formulation 校验。 |
| `tests/test_optimization_compilers.py` | 领域到 IR 的语义映射和解回写。 |
| `tests/test_optimization_registry.py` | 后端能力选择和无兼容后端错误。 |
| `tests/test_modeling_service.py` | 状态流转、确认摘要和求解/验证闭环。 |
| `tests/test_knowledge_service.py` | 分块、过滤、混合排序、引用完整性和知识来源边界。 |
| `tests/test_langchain_adapter.py` | LangChain 工具 schema、应用服务调用和确认守门。 |
| `tests/test_energy_park_ir.py` | 问题四 IR 编译、储能状态约束和独立能源校验。 |
| `tests/test_knowledge_eval.py`、`evals/knowledge_cases.json` | 人工标注的检索问题、期望来源和离线检索指标。 |

## 任务一：求解器中立 IR

- [x] **步骤 1：写 IR 红灯测试。** 测试合法线性 IR 与网络流 IR；拒绝重复变量/约束编号、表达式引用未声明变量、非有限系数、上下界倒置、未知字段和不支持的 schema 版本。

```python
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from math_modeling_agent.models import FlowArc, FlowNode


def make_ir_with_unknown_variable() -> dict[str, object]:
    return {
        "schema_version": "1",
        "problem_id": "bad-reference",
        "problem_family": "linear_programming",
        "formulation": {
            "kind": "linear",
            "variables": [
                {
                    "variable_id": "x",
                    "name": "产量",
                    "unit": "件",
                    "domain": "continuous",
                }
            ],
            "objective": {
                "direction": "maximize",
                "expression": {
                    "terms": [{"variable_id": "missing", "coefficient": 1}],
                    "constant": 0,
                },
            },
            "constraints": [],
        },
        "evidence": [],
        "assumptions": [],
    }


def test_ir_rejects_unknown_variable_reference() -> None:
    from math_modeling_agent.optimization_ir import OptimizationIR

    with pytest.raises(ValidationError, match="未声明"):
        OptimizationIR.model_validate(make_ir_with_unknown_variable())
```

- [x] **步骤 2：运行红灯。**

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_optimization_ir.py -q
```

预期：因 `optimization_ir.py` 和 IR 类型尚不存在而失败。

- [x] **步骤 3：实现 IR 契约。** 线性 formulation 表达连续/整数/二进制变量、上下界、单位、仿射目标和 `<=`、`>=`、`==` 约束；网络流 formulation 保留节点、供需、弧容量、弧费用和流量单位。IR 顶层记录 schema 版本、问题族、证据引用与待确认假设，不包含 OR-Tools 对象或可执行代码。

```python
class IRVariable(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variable_id: str
    name: str
    unit: str
    domain: Literal["continuous", "integer", "binary"]
    lower_bound: float | None = None
    upper_bound: float | None = None
    source_fact_ids: list[str] = Field(default_factory=list)


class IRTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variable_id: str
    coefficient: float


class IRAffineExpression(BaseModel):
    terms: list[IRTerm]
    constant: float = 0


class IRObjective(BaseModel):
    direction: Literal["minimize", "maximize"]
    expression: IRAffineExpression
    unit: str


class IRConstraint(BaseModel):
    constraint_id: str
    expression: IRAffineExpression
    relation: Literal["<=", ">=", "=="]
    rhs: float
    constant: float = 0
    unit: str
    source_fact_ids: list[str] = Field(default_factory=list)


class LinearFormulationIR(BaseModel):
    kind: Literal["linear"]
    variables: list[IRVariable]
    objective: IRObjective
    constraints: list[IRConstraint]


class NetworkFlowFormulationIR(BaseModel):
    kind: Literal["network_flow"]
    nodes: list[FlowNode]
    arcs: list[FlowArc]
    flow_unit: str
    cost_unit: str


class EvidenceIR(BaseModel):
    source_id: str
    locator: str
    title: str


class ModelAssumption(BaseModel):
    assumption_id: str
    statement: str
    source_fact_ids: list[str] = Field(default_factory=list)


class OptimizationIR(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1"]
    problem_id: str = Field(min_length=1)
    problem_family: Literal[
        "employee_scheduling",
        "linear_programming",
        "minimum_cost_flow",
        "energy_park",
    ]
    formulation: Annotated[
        LinearFormulationIR | NetworkFlowFormulationIR,
        Field(discriminator="kind"),
    ]
    evidence: list[EvidenceIR] = Field(default_factory=list)
    assumptions: list[ModelAssumption] = Field(default_factory=list)
```

- [x] **步骤 4：复跑 IR 测试并增加 JSON 往返测试。**

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_optimization_ir.py -q
```

## 任务二：领域编译器与解映射

统一签名如下；所有编译器在输入校验失败时抛出 `ValidationError`，不得静默修补输入：

```python
def compile_linear_problem(
    problem: LinearProgramProblem,
    *,
    problem_id: str,
    evidence: list[EvidenceIR] | None = None,
) -> OptimizationIR: ...


def compile_scheduling_problem(
    problem: SchedulingProblem,
    *,
    problem_id: str,
    evidence: list[EvidenceIR] | None = None,
) -> OptimizationIR: ...


def compile_min_cost_flow_problem(
    problem: MinCostFlowProblem,
    *,
    problem_id: str,
    evidence: list[EvidenceIR] | None = None,
) -> OptimizationIR: ...
```

当前 `LinearProgramProblem` 尚无变量上下界、表达式单位或来源事实编号字段，因此编译器将这些未知信息保留为空，不推断或伪造来源；调用方显式提供的 `EvidenceIR` 会作为 IR 的证据集合保留。后续若领域输入开始携带逐项来源编号，再扩展逐变量/逐约束的来源映射。

- [x] **步骤 1：为 LP/MILP 编译写失败测试。** 对同一 `LinearProgramProblem`，验证变量域、单位、目标方向/系数、约束关系/右端值完全映射；源模型未提供的上下界、表达式单位和来源事实编号不得臆造。
- [x] **步骤 2：运行 LP/MILP 红灯。** 编译器实现前，新增的 12 项编译器测试因模块不存在而失败。

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_optimization_compilers.py -k linear -q
```

预期：`compile_linear_problem` 尚不存在。

- [x] **步骤 3：实现并验证线性模型编译。** `compile_linear_problem()` 不修改原输入对象；转换变量域、单位、目标和约束，缺失的来源信息保持显式为空。
- [x] **步骤 4：运行 LP/MILP 绿灯。** 编译器定向测试通过。
- [x] **步骤 5：为排班编译写失败测试。** 每个员工—班次对生成二进制变量；人数、技能人数、员工工时和最小总排班分钟数目标与现有 `solve_schedule()` 一致，并覆盖无人具备某个必需技能的约束表达。
- [x] **步骤 6：运行排班红灯。** 编译器实现前对应测试失败。

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_optimization_compilers.py -k scheduling -q
```

预期：`compile_scheduling_problem` 尚不存在。

- [x] **步骤 7：实现并验证排班编译与解映射。** `decode_schedule_solution()` 只返回取值为 1 的员工—班次指派；小时换算复用 `_hours_to_minutes()`，并拒绝缺失、未知、非有限和非二进制容差外的解值。
- [x] **步骤 8：运行排班绿灯。** 编译器定向测试通过。
- [x] **步骤 9：为网络流编译写失败测试。** IR 保留节点供需、弧容量、单位费用、流量单位和节点/弧编号，并可映射整数弧流。
- [x] **步骤 10：运行网络流红灯。** 编译器实现前对应测试失败。

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_optimization_compilers.py -k flow -q
```

预期：`compile_min_cost_flow_problem` 尚不存在。

- [x] **步骤 11：实现网络流编译与解映射。** 网络语义保持结构化，SimpleMinCostFlow 后端不从自然语言或不稳定文本重建图；解映射检查弧编号、有限性、非负整数值及容量上限。
- [x] **步骤 12：运行全部编译器定向测试。** 最终结果为 `12 passed`；全库回归 `205 passed`。

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_optimization_compilers.py -q
```

## 任务三：求解器能力注册表

- [ ] **步骤 1：写后端选择红灯测试。** 全连续线性 formulation 选择 GLOP；含连续与离散变量的线性 formulation 选择 SCIP；全整数且系数可精确整数化的排班 formulation 选择 CP-SAT；网络流 formulation 优先选择 SimpleMinCostFlow。无法兼容的 formulation 返回 `UNSUPPORTED_MODEL`，不自动删约束。
- [ ] **步骤 2：运行后端红灯。**

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_optimization_registry.py -q
```

预期：注册表和 `OptimizationResult` 尚不存在。

- [ ] **步骤 3：定义后端接口与能力。**

```python
from dataclasses import dataclass
from typing import Protocol


@dataclass
class OptimizationResult:
    status: str
    objective_value: float | None
    variable_values: dict[str, float]
    backend_id: str
    message: str | None = None


class SolverBackend(Protocol):
    backend_id: str

    def supports(self, problem: OptimizationIR) -> bool: ...

    def solve(self, problem: OptimizationIR) -> OptimizationResult: ...
```

- [ ] **步骤 4：实现 IR 到现有 OR-Tools 后端的转换。** LP/MILP 编译线性约束与变量上下界；CP-SAT 路径拒绝非整数系数和连续变量；SimpleMinCostFlow 只接受已校验网络流 formulation。
- [ ] **步骤 5：实现 `SolverRegistry` 和稳定求解结果。** `OptimizationResult` 包含状态、目标值、变量/弧解、后端编号和求解说明；非可行状态不得带伪造数值解。
- [ ] **步骤 6：运行后端绿灯。**

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_optimization_registry.py -q
```

## 任务四：公共 ModelingService 与旧领域迁移

- [ ] **步骤 1：写服务状态红灯测试。** 测试 `draft -> validated -> confirmed -> solved -> verified` 顺序；未确认、确认摘要过期、IR 编译失败和 solver 不支持时都不得进入求解。
- [ ] **步骤 2：运行服务红灯。**

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_modeling_service.py -q
```

预期：`modeling_service.py` 尚不存在，相关导入失败。

- [ ] **步骤 3：实现 `ModelingService`。** 服务使用稳定的普通 Python/Pydantic 类型，不导入 LangChain 包；包含 `create_draft`、`validate_draft`、`confirm_draft`、`solve_confirmed` 和 `verify_result`。

```python
from typing import Protocol

from math_modeling_agent.analysis_agent import ProblemAnalysis


class DraftValidation(BaseModel):
    valid: bool
    errors: list[str]
    draft_hash: str


class ConfirmationToken(BaseModel):
    session_id: str
    draft_hash: str
    confirmed_at: str


class ModelingSession(BaseModel):
    session_id: str
    state: Literal["draft", "validated", "confirmed", "solved", "verified"]
    draft_hash: str
    analysis: ProblemAnalysis | None = None
    ir: OptimizationIR | None = None


class ModelingReport(BaseModel):
    problem_id: str
    problem_family: str
    solver_status: str
    result: dict[str, object]
    validation_errors: list[str]
    evidence: list[EvidenceIR]


class ModelingService(Protocol):
    def create_draft(self, request: str) -> ModelingSession: ...
    def validate_draft(self, session_id: str, draft_hash: str) -> DraftValidation: ...
    def confirm_draft(self, session_id: str, draft_hash: str) -> ConfirmationToken: ...
    def solve_confirmed(self, token: ConfirmationToken) -> ModelingReport: ...
```

`ModelingSession.analysis` 保存现有 `ProblemAnalysis`；`ir` 是由领域编译器得到的规范模型。`ConfirmationToken` 由服务端按 `session_id + draft_hash` 生成和校验，不能由 LLM 工具参数自行构造。
- [ ] **步骤 4：实现服务并运行绿灯。** 同一服务测试通过，重点确认未确认和过期摘要没有触发 solver。
- [ ] **步骤 5：把排班、LP/MILP、网络流 agent 迁移到编译器和注册表。** 保留 HMML 方法建议及各领域独立 validator；映射后的输出仍保留现有 JSON 字段，另增加 IR 版本和后端信息。原有排班 `--input` 格式保持兼容。
- [ ] **步骤 6：为 CLI/Evals 迁移写失败测试。** `--request` 只返回待审阅草稿；`--solve-draft` 处理经审阅的排班/LP/网络流草稿；未确认草稿不得求解；现有排班 `--input` JSON 仍有效。
- [ ] **步骤 7：运行 CLI/Evals 红灯。**

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_cli.py tests\test_evals.py -q
```

预期：现有 CLI 尚未通过 `ModelingService` 路由新编译器。
- [ ] **步骤 8：实现 CLI/Evals 公共服务调用。** `--solve-draft` 先生成并验证当前草稿摘要，再调用确认服务；`--preview` 本期不实现。
- [ ] **步骤 9：运行 CLI/Evals 绿灯并比较旧结果。** 排班指派、LP 最优值、网络流弧流/总费用需与迁移前一致，且原领域 validator 全部通过。
- [ ] **步骤 10：运行服务与 CLI 定向测试。**

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_modeling_service.py tests\test_cli.py tests\test_evals.py -q
```

## 任务五：RAG 知识库与混合检索

- [ ] **步骤 1：写证据结构和检索失败测试。** 测试每个结果必须含来源 ID、文件/URL、章节/页码、审核状态和问题族；来源不存在、元数据缺失和错误过滤器均返回明确错误或空结果。
- [ ] **步骤 2：运行证据结构红灯。**

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_knowledge_service.py -k evidence_model -q
```

预期：`knowledge_models.py` 尚不存在。
- [ ] **步骤 3：定义检索数据契约。**

```python
class KnowledgeChunk(BaseModel):
    chunk_id: str
    text: str
    source_id: str
    source_uri: str
    locator: str
    problem_families: list[str]
    solver_ids: list[str]
    review_status: Literal["approved", "draft"]


class RetrievedEvidence(BaseModel):
    chunk_id: str
    source_id: str
    source_uri: str
    locator: str
    text: str
    retrieval_method: Literal["keyword", "vector", "hybrid"]
    rank: int
    problem_families: list[str]
    review_status: Literal["approved"]
```

- [ ] **步骤 4：写 `KnowledgeService` 假向量库测试。** 假向量库返回预设来源，关键词侧用固定查询验证 Reciprocal Rank Fusion 去重、顺序稳定和问题族过滤。

```python
class KnowledgeService(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        problem_family: str,
        solver_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]: ...
```

- [ ] **步骤 5：运行检索服务红灯。**

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_knowledge_service.py -q
```

预期：知识分块、关键词/向量合并和证据来源测试失败。
- [ ] **步骤 6：实现清单、分块和稳定来源定位。** 创建 `linear-modeling.md`（变量/目标/线性约束和单位）、`solver-capabilities.md`（后端适用边界）、`model-validation.md`（模型来源与独立核验）和 `energy-storage.md`（已确认的储能假设）四张中文知识卡。`data/knowledge/manifest.json` 逐条列出 ID、标题、来源 URL/路径、版本、问题族、审核状态和 Markdown 卡片路径；只收录审核过的本地知识卡及公开资料摘要。`scripts/build_knowledge_index.py` 校验清单后按 Markdown 标题切块，使用 `sha256(source_id + locator + normalized_text)` 生成稳定 chunk ID，并写入 Chroma 元数据。
- [ ] **步骤 7：加入本地 Chroma 与嵌入提供方。** `rag` extra 使用 `langchain-chroma` 和 `langchain-huggingface`；索引保存在 `.cache/optimization-rag` 并写入 `.gitignore`，关闭 Chroma 匿名遥测。默认嵌入候选为中文多语种 BGE-M3；索引构建命令显式触发模型加载/下载，单元测试始终使用假嵌入，不自动联网。
- [ ] **步骤 8：实现 BM25、HMML 关键词与 Chroma 向量检索的 RRF 合并。** BM25 先对文档分块召回，HMML 关键词检索返回结构化方法；按 `1 / (60 + rank)` 合并各路排名，以问题族/方法/求解器元数据过滤，返回 `RetrievedEvidence`；每个结果都保留可显示的来源，不把向量分数伪装成概率。
- [ ] **步骤 9：把知识检索接入公共服务。** 检索证据可支持方法建议、公式和假设检查；生成模型时只接受题目/附件中的事实，RAG 数值仅可作为带来源的参考假设，并需用户确认。
- [ ] **步骤 10：建立离线检索评测并运行 RAG 测试。** `evals/knowledge_cases.json` 包含查询和预期 source ID；报告 Recall@k、MRR、引用来源准确率和无相关证据时的拒答率。

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_knowledge_service.py tests\test_knowledge_eval.py -q
```

## 任务六：LangChain `create_agent` 适配

- [ ] **步骤 1：写适配层红灯测试。** 核心包在未安装 LangChain 时仍可导入；配置 `langchain` extra 后，Agent 工具使用相同 `ModelingService` 和 Pydantic DTO。
- [ ] **步骤 2：运行适配层红灯。**

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_langchain_adapter.py -q
```

预期：`langchain_adapter.py` 尚不存在，测试无法构建 agent。
- [ ] **步骤 3：新增可选依赖。** `langchain` extra 安装 `langchain>=1.0,<2` 与 `langchain-openai>=1.0,<2`；`rag` extra 安装 `langchain-chroma>=0.1.2`、`langchain-huggingface`、`sentence-transformers` 和 `rank-bm25`。普通 `pip install -e .` 不安装这些依赖。

安装和运行可选能力的 PowerShell 命令：

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pip install -e '.[dev,agent,langchain,rag]'
$env:PYTHONPATH = 'D:\Agent\.worktrees\min-cost-network-flow\src'
& 'D:\Agent\.venv\Scripts\python.exe' scripts\build_knowledge_index.py --knowledge-dir .\data\knowledge --index-dir .\.cache\optimization-rag
```
- [ ] **步骤 4：实现工具封装。** RAG 检索、问题分析/草稿、IR 校验、能力查询、已确认求解、独立验证和报告各自为独立工具；求解工具只接受服务签发且与当前草稿摘要匹配的确认令牌。
- [ ] **步骤 5：实现 `build_langchain_agent()`。** 通过 `create_agent(model=..., tools=..., response_format=...)` 组合工具；DeepSeek OpenAI 兼容客户端通过配置注入，密钥只从环境变量读取且禁止写入日志。

```python
import os
from dataclasses import dataclass

from langchain.agents import AgentState, create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI


class ModelingAgentState(AgentState):
    session_id: str
    draft_hash: str | None
    workflow_state: str
    evidence_ids: list[str]


@dataclass(frozen=True)
class AgentRunContext:
    session_id: str
    confirmation_token: ConfirmationToken | None


def build_deepseek_chat_model() -> ChatOpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        extra_body={"thinking": {"type": "disabled"}},
    )


def build_modeling_tools(
    service: ModelingService,
    retriever: KnowledgeService,
) -> list[BaseTool]: ...


def build_langchain_agent(
    service: ModelingService,
    *,
    model: BaseChatModel,
    retriever: KnowledgeService,
):
    tools = build_modeling_tools(service, retriever)
    return create_agent(
        model=model,
        tools=tools,
        response_format=ToolStrategy(ProblemAnalysis),
        state_schema=ModelingAgentState,
        context_schema=AgentRunContext,
    )
```

DeepSeek V4.1 Flash 的工具调用默认处于思考模式；工具循环若不回传 `reasoning_content` 会失败。v1 显式关闭思考模式；启用思考模式前必须增加多轮工具历史字段的回传测试。
- [ ] **步骤 6：添加状态和确认守门测试。** Fake model 可以提出求解工具调用，但没有由宿主在用户明确确认后注入的 `AgentRunContext.confirmation_token` 时，服务必须拒绝；旧草稿摘要也必须拒绝。LLM 工具 schema 不包含可自行填写的 `confirmed=true` 字段。
- [ ] **步骤 7：运行适配层绿灯。**

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_langchain_adapter.py -q
```

- [ ] **步骤 8：在 CLI 增加 `--framework {native,langchain}`。** 该参数仅作用于 `--request`；默认保持 `native` 以便回归对照。同一输入分别走两种编排器，必须返回兼容的 `ProblemAnalysis` 结构；LangChain 路径未配置可选依赖时返回明确安装指引。

## 任务七：能源园区 IR 接入与回归

```python
class StorageAssumptions(BaseModel):
    soc_min_fraction: float = 0.10
    soc_max_fraction: float = 0.90
    duration_hours: float = 4.0
    self_loss_per_day: float = 0.002
    cyclic_daily_soc: bool = True


def compile_energy_park_problem_four(
    dataset: EnergyParkDataset,
    assumptions: StorageAssumptions,
    *,
    problem_id: str,
) -> OptimizationIR: ...
```

- [ ] **步骤 1：写 Q4 IR 红灯测试。** 构造固定 24 小时的有效数据集，确认 SOC 上下限、每日循环、充放电互斥、功率—能量关系和来源假设都出现在 IR 中；另测弃电最多场景选择。
- [ ] **步骤 2：运行 Q4 红灯。**

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\test_energy_park_ir.py -q
```

预期：`energy_park_milp.py` 尚不存在。

- [ ] **步骤 3：实现储能假设模型和 Q4 builder。** 保持已批准的 SOC 10%—90%、4 小时功率额定、90%/90% 效率、0.2%/日损耗、代表日首末 SOC 相等和敏感性方案不变；运行连续仿射/二进制 IR。
- [ ] **步骤 4：实现 IR solver 路由与园区独立 validator。** 通过能力注册表路由含二进制充放电变量的能源模型到 SCIP；validator 独立重算功率平衡、氢氨守恒、SOC 动态、充放电互斥和成本。
- [ ] **步骤 5：运行 Q4 IR 绿灯。** 同一 `tests\test_energy_park_ir.py` 全部通过。
- [ ] **步骤 6：加入 CLI 与 README PowerShell 命令。** 报告 24 场景各 15 天的 360 天代表年、求解状态、引用证据和实际使用的模型假设。
- [ ] **步骤 7：验证离线端到端流程。** 运行问题四合成输入、离线 Evals、RAG 检索评测；仅用户显式传入 `--live` 才调用 DeepSeek。

## 任务八：最终验证、文档与推送

- [ ] **步骤 1：运行全部测试和静态空白检查。**

```powershell
& 'D:\Agent\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider --basetemp 'D:\Agent\.worktrees\min-cost-network-flow\.pytest-basetemp-ir-rag' -q
git diff --check
```

- [ ] **步骤 2：运行离线 Evals。**

```powershell
$env:PYTHONPATH = 'D:\Agent\.worktrees\min-cost-network-flow\src'
& 'D:\Agent\.venv\Scripts\python.exe' -m math_modeling_agent.evals
```

预期 `api_calls=0`；不输出或记录 API 密钥。

- [ ] **步骤 3：检查 README、`pyproject.toml` optional extras、知识清单和确认流程说明。** 更新能力边界，不声称支持非线性/随机模型。
- [ ] **步骤 4：分阶段用中文提交并普通推送到 `origin/feature/min-cost-network-flow`。** 不使用强推，不合并到 main；最后确认工作树干净且本地分支与远端一致。

## 执行记录

- [x] 任务一 IR 模型与测试已提交：`8f66390`；定向测试 29 项通过。
- [x] 代码复核发现二进制变量单边界越界问题，已补回归测试并修复：`78ad84a`；定向测试 31 项通过。
- [x] IR 规格复核六项全部通过；边界修复的代码质量复核通过。
