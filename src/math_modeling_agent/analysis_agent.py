"""调用 DeepSeek 分析建模问题，并为子任务准备 HMML 检索词。"""

import os
from math import isfinite
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .models import (
    CoverageRequirement,
    Employee,
    LinearConstraint,
    LinearObjective,
    LinearProgramProblem,
    LinearTerm,
    LinearVariable,
    SchedulingProblem,
    Shift,
)
from .retriever import HMMLRetriever, MethodRecommendation


class AnalysisSubtask(BaseModel):
    """一个子任务及其依赖和 HMML 检索线索。"""

    task_id: str = Field(description="子任务编号，例如 T1")
    description: str = Field(description="子任务要处理的具体问题")
    objective: str = Field(description="子任务要达到的目标")
    data_needed: list[str] = Field(description="完成任务需要的数据；没有则为空数组")
    depends_on: list[str] = Field(description="依赖的其他子任务编号；没有则为空数组")
    hmml_problem_query: str = Field(description="供 HMML 问题侧检索的描述")
    hmml_goal_query: str = Field(description="供 HMML 目标侧检索的描述")


class EmployeeDraft(BaseModel):
    """模型从自然语言中提取的一名员工。"""

    employee_id: str = Field(description="员工编号；没有编号时生成 E1、E2 等稳定编号")
    name: str = Field(description="员工姓名或编号")
    skills: list[str] = Field(description="明确提到的技能；没有技能时返回空数组")
    max_hours: float = Field(description="本次排班周期内的最大总工时")


class ShiftDraft(BaseModel):
    """模型从自然语言中提取的一个班次。"""

    shift_id: str = Field(description="班次编号，例如 S1")
    duration_hours: float = Field(description="班次持续小时数")


class SkillRequirementDraft(BaseModel):
    """某个班次对某项技能的最低人数要求。"""

    skill: str = Field(description="技能名称")
    minimum_count: int = Field(description="至少需要的具备此技能人数")


class CoverageDraft(BaseModel):
    """一个班次的人员和技能覆盖要求。"""

    shift_id: str = Field(description="对应的班次编号")
    minimum_employees: int = Field(description="该班次最低总人数")
    skill_requirements: list[SkillRequirementDraft] = Field(
        description="该班次技能人数要求；没有要求时返回空数组"
    )

    @model_validator(mode="after")
    def skill_names_must_be_unique(self) -> "CoverageDraft":
        skill_names = [item.skill for item in self.skill_requirements]
        if len(skill_names) != len(set(skill_names)):
            raise ValueError("同一班次的技能要求不能重复")
        return self


class SchedulingDraft(BaseModel):
    """自然语言分析得到的排班草稿；随后会转换成内部求解模型。"""

    employees: list[EmployeeDraft] = Field(description="员工草稿；无可用数据时返回空数组")
    shifts: list[ShiftDraft] = Field(description="班次草稿；无可用数据时返回空数组")
    coverage_requirements: list[CoverageDraft] = Field(
        description="覆盖要求草稿；无可用数据时返回空数组"
    )


class LinearVariableDraft(BaseModel):
    """自然语言分析得到的一个连续线性规划变量。"""

    name: str = Field(description="变量名称，例如 A")
    unit: str = Field(description="变量单位；没有单位概念时填写无单位")


class LinearTermDraft(BaseModel):
    """目标函数或约束表达式中的一个系数项。"""

    variable: str = Field(description="已声明的变量名称")
    coefficient: float = Field(description="该变量在线性表达式中的系数")

    @field_validator("coefficient")
    @classmethod
    def coefficient_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("线性系数必须是有限数值")
        return value


class LinearConstraintDraft(BaseModel):
    """一个线性约束；变量界限也表示成此类约束。"""

    constraint_id: str = Field(description="约束编号")
    terms: list[LinearTermDraft] = Field(description="线性表达式的系数项")
    relation: Literal["<=", ">=", "==", "not_applicable"] = Field(
        description="约束关系；非活动草稿填写 not_applicable"
    )
    rhs: float = Field(description="约束右侧数值；非活动草稿填写 0")

    @field_validator("rhs")
    @classmethod
    def right_hand_side_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("约束右侧必须是有限数值")
        return value


class LinearProgramDraft(BaseModel):
    """模型提取出的连续单目标线性规划草稿。"""

    variables: list[LinearVariableDraft] = Field(description="决策变量；未使用 LP 时为空")
    objective_direction: Literal["maximize", "minimize", "not_applicable"] = Field(
        description="目标方向；未使用 LP 时填写 not_applicable"
    )
    objective_terms: list[LinearTermDraft] = Field(
        description="目标函数系数项；未使用 LP 时为空"
    )
    constraints: list[LinearConstraintDraft] = Field(
        description="线性约束；未使用 LP 时为空"
    )

    @model_validator(mode="after")
    def validate_expression_references(self) -> "LinearProgramDraft":
        """校验变量和表达式引用；空草稿仅允许 not_applicable 方向。"""

        if self.objective_direction == "not_applicable":
            if self.variables or self.objective_terms or self.constraints:
                raise ValueError("未使用的 LP 草稿必须全部为空")
            return self

        variable_names = [variable.name for variable in self.variables]
        if not variable_names or len(variable_names) != len(set(variable_names)):
            raise ValueError("LP 变量必须存在且名称不能重复")
        if not self.objective_terms:
            raise ValueError("LP 目标函数必须至少包含一项")

        expressions = [self.objective_terms]
        constraint_ids = [constraint.constraint_id for constraint in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("LP 约束编号不能重复")
        for constraint in self.constraints:
            if constraint.relation == "not_applicable":
                raise ValueError("活动 LP 约束必须指定关系")
            if not constraint.terms:
                raise ValueError("活动 LP 约束必须至少包含一项")
            expressions.append(constraint.terms)

        known_variables = set(variable_names)
        for terms in expressions:
            term_variables = [term.variable for term in terms]
            if len(term_variables) != len(set(term_variables)):
                raise ValueError("同一 LP 表达式中变量不能重复")
            unknown_variables = set(term_variables) - known_variables
            if unknown_variables:
                raise ValueError(
                    f"LP 表达式引用了未声明的变量：{sorted(unknown_variables)}"
                )

        return self


class ProblemAnalysis(BaseModel):
    """问题分析、缺失信息与子任务分解结果。"""

    status: Literal["ready", "needs_clarification", "unsupported"] = Field(
        description="ready 表示信息足够；needs_clarification 表示要追问；unsupported 表示超出系统范围"
    )
    problem_family: Literal[
        "employee_scheduling",
        "linear_programming",
        "other",
    ] = Field(description="问题领域；other 表示当前尚未实现的其他优化问题")
    summary: str = Field(description="对用户问题的简明重述")
    known_facts: list[str] = Field(description="用户明确给出的事实")
    missing_information: list[str] = Field(description="建立模型所需但尚缺的信息")
    clarifying_questions: list[str] = Field(description="需要向用户提出的问题")
    unsupported_reasons: list[str] = Field(description="当前排班模型无法处理的要求")
    subtasks: list[AnalysisSubtask] = Field(description="拆分后的子任务")
    scheduling_draft: SchedulingDraft = Field(
        description="信息完整时填写排班草稿；需要追问或超出范围时各列表返回空数组"
    )
    linear_program_draft: LinearProgramDraft = Field(
        description="连续单目标 LP 草稿；未使用 LP 时用固定 not_applicable 空草稿"
    )

    @model_validator(mode="after")
    def validate_analysis(self) -> "ProblemAnalysis":
        """校验状态、子任务编号及依赖关系。"""

        scheduling_is_empty = not any(
            (
                self.scheduling_draft.employees,
                self.scheduling_draft.shifts,
                self.scheduling_draft.coverage_requirements,
            )
        )
        linear_program_is_empty = (
            self.linear_program_draft.objective_direction == "not_applicable"
            and not self.linear_program_draft.variables
            and not self.linear_program_draft.objective_terms
            and not self.linear_program_draft.constraints
        )

        if self.status == "ready":
            if self.missing_information or self.clarifying_questions:
                raise ValueError("信息不完整时不能标记为 ready")
            if self.unsupported_reasons:
                raise ValueError("存在不支持的要求时不能标记为 ready")
            if not self.subtasks:
                raise ValueError("ready 状态至少需要一个子任务")
            if self.problem_family == "employee_scheduling":
                if scheduling_is_empty:
                    raise ValueError("ready 排班问题必须完整提供结构化排班草稿")
                if not linear_program_is_empty:
                    raise ValueError("排班问题不能同时提供 LP 草稿")
            elif self.problem_family == "linear_programming":
                if not scheduling_is_empty:
                    raise ValueError("LP 问题不能同时提供排班草稿")
                if linear_program_is_empty:
                    raise ValueError("ready LP 问题必须提供完整线性规划草稿")
            else:
                raise ValueError("当前未实现的领域不能标记为 ready")
        elif not scheduling_is_empty or not linear_program_is_empty:
            raise ValueError("需要追问或不支持的问题必须使用空领域草稿")

        if self.status == "needs_clarification" and not self.clarifying_questions:
            raise ValueError("needs_clarification 状态必须列出追问")

        if self.status == "unsupported" and not self.unsupported_reasons:
            raise ValueError("unsupported 状态必须说明原因")

        task_ids = [task.task_id for task in self.subtasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("子任务编号不能重复")

        task_by_id = {task.task_id: task for task in self.subtasks}
        for task in self.subtasks:
            if task.task_id in task.depends_on:
                raise ValueError(f"子任务 {task.task_id} 不能依赖自身")
            unknown = set(task.depends_on) - set(task_by_id)
            if unknown:
                raise ValueError(
                    f"子任务 {task.task_id} 引用了不存在的依赖：{sorted(unknown)}"
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("子任务依赖关系不能形成循环")
            if task_id in visited:
                return

            visiting.add(task_id)
            for dependency_id in task_by_id[task_id].depends_on:
                visit(dependency_id)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in task_ids:
            visit(task_id)

        return self


ANALYSIS_INSTRUCTIONS = """
你是数学建模问题分析助手，负责理解和拆分问题；不要求解问题，也不要生成程序代码。

当前已实现两个领域：员工排班和连续线性规划（仅限连续变量、单目标）。
员工排班可处理员工技能、员工最大总工时、班次时长、每班最低人数和每班技能人数要求。
线性规划可处理连续决策变量、一个线性最大化/最小化目标及线性不等式/等式约束；变量界限写成显式线性约束。
当前不支持班次起止时间、员工可用时间、工资、公平性权重、整数或二进制变量、非线性、多目标和运输/网络流专用问题。

请遵守以下规则：
1. 只把用户明确提供的信息写入 known_facts；不能编造系数、资源数量、员工、技能、时长或限制。
2. problem_family 必须是 employee_scheduling、linear_programming 或 other；排班选 employee_scheduling，当前支持范围内的连续 LP 选 linear_programming，其余选 other。
3. 先区分“信息缺失”和“功能不支持”：员工名单、技能、最大工时、资源系数、目标系数等必要数据没提供，属于 needs_clarification，绝不能仅因数据不全标记 unsupported。
4. 只有用户明确要求了当前无法表达/求解的功能或约束（例如班次时间重叠、整数变量、非线性或多目标）时，才标记 unsupported，并在 unsupported_reasons 中指出具体功能；不得把未知员工、缺少最大工时或缺少 LP 系数当成不支持的功能。
5. 若同时存在明确的不支持功能和缺失数据，unsupported 优先；否则只要存在影响建模的缺失数据，就必须 needs_clarification 并列出具体追问。
6. 只有信息充分且问题属于已实现领域时，status 才能是 ready。
7. 将复杂问题拆成相互依赖的子任务；简单问题可以只拆成一个任务。
8. 每个子任务填写 hmml_problem_query 和 hmml_goal_query，供分层建模方法库分别检索问题和目标。
9. depends_on 只能填写本次输出中已有的 task_id，依赖关系不得形成循环。
10. ready 且 problem_family 为 employee_scheduling 时，必须完整填写 scheduling_draft：
   - employees 中逐人填写 employee_id、name、skills 和 max_hours；
   - shifts 中逐班填写 shift_id 和 duration_hours；
   - coverage_requirements 中逐班填写 shift_id、minimum_employees 和 skill_requirements；
   - skill_requirements 使用 skill 和 minimum_count 两个字段；linear_program_draft 必须为空草稿。
11. 用户没有提供员工编号但提供了姓名时，可依次生成 E1、E2 等内部编号；不得编造姓名或工时。
12. 用户没有提出某班的额外技能要求时，该班 skill_requirements 返回空数组，不要为此追问。
13. ready 且 problem_family 为 linear_programming 时，必须完整填写 linear_program_draft：
    - variables 中为每个决策变量填写 name 和 unit；单位缺失且会影响表达时应追问，不涉及单位时填写“无单位”；
    - objective_direction 只能为 maximize 或 minimize；objective_terms 填写用户给出的数值系数项，不得编造；
    - constraints 中为每个约束填写 constraint_id、terms、relation 和 rhs；relation 只能为 <=、>= 或 ==；
    - 非负条件等变量界限必须写成显式约束；不得隐含添加用户未提供的限制；scheduling_draft 必须为空草稿。
14. 当前 LP 仅支持连续变量、单一线性目标和线性约束。用户要求整数或二进制变量、目标或约束非线性、多目标，或运输/网络流专用建模时，必须标记 unsupported，不要转写成其他已实现模型。
15. 当前版本不支持的时间、工资、公平性、整数规划等要求，必须标记 unsupported 并说明具体原因。
16. status 为 needs_clarification 或 unsupported 时，两份领域草稿都必须为空：scheduling_draft 为 {"employees": [], "shifts": [], "coverage_requirements": []}；linear_program_draft 为 {"variables": [], "objective_direction": "not_applicable", "objective_terms": [], "constraints": []}。
17. 所有字段都必须返回；没有内容的列表返回空数组；不得将任何 draft 设为 null。
"""


UNSUPPORTED_FEATURE_MARKERS = (
    "整数",
    "二进制",
    "非线性",
    "多目标",
    "运输",
    "网络流",
    "班次起止",
    "时间重叠",
    "重叠约束",
    "工资",
    "公平",
    "员工可用时间",
    "时间窗",
    "integer",
    "binary",
    "nonlinear",
    "multi-objective",
    "network flow",
    "time window",
)


def _normalize_incomplete_analysis(analysis: ProblemAnalysis) -> ProblemAnalysis:
    """将缺少数据但无明确超范围功能的误判改为追问状态。"""

    if (
        analysis.status != "unsupported"
        or analysis.problem_family not in {"employee_scheduling", "linear_programming"}
        or not analysis.missing_information
        or not analysis.clarifying_questions
    ):
        return analysis

    unsupported_text = " ".join(analysis.unsupported_reasons).casefold()
    if any(marker.casefold() in unsupported_text for marker in UNSUPPORTED_FEATURE_MARKERS):
        return analysis

    corrected = analysis.model_dump()
    corrected["status"] = "needs_clarification"
    corrected["unsupported_reasons"] = []
    return ProblemAnalysis.model_validate(corrected)


def load_project_environment() -> None:
    """从项目根目录加载本地环境变量，不覆盖当前进程已设置的值。"""

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            '缺少环境变量加载依赖；请运行 python -m pip install -e ".[dev,agent]"'
        ) from exc

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=project_root / ".env", override=False)


def analyze_problem(
    problem_text: str,
    client: Any | None = None,
    model: str | None = None,
) -> ProblemAnalysis:
    """用 DeepSeek Responses API 生成结构化问题分析。"""

    if not problem_text.strip():
        raise ValueError("问题描述不能为空")

    if client is None:
        load_project_environment()

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "缺少 OpenAI Python SDK；请运行 python -m pip install -e "
                "\".[dev,agent]\""
            ) from exc

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("请先设置 DEEPSEEK_API_KEY 环境变量")

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    try:
        response = client.responses.parse(
            model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            instructions=ANALYSIS_INSTRUCTIONS,
            input=problem_text,
            text_format=ProblemAnalysis,
        )
    except Exception as exc:
        raise RuntimeError(f"DeepSeek API 调用失败：{exc}") from exc

    analysis = response.output_parsed

    if analysis is None:
        raise RuntimeError("DeepSeek 没有返回可解析的分析结果")

    return _normalize_incomplete_analysis(analysis)


def to_scheduling_problem(analysis: ProblemAnalysis) -> SchedulingProblem:
    """把 ready 状态的自然语言分析结果转换成内部排班模型。"""

    if analysis.status != "ready":
        raise ValueError("只有 ready 状态才能开始求解")

    draft = analysis.scheduling_draft
    return SchedulingProblem(
        employees=[
            Employee(
                employee_id=employee.employee_id,
                name=employee.name,
                skills=set(employee.skills),
                max_hours=employee.max_hours,
            )
            for employee in draft.employees
        ],
        shifts=[
            Shift(
                shift_id=shift.shift_id,
                duration_hours=shift.duration_hours,
            )
            for shift in draft.shifts
        ],
        coverage_requirements=[
            CoverageRequirement(
                shift_id=requirement.shift_id,
                minimum_employees=requirement.minimum_employees,
                required_skill_counts={
                    skill_requirement.skill: skill_requirement.minimum_count
                    for skill_requirement in requirement.skill_requirements
                },
            )
            for requirement in draft.coverage_requirements
        ],
    )


def to_linear_program_problem(analysis: ProblemAnalysis) -> LinearProgramProblem:
    """把 ready 的连续 LP 草稿转换为求解器内部模型。"""

    if analysis.status != "ready" or analysis.problem_family != "linear_programming":
        raise ValueError("只有 ready 的线性规划问题才能开始求解")

    draft = analysis.linear_program_draft
    return LinearProgramProblem(
        variables=[
            LinearVariable(name=variable.name, unit=variable.unit)
            for variable in draft.variables
        ],
        objective=LinearObjective(
            direction=draft.objective_direction,
            terms=[
                LinearTerm(
                    variable=term.variable,
                    coefficient=term.coefficient,
                )
                for term in draft.objective_terms
            ],
        ),
        constraints=[
            LinearConstraint(
                name=constraint.constraint_id,
                terms=[
                    LinearTerm(
                        variable=term.variable,
                        coefficient=term.coefficient,
                    )
                    for term in constraint.terms
                ],
                relation=constraint.relation,
                rhs=constraint.rhs,
            )
            for constraint in draft.constraints
        ],
    )


def retrieve_methods_for_subtasks(
    analysis: ProblemAnalysis,
    retriever: HMMLRetriever | None = None,
) -> dict[str, list[MethodRecommendation]]:
    """对已就绪问题中的每个子任务分别检索 HMML 方法。"""

    if analysis.status != "ready":
        return {}

    method_retriever = retriever or HMMLRetriever()
    return {
        task.task_id: method_retriever.retrieve(
            problem_description=task.hmml_problem_query,
            desired_outcome=task.hmml_goal_query,
        )
        for task in analysis.subtasks
    }
