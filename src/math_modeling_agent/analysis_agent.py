"""调用 DeepSeek 分析建模问题，并为子任务准备 HMML 检索词。"""

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .models import (
    CoverageRequirement,
    Employee,
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


class ProblemAnalysis(BaseModel):
    """问题分析、缺失信息与子任务分解结果。"""

    status: Literal["ready", "needs_clarification", "unsupported"] = Field(
        description="ready 表示信息足够；needs_clarification 表示要追问；unsupported 表示超出系统范围"
    )
    summary: str = Field(description="对用户问题的简明重述")
    known_facts: list[str] = Field(description="用户明确给出的事实")
    missing_information: list[str] = Field(description="建立模型所需但尚缺的信息")
    clarifying_questions: list[str] = Field(description="需要向用户提出的问题")
    unsupported_reasons: list[str] = Field(description="当前排班模型无法处理的要求")
    subtasks: list[AnalysisSubtask] = Field(description="拆分后的子任务")
    scheduling_draft: SchedulingDraft = Field(
        description="信息完整时填写排班草稿；需要追问或超出范围时各列表返回空数组"
    )

    @model_validator(mode="after")
    def validate_analysis(self) -> "ProblemAnalysis":
        """校验状态、子任务编号及依赖关系。"""

        if self.status == "ready":
            if self.missing_information or self.clarifying_questions:
                raise ValueError("信息不完整时不能标记为 ready")
            if self.unsupported_reasons:
                raise ValueError("存在不支持的要求时不能标记为 ready")
            if not self.subtasks:
                raise ValueError("ready 状态至少需要一个子任务")
            if not (
                self.scheduling_draft.employees
                and self.scheduling_draft.shifts
                and self.scheduling_draft.coverage_requirements
            ):
                raise ValueError("ready 状态必须完整提供结构化排班草稿")
        elif any(
            (
                self.scheduling_draft.employees,
                self.scheduling_draft.shifts,
                self.scheduling_draft.coverage_requirements,
            )
        ):
            raise ValueError("需要追问或不支持的问题必须使用空排班草稿")

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

当前系统只支持员工排班，能处理：员工技能、员工最大总工时、班次时长、每班最低人数和每班技能人数要求。
当前模型没有班次起止时间、员工可用时间、工资、公平性权重等字段。

请遵守以下规则：
1. 只把用户明确提供的信息写入 known_facts，不能编造员工、人数、技能、时长或限制。
2. 如果缺少建立排班模型的必要条件，status 必须是 needs_clarification，并列出 missing_information 和具体问题。
3. 如果用户要求当前系统无法表达的约束，status 必须是 unsupported，并解释 unsupported_reasons。
4. 只有信息足够且要求均在支持范围内，status 才能是 ready。
5. 将复杂问题拆成相互依赖的子任务；简单问题可以只拆成一个任务。
6. 每个子任务填写 hmml_problem_query 和 hmml_goal_query，供分层建模方法库分别检索问题和目标。
7. depends_on 只能填写本次输出中已有的 task_id，依赖关系不得形成循环。
8. status 为 ready 时，必须完整填写 scheduling_draft：
   - employees 中逐人填写 employee_id、name、skills 和 max_hours；
   - shifts 中逐班填写 shift_id 和 duration_hours；
   - coverage_requirements 中逐班填写 shift_id、minimum_employees 和 skill_requirements；
   - skill_requirements 使用 skill 和 minimum_count 两个字段。
9. 用户没有提供员工编号但提供了姓名时，可依次生成 E1、E2 等内部编号；不得编造姓名或工时。
10. 用户没有提出某班的额外技能要求时，该班 skill_requirements 返回空数组，不要为此追问。
11. 只有当缺失信息会影响已明确提出的约束能否检查时才追问相关信息；不相关的技能或可用时间不要追问。
12. status 为 needs_clarification 或 unsupported 时，scheduling_draft 必须为 {"employees": [], "shifts": [], "coverage_requirements": []}。
13. 所有字段都必须返回；没有内容的列表返回空数组。
14. 如果用户请求生产计划线性规划、运输/网络流或其他非员工排班问题，必须标记为 unsupported，说明当前系统只支持员工排班；不得把其他领域问题硬转成排班草稿。
"""


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

    return analysis


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
