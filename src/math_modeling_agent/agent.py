"""编排排班求解和结果验证流程。"""

from dataclasses import dataclass

from .models import SchedulingProblem
from .retriever import HMMLRetriever, MethodRecommendation
from .solver import SolverResult, solve_schedule
from .validator import ValidationReport, validate_solution


@dataclass
class ModelingRun:
    """保存方法建议、求解结果和独立验证报告。"""

    solver_result: SolverResult
    validation_report: ValidationReport | None
    method_recommendations: list[MethodRecommendation]


def _describe_scheduling_problem(problem: SchedulingProblem) -> str:
    """把结构化排班数据整理成 HMML 检索可用的问题描述。"""

    employee_skills = sorted(
        {skill for employee in problem.employees for skill in employee.skills}
    )
    required_skills = sorted(
        {
            skill
            for requirement in problem.coverage_requirements
            for skill in requirement.required_skill_counts
        }
    )
    skill_text = "、".join(employee_skills) if employee_skills else "无"
    required_text = "、".join(required_skills) if required_skills else "无"

    return (
        f"员工排班与人员资源分配问题，共有 {len(problem.employees)} 名员工、"
        f"{len(problem.shifts)} 个班次；员工技能包括 {skill_text}，"
        f"班次技能需求包括 {required_text}。需要考虑最低覆盖人数、"
        "技能覆盖和员工最大工时等离散约束。"
    )


def run_modeling(problem: SchedulingProblem) -> ModelingRun:
    """先检索建模方法，再求解排班问题并独立验证结果。"""

    # 根据问题特征和目标，从 HMML 层级知识树中检索候选方法
    method_recommendations = HMMLRetriever().retrieve(
        problem_description=_describe_scheduling_problem(problem),
        desired_outcome=(
            "满足每班最低人数、技能需求和员工工时上限，"
            "并尽量减少总排班分钟数。"
        ),
    )

    # 计算阶段仍使用项目当前实现的 OR-Tools 求解器
    solver_result = solve_schedule(problem)

    # 无解或求解未完成时，不对不存在的排班做验证
    if solver_result.status not in {"OPTIMAL", "FEASIBLE"}:
        return ModelingRun(
            solver_result=solver_result,
            validation_report=None,
            method_recommendations=method_recommendations,
        )

    # 第二步：把求解结果交给独立验证器复核
    validation_report = validate_solution(
        problem,
        solver_result.assignments,
    )

    return ModelingRun(
        solver_result=solver_result,
        validation_report=validation_report,
        method_recommendations=method_recommendations,
    )
