"""编排连续线性规划求解、HMML 方法推荐与独立验证。"""

from dataclasses import dataclass

from .linear_solver import LinearSolverResult, solve_linear_program
from .linear_validator import LinearValidationReport, validate_linear_solution
from .models import LinearProgramProblem
from .retriever import HMMLRetriever, MethodRecommendation


@dataclass
class LinearModelingRun:
    """保存 LP 方法建议、求解结果和独立验证报告。"""

    solver_result: LinearSolverResult
    validation_report: LinearValidationReport | None
    method_recommendations: list[MethodRecommendation]


def _describe_linear_program(problem: LinearProgramProblem) -> str:
    """将 LP 结构压缩成 HMML 检索可使用的问题与目标描述。"""

    variable_text = "、".join(variable.name for variable in problem.variables)
    constraint_count = len(problem.constraints)
    direction = "最大化" if problem.objective.direction == "maximize" else "最小化"
    return (
        f"连续变量线性规划 LP，变量为 {variable_text}，含 {constraint_count} 条线性约束；"
        f"目标为{direction}单一线性目标。"
    )


def run_linear_modeling(problem: LinearProgramProblem) -> LinearModelingRun:
    """检索连续 LP 方法，调用 GLOP，并独立验证可行解。"""

    method_recommendations = HMMLRetriever().retrieve(
        problem_description=_describe_linear_program(problem),
        desired_outcome="使用连续线性规划满足线性约束并优化单一目标函数。",
    )
    solver_result = solve_linear_program(problem)

    if solver_result.status not in {"OPTIMAL", "FEASIBLE"}:
        return LinearModelingRun(
            solver_result=solver_result,
            validation_report=None,
            method_recommendations=method_recommendations,
        )

    validation_report = validate_linear_solution(
        problem,
        solver_result.variable_values,
        solver_result.objective_value,
    )
    return LinearModelingRun(
        solver_result=solver_result,
        validation_report=validation_report,
        method_recommendations=method_recommendations,
    )
