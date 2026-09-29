"""编排连续/混合整数线性规划求解、HMML 推荐与独立验证。"""

from dataclasses import dataclass

from .linear_solver import LinearSolverResult
from .linear_validator import LinearValidationReport, validate_linear_solution
from .models import LinearProgramProblem
from .optimization_compilers import compile_linear_problem
from .optimization_registry import SolverRegistry
from .retriever import HMMLRetriever, MethodRecommendation


@dataclass
class LinearModelingRun:
    """保存 LP 方法建议、求解结果和独立验证报告。"""

    solver_result: LinearSolverResult
    validation_report: LinearValidationReport | None
    method_recommendations: list[MethodRecommendation]


def _describe_linear_program(problem: LinearProgramProblem) -> str:
    """将 LP 结构压缩成 HMML 检索可使用的问题与目标描述。"""

    domain_names = {
        "continuous": "连续",
        "integer": "整数",
        "binary": "二进制",
    }
    variable_text = "、".join(
        f"{variable.name}为{domain_names[variable.domain]}变量"
        for variable in problem.variables
    )
    constraint_count = len(problem.constraints)
    direction = "最大化" if problem.objective.direction == "maximize" else "最小化"
    discrete_variables = [
        variable for variable in problem.variables if variable.domain != "continuous"
    ]
    if discrete_variables:
        problem_kind = "整数/混合整数线性规划 ILP/MILP"
    else:
        problem_kind = "连续线性规划 LP"
    return (
        f"{problem_kind}，变量为 {variable_text}，含 {constraint_count} 条线性约束；"
        f"目标为{direction}单一线性目标。"
    )


def run_linear_modeling(problem: LinearProgramProblem) -> LinearModelingRun:
    """按变量域选择连续 LP 或 MILP 方法，再独立验证可行解。"""

    has_discrete_variables = any(
        variable.domain != "continuous" for variable in problem.variables
    )
    required_method_id = (
        "integer_programming"
        if has_discrete_variables
        else "continuous_linear_programming"
    )
    if has_discrete_variables:
        desired_outcome = "使用整数/混合整数线性规划和 SCIP 满足线性约束并优化单一目标函数。"
    else:
        desired_outcome = "使用连续线性规划和 GLOP 满足线性约束并优化单一目标函数。"
    retrieved_methods = HMMLRetriever().retrieve(
        problem_description=_describe_linear_program(problem),
        desired_outcome=desired_outcome,
        required_method_id=required_method_id,
    )
    method_recommendations = retrieved_methods
    # 线性模型统一编译到 IR，再由注册表选 GLOP 或 SCIP。
    ir = compile_linear_problem(problem, problem_id="linear-run")
    backend = SolverRegistry().select(ir)
    if backend is None:
        solver_result = LinearSolverResult(
            status="UNSUPPORTED_MODEL",
            objective_value=None,
            variable_values={},
            solver_name="none",
        )
    else:
        ir_result = backend.solve(ir)
        solver_name = {
            "ortools_glop": "GLOP",
            "ortools_scip": "SCIP",
            "ortools_cp_sat": "CP-SAT",
        }.get(backend.backend_id, backend.backend_id)
        solver_result = LinearSolverResult(
            status=ir_result.status,
            objective_value=ir_result.objective_value,
            variable_values=dict(ir_result.variable_values),
            solver_name=solver_name,
        )

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
