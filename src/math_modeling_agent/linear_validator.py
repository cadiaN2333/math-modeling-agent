"""独立验证连续线性规划的变量解、约束和目标值。"""

from dataclasses import dataclass
from math import isfinite

from .models import LinearProgramProblem


DEFAULT_TOLERANCE = 1e-6


@dataclass
class LinearValidationReport:
    """保存 LP 解的独立验证结果。"""

    is_valid: bool
    errors: list[str]
    recomputed_objective: float | None


def validate_linear_solution(
    problem: LinearProgramProblem,
    variable_values: dict[str, float],
    reported_objective: float | None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> LinearValidationReport:
    """重新计算每条约束和目标值，不依赖求解器的可行状态描述。"""

    errors: list[str] = []
    variable_names = {variable.name for variable in problem.variables}
    supplied_names = set(variable_values)
    missing_names = variable_names - supplied_names
    unknown_names = supplied_names - variable_names

    for name in sorted(missing_names):
        errors.append(f"求解结果缺少变量值：{name}")
    for name in sorted(unknown_names):
        errors.append(f"求解结果包含未知变量：{name}")

    finite_values: dict[str, float] = {}
    for name, value in variable_values.items():
        if not isfinite(value):
            errors.append(f"变量 {name} 的值不是有限数值")
        elif name in variable_names:
            finite_values[name] = value

    for constraint in problem.constraints:
        if any(term.variable not in finite_values for term in constraint.terms):
            continue

        left_hand_side = sum(
            term.coefficient * finite_values[term.variable]
            for term in constraint.terms
        )
        if constraint.relation == "<=" and left_hand_side > constraint.rhs + tolerance:
            errors.append(
                f"约束 {constraint.name} 不满足：左侧 {left_hand_side} 大于右侧 {constraint.rhs}"
            )
        elif constraint.relation == ">=" and left_hand_side < constraint.rhs - tolerance:
            errors.append(
                f"约束 {constraint.name} 不满足：左侧 {left_hand_side} 小于右侧 {constraint.rhs}"
            )
        elif constraint.relation == "==" and abs(left_hand_side - constraint.rhs) > tolerance:
            errors.append(
                f"约束 {constraint.name} 不满足：左侧 {left_hand_side} 不等于右侧 {constraint.rhs}"
            )

    objective_variables = {
        term.variable for term in problem.objective.terms
    }
    recomputed_objective = None
    if objective_variables <= finite_values.keys():
        recomputed_objective = sum(
            term.coefficient * finite_values[term.variable]
            for term in problem.objective.terms
        )

    if reported_objective is None or not isfinite(reported_objective):
        errors.append("求解结果缺少有限的目标值")
    elif recomputed_objective is not None and abs(
        recomputed_objective - reported_objective
    ) > tolerance:
        errors.append(
            f"目标值不一致：重算为 {recomputed_objective}，求解器报告 {reported_objective}"
        )

    return LinearValidationReport(
        is_valid=not errors,
        errors=errors,
        recomputed_objective=recomputed_objective,
    )
