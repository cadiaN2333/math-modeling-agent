"""把当前领域问题编译到求解器中立 IR，并将 IR 解映射回领域结果。"""

from math import isclose, isfinite
from urllib.parse import quote

from .models import LinearProgramProblem, MinCostFlowProblem, SchedulingProblem
from .optimization_ir import (
    EvidenceIR,
    LinearConstraintIR as IRConstraint,
    LinearFormulationIR,
    LinearObjectiveIR as IRObjective,
    LinearTermIR as IRTerm,
    LinearVariableIR,
    NetworkFlowArcIR,
    NetworkFlowFormulationIR,
    NetworkFlowNodeIR,
    OptimizationIR,
)
from .solver import Assignment, _hours_to_minutes


_DECODE_TOLERANCE = 1e-6


def compile_linear_problem(
    problem: LinearProgramProblem,
    *,
    problem_id: str,
    evidence: list[EvidenceIR] | None = None,
) -> OptimizationIR:
    """把 LP/MILP 领域模型转成带来源容器的仿射 IR。"""

    variables = [
        LinearVariableIR(
            variable_id=item.name,
            name=item.name,
            domain=item.domain,
            lower_bound=None,
            upper_bound=None,
            unit=item.unit,
        )
        for item in problem.variables
    ]
    objective = IRObjective(
        direction=problem.objective.direction,
        terms=[
            IRTerm(variable_id=term.variable, coefficient=term.coefficient)
            for term in problem.objective.terms
        ],
        constant=0,
        unit=None,
    )
    constraints = [
        IRConstraint(
            constraint_id=item.name,
            name=item.name,
            terms=[
                IRTerm(variable_id=term.variable, coefficient=term.coefficient)
                for term in item.terms
            ],
            relation=item.relation,
            rhs=item.rhs,
            constant=0,
            unit=None,
        )
        for item in problem.constraints
    ]
    return OptimizationIR(
        schema_version="1",
        problem_id=problem_id,
        problem_family="linear_programming",
        formulation=LinearFormulationIR(
            kind="linear",
            variables=variables,
            objective=objective,
            constraints=constraints,
        ),
        evidence=list(evidence or []),
        assumptions=[],
    )


def _schedule_variable_id(employee_id: str, shift_id: str) -> str:
    """用百分号编码生成无歧义且可逆的员工—班次变量编号。"""

    return f"assign:{quote(employee_id, safe='')}:{quote(shift_id, safe='')}"


def compile_scheduling_problem(
    problem: SchedulingProblem,
    *,
    problem_id: str,
    evidence: list[EvidenceIR] | None = None,
) -> OptimizationIR:
    """把排班人数、技能和最大工时规则展开为二进制仿射模型。"""

    variables = [
        LinearVariableIR(
            variable_id=_schedule_variable_id(employee.employee_id, shift.shift_id),
            name=f"{employee.employee_id} 在 {shift.shift_id} 上班",
            domain="binary",
            lower_bound=0,
            upper_bound=1,
            unit="1",
        )
        for employee in problem.employees
        for shift in problem.shifts
    ]

    constraints: list[IRConstraint] = []
    for requirement in problem.coverage_requirements:
        shift_id = quote(requirement.shift_id, safe="")
        assignment_terms = [
            IRTerm(
                variable_id=_schedule_variable_id(
                    employee.employee_id,
                    requirement.shift_id,
                ),
                coefficient=1,
            )
            for employee in problem.employees
        ]
        constraints.append(
            IRConstraint(
                constraint_id=f"coverage:{shift_id}",
                name=f"班次 {requirement.shift_id} 最低人数",
                terms=assignment_terms,
                relation=">=",
                rhs=requirement.minimum_employees,
                unit="人",
            )
        )

        for skill, required_count in sorted(
            requirement.required_skill_counts.items()
        ):
            skilled_terms = [
                IRTerm(
                    variable_id=_schedule_variable_id(
                        employee.employee_id,
                        requirement.shift_id,
                    ),
                    coefficient=1,
                )
                for employee in problem.employees
                if skill in employee.skills
            ]
            if not skilled_terms:
                # 用已声明变量的零系数项表达“当前无人拥有该技能”的常量左侧0。
                first_employee = problem.employees[0]
                skilled_terms = [
                    IRTerm(
                        variable_id=_schedule_variable_id(
                            first_employee.employee_id,
                            requirement.shift_id,
                        ),
                        coefficient=0,
                    )
                ]
            constraints.append(
                IRConstraint(
                    constraint_id=(
                        f"skill:{shift_id}:{quote(skill, safe='')}"
                    ),
                    name=f"班次 {requirement.shift_id} 技能 {skill} 人数",
                    terms=skilled_terms,
                    relation=">=",
                    rhs=required_count,
                    unit="人",
                )
            )

    for employee in problem.employees:
        encoded_employee_id = quote(employee.employee_id, safe="")
        work_terms = [
            IRTerm(
                variable_id=_schedule_variable_id(
                    employee.employee_id,
                    shift.shift_id,
                ),
                coefficient=_hours_to_minutes(shift.duration_hours),
            )
            for shift in problem.shifts
        ]
        constraints.append(
            IRConstraint(
                constraint_id=f"hours:{encoded_employee_id}",
                name=f"员工 {employee.employee_id} 最大工时",
                terms=work_terms,
                relation="<=",
                rhs=_hours_to_minutes(employee.max_hours),
                unit="分钟",
            )
        )

    objective_terms = [
        IRTerm(
            variable_id=_schedule_variable_id(
                employee.employee_id,
                shift.shift_id,
            ),
            coefficient=_hours_to_minutes(shift.duration_hours),
        )
        for employee in problem.employees
        for shift in problem.shifts
    ]
    return OptimizationIR(
        schema_version="1",
        problem_id=problem_id,
        problem_family="employee_scheduling",
        formulation=LinearFormulationIR(
            kind="linear",
            variables=variables,
            objective=IRObjective(
                direction="minimize",
                terms=objective_terms,
                constant=0,
                unit="分钟",
            ),
            constraints=constraints,
        ),
        evidence=list(evidence or []),
        assumptions=[],
    )


def _validate_solution_keys(
    expected_ids: set[str],
    variable_values: dict[str, float],
    *,
    result_kind: str,
) -> None:
    """确保 IR 解的变量集合与模型完全相同。"""

    supplied_ids = set(variable_values)
    missing_ids = expected_ids - supplied_ids
    unknown_ids = supplied_ids - expected_ids
    if missing_ids:
        raise ValueError(f"求解结果缺少{result_kind}：{sorted(missing_ids)}")
    if unknown_ids:
        raise ValueError(f"求解结果包含未知{result_kind}：{sorted(unknown_ids)}")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        for value in variable_values.values()
    ):
        raise ValueError("求解结果包含非有限数值")


def decode_schedule_solution(
    problem: SchedulingProblem,
    variable_values: dict[str, float],
) -> list[Assignment]:
    """把 IR 二进制解映射成排班记录并检查二进制容差。"""

    expected_ids = {
        _schedule_variable_id(employee.employee_id, shift.shift_id)
        for employee in problem.employees
        for shift in problem.shifts
    }
    _validate_solution_keys(
        expected_ids,
        variable_values,
        result_kind="排班变量",
    )

    assignments: list[Assignment] = []
    for employee in problem.employees:
        for shift in problem.shifts:
            variable_id = _schedule_variable_id(
                employee.employee_id,
                shift.shift_id,
            )
            value = variable_values[variable_id]
            if isclose(value, 1.0, rel_tol=0, abs_tol=_DECODE_TOLERANCE):
                assignments.append(
                    Assignment(
                        employee_id=employee.employee_id,
                        shift_id=shift.shift_id,
                    )
                )
            elif not isclose(
                value,
                0.0,
                rel_tol=0,
                abs_tol=_DECODE_TOLERANCE,
            ):
                raise ValueError(f"排班变量值不在 0/1 容差内：{variable_id}={value}")
    return assignments


def compile_min_cost_flow_problem(
    problem: MinCostFlowProblem,
    *,
    problem_id: str,
    evidence: list[EvidenceIR] | None = None,
) -> OptimizationIR:
    """保留图结构把最小费用流领域模型编译到网络流 IR。"""

    return OptimizationIR(
        schema_version="1",
        problem_id=problem_id,
        problem_family="minimum_cost_flow",
        formulation=NetworkFlowFormulationIR(
            kind="network_flow",
            nodes=[
                NetworkFlowNodeIR(
                    node_id=node.node_id,
                    name=node.name,
                    supply=node.supply,
                )
                for node in problem.nodes
            ],
            arcs=[
                NetworkFlowArcIR(
                    arc_id=arc.arc_id,
                    from_node=arc.from_node,
                    to_node=arc.to_node,
                    capacity=arc.capacity,
                    unit_cost=arc.unit_cost,
                )
                for arc in problem.arcs
            ],
            flow_unit=problem.flow_unit,
            cost_unit=problem.cost_unit,
        ),
        evidence=list(evidence or []),
        assumptions=[],
    )


def decode_flow_solution(
    problem: MinCostFlowProblem,
    variable_values: dict[str, float],
) -> dict[str, int]:
    """把按弧编号返回的 IR 数值解映射为整数弧流。"""

    arcs_by_id = {arc.arc_id: arc for arc in problem.arcs}
    _validate_solution_keys(
        set(arcs_by_id),
        variable_values,
        result_kind="路线流量",
    )

    decoded: dict[str, int] = {}
    for arc_id, value in variable_values.items():
        rounded_flow = round(value)
        if abs(value - rounded_flow) > _DECODE_TOLERANCE:
            raise ValueError(f"路线流量必须是整数：{arc_id}={value}")
        if rounded_flow < 0 or rounded_flow > arcs_by_id[arc_id].capacity:
            raise ValueError(f"路线流量超出合法范围：{arc_id}={value}")
        decoded[arc_id] = int(rounded_flow)
    return decoded
