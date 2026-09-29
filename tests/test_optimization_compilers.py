from urllib.parse import quote

import pytest
from pydantic import ValidationError

from math_modeling_agent.models import (
    CoverageRequirement,
    Employee,
    FlowArc,
    FlowNode,
    LinearConstraint,
    LinearObjective,
    LinearProgramProblem,
    LinearTerm,
    LinearVariable,
    MinCostFlowProblem,
    SchedulingProblem,
    Shift,
)
from math_modeling_agent.solver import Assignment


def _linear_problem() -> LinearProgramProblem:
    return LinearProgramProblem(
        variables=[
            LinearVariable(name="产品 A", unit="件"),
            LinearVariable(name="产品 B", unit="箱", domain="integer"),
        ],
        objective=LinearObjective(
            direction="maximize",
            terms=[
                LinearTerm(variable="产品 A", coefficient=3.5),
                LinearTerm(variable="产品 B", coefficient=2),
            ],
        ),
        constraints=[
            LinearConstraint(
                name="机器工时",
                terms=[
                    LinearTerm(variable="产品 A", coefficient=1),
                    LinearTerm(variable="产品 B", coefficient=2),
                ],
                relation="<=",
                rhs=10,
            ),
            LinearConstraint(
                name="最低产量",
                terms=[LinearTerm(variable="产品 A", coefficient=1)],
                relation=">=",
                rhs=2,
            ),
        ],
    )


def _scheduling_problem() -> SchedulingProblem:
    return SchedulingProblem(
        employees=[
            Employee(
                employee_id="E:1",
                name="林晓",
                skills={"急救"},
                max_hours=8,
            ),
            Employee(
                employee_id="E2",
                name="陈立",
                skills=set(),
                max_hours=8,
            ),
        ],
        shifts=[Shift(shift_id="S:1", duration_hours=8)],
        coverage_requirements=[
            CoverageRequirement(
                shift_id="S:1",
                minimum_employees=1,
                required_skill_counts={"急救": 1, "驾驶": 1},
            )
        ],
    )


def _flow_problem() -> MinCostFlowProblem:
    return MinCostFlowProblem(
        nodes=[
            FlowNode(node_id="仓库", name="仓库一", supply=5),
            FlowNode(node_id="门店", name="门店一", supply=-5),
        ],
        arcs=[
            FlowArc(
                arc_id="route-1",
                from_node="仓库",
                to_node="门店",
                capacity=8,
                unit_cost=4,
            )
        ],
        flow_unit="箱",
        cost_unit="元/箱",
    )


def test_compile_linear_problem_preserves_domains_units_and_expressions() -> None:
    from math_modeling_agent.optimization_compilers import compile_linear_problem
    from math_modeling_agent.optimization_ir import LinearFormulationIR

    problem = _linear_problem()
    ir = compile_linear_problem(problem, problem_id="linear-001")

    assert ir.problem_id == "linear-001"
    assert ir.problem_family == "linear_programming"
    assert isinstance(ir.formulation, LinearFormulationIR)
    assert [variable.domain for variable in ir.formulation.variables] == [
        "continuous",
        "integer",
    ]
    assert [variable.unit for variable in ir.formulation.variables] == ["件", "箱"]
    assert all(
        variable.lower_bound is None and variable.upper_bound is None
        for variable in ir.formulation.variables
    )
    assert ir.formulation.objective.direction == "maximize"
    assert [term.coefficient for term in ir.formulation.objective.terms] == [3.5, 2]
    assert ir.formulation.constraints[0].relation == "<="
    assert ir.formulation.constraints[0].rhs == 10
    assert ir.formulation.constraints[0].unit is None
    assert ir.formulation.constraints[0].source_fact_ids == []


def test_compile_scheduling_problem_preserves_coverage_skills_and_work_minutes() -> None:
    from math_modeling_agent.optimization_compilers import compile_scheduling_problem
    from math_modeling_agent.optimization_ir import LinearFormulationIR

    problem = _scheduling_problem()
    ir = compile_scheduling_problem(problem, problem_id="schedule-001")

    assert ir.problem_family == "employee_scheduling"
    assert isinstance(ir.formulation, LinearFormulationIR)
    variables = {item.variable_id: item for item in ir.formulation.variables}
    expected_e1 = f"assign:{quote('E:1', safe='')}:{quote('S:1', safe='')}"
    expected_e2 = f"assign:{quote('E2', safe='')}:{quote('S:1', safe='')}"
    assert set(variables) == {expected_e1, expected_e2}
    assert all(item.domain == "binary" for item in variables.values())

    constraints = {item.constraint_id: item for item in ir.formulation.constraints}
    assert constraints["coverage:S%3A1"].relation == ">="
    assert constraints["coverage:S%3A1"].rhs == 1
    assert constraints["skill:S%3A1:%E6%80%A5%E6%95%91"].relation == ">="
    assert constraints["skill:S%3A1:%E6%80%A5%E6%95%91"].rhs == 1
    assert constraints["skill:S%3A1:%E9%A9%BE%E9%A9%B6"].relation == ">="
    assert constraints["skill:S%3A1:%E9%A9%BE%E9%A9%B6"].rhs == 1
    assert constraints["hours:E%3A1"].rhs == 480
    assert constraints["hours:E2"].rhs == 480
    assert all(
        term.coefficient == 480
        for constraint_id, constraint in constraints.items()
        if constraint_id.startswith("hours:")
        for term in constraint.terms
    )
    assert ir.formulation.objective.direction == "minimize"
    assert all(term.coefficient == 480 for term in ir.formulation.objective.terms)


def test_decode_schedule_solution_maps_binary_values_to_assignments() -> None:
    from math_modeling_agent.optimization_compilers import decode_schedule_solution

    problem = _scheduling_problem()
    assignments = decode_schedule_solution(
        problem,
        {
            f"assign:{quote('E:1', safe='')}:{quote('S:1', safe='')}": 1.0,
            f"assign:{quote('E2', safe='')}:{quote('S:1', safe='')}": 0.0,
        },
    )

    assert assignments == [Assignment(employee_id="E:1", shift_id="S:1")]


@pytest.mark.parametrize(
    "values",
    [
        {"assign:E%3A1:S%3A1": 1.0},
        {
            "assign:E%3A1:S%3A1": 1.0,
            "assign:E2:S%3A1": 0.0,
            "assign:unknown:S%3A1": 1.0,
        },
        {"assign:E%3A1:S%3A1": float("nan"), "assign:E2:S%3A1": 0.0},
    ],
)
def test_decode_schedule_solution_rejects_incomplete_unknown_or_nonfinite_values(
    values: dict[str, float],
) -> None:
    from math_modeling_agent.optimization_compilers import decode_schedule_solution

    with pytest.raises(ValueError):
        decode_schedule_solution(_scheduling_problem(), values)


def test_compile_min_cost_flow_problem_preserves_network_and_units() -> None:
    from math_modeling_agent.optimization_compilers import compile_min_cost_flow_problem
    from math_modeling_agent.optimization_ir import NetworkFlowFormulationIR

    ir = compile_min_cost_flow_problem(_flow_problem(), problem_id="flow-001")

    assert ir.problem_family == "minimum_cost_flow"
    assert isinstance(ir.formulation, NetworkFlowFormulationIR)
    assert ir.formulation.nodes[0].supply == 5
    assert ir.formulation.nodes[1].supply == -5
    assert ir.formulation.arcs[0].capacity == 8
    assert ir.formulation.arcs[0].unit_cost == 4
    assert ir.formulation.flow_unit == "箱"
    assert ir.formulation.cost_unit == "元/箱"


def test_decode_flow_solution_preserves_arc_ids_and_integer_flows() -> None:
    from math_modeling_agent.optimization_compilers import decode_flow_solution

    assert decode_flow_solution(_flow_problem(), {"route-1": 5.0}) == {"route-1": 5}


@pytest.mark.parametrize(
    "values",
    [{}, {"unknown": 1.0}, {"route-1": -1.0}, {"route-1": 2.2}],
)
def test_decode_flow_solution_rejects_missing_unknown_negative_or_fractional_flows(
    values: dict[str, float],
) -> None:
    from math_modeling_agent.optimization_compilers import decode_flow_solution

    with pytest.raises(ValueError):
        decode_flow_solution(_flow_problem(), values)
