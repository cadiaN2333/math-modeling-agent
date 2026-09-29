import pytest

from math_modeling_agent.models import (
    CoverageRequirement,
    Employee,
    FlowArc,
    FlowNode,
    SchedulingProblem,
    Shift,
)
from math_modeling_agent.optimization_compilers import (
    compile_min_cost_flow_problem,
    compile_scheduling_problem,
)
from math_modeling_agent.optimization_ir import (
    LinearConstraintIR as IRConstraint,
    LinearFormulationIR,
    LinearObjectiveIR,
    LinearTermIR,
    LinearVariableIR,
    MinCostFlowProblem,
    OptimizationIR,
)


def _linear_ir(
    *,
    problem_id: str = "linear-001",
    problem_family: str = "linear_programming",
    variable_domain: str = "continuous",
    lower_bound: float | None = 0,
) -> OptimizationIR:
    return OptimizationIR(
        schema_version="1",
        problem_id=problem_id,
        problem_family=problem_family,
        formulation=LinearFormulationIR(
            kind="linear",
            variables=[
                LinearVariableIR(
                    variable_id="x",
                    name="产量",
                    domain=variable_domain,
                    lower_bound=lower_bound,
                    upper_bound=None,
                    unit="件",
                )
            ],
            objective=LinearObjectiveIR(
                direction="maximize",
                terms=[LinearTermIR(variable_id="x", coefficient=2)],
                constant=7,
            ),
            constraints=[
                IRConstraint(
                    constraint_id="resource-limit",
                    name="资源上限",
                    terms=[LinearTermIR(variable_id="x", coefficient=1)],
                    relation="<=",
                    rhs=5,
                    constant=2,
                )
            ],
        ),
        evidence=[],
        assumptions=[],
    )


def _schedule_ir() -> OptimizationIR:
    problem = SchedulingProblem(
        employees=[
            Employee(employee_id="E1", name="林晓", skills={"急救"}, max_hours=8),
            Employee(employee_id="E2", name="陈立", skills=set(), max_hours=8),
        ],
        shifts=[Shift(shift_id="S1", duration_hours=8)],
        coverage_requirements=[
            CoverageRequirement(
                shift_id="S1",
                minimum_employees=1,
                required_skill_counts={"急救": 1},
            )
        ],
    )
    return compile_scheduling_problem(problem, problem_id="schedule-001")


def _mixed_integer_ir() -> OptimizationIR:
    return OptimizationIR(
        schema_version="1",
        problem_id="mixed-001",
        problem_family="linear_programming",
        formulation=LinearFormulationIR(
            kind="linear",
            variables=[
                LinearVariableIR(
                    variable_id="x",
                    name="连续产量",
                    domain="continuous",
                    lower_bound=0,
                    unit="件",
                ),
                LinearVariableIR(
                    variable_id="y",
                    name="整数批次",
                    domain="integer",
                    lower_bound=0,
                    upper_bound=3,
                    unit="批",
                ),
            ],
            objective=LinearObjectiveIR(
                direction="maximize",
                terms=[
                    LinearTermIR(variable_id="x", coefficient=1),
                    LinearTermIR(variable_id="y", coefficient=2),
                ],
            ),
            constraints=[
                IRConstraint(
                    constraint_id="capacity",
                    name="产能",
                    terms=[
                        LinearTermIR(variable_id="x", coefficient=1),
                        LinearTermIR(variable_id="y", coefficient=1),
                    ],
                    relation="<=",
                    rhs=5,
                )
            ],
        ),
        evidence=[],
        assumptions=[],
    )


def _flow_ir() -> OptimizationIR:
    problem = MinCostFlowProblem(
        nodes=[
            FlowNode(node_id="source", name="仓库", supply=5),
            FlowNode(node_id="sink", name="门店", supply=-5),
        ],
        arcs=[
            FlowArc(
                arc_id="route-1",
                from_node="source",
                to_node="sink",
                capacity=8,
                unit_cost=4,
            )
        ],
        flow_unit="箱",
        cost_unit="元/箱",
    )
    return compile_min_cost_flow_problem(problem, problem_id="flow-001")


def test_registry_selects_glop_and_accounts_for_expression_constants() -> None:
    from math_modeling_agent.optimization_registry import SolverRegistry

    result = SolverRegistry().solve(_linear_ir())

    assert result.backend_id == "ortools_glop"
    assert result.status == "OPTIMAL"
    assert result.variable_values["x"] == pytest.approx(3)
    assert result.objective_value == pytest.approx(13)


def test_registry_selects_scip_for_mixed_integer_linear_model() -> None:
    from math_modeling_agent.optimization_registry import SolverRegistry

    result = SolverRegistry().solve(_mixed_integer_ir())

    assert result.backend_id == "ortools_scip"
    assert result.status == "OPTIMAL"
    assert result.variable_values == {"x": pytest.approx(2), "y": pytest.approx(3)}
    assert result.objective_value == pytest.approx(8)


def test_registry_selects_cp_sat_for_integer_schedule_and_returns_variable_values() -> None:
    from math_modeling_agent.optimization_registry import SolverRegistry

    result = SolverRegistry().solve(_schedule_ir())

    assert result.backend_id == "ortools_cp_sat"
    assert result.status in {"OPTIMAL", "FEASIBLE"}
    assert result.objective_value == pytest.approx(480)
    assert len(result.variable_values) == 2
    assert sum(result.variable_values.values()) == pytest.approx(1)


def test_registry_uses_scip_instead_of_rounding_non_integer_schedule_coefficients() -> None:
    from math_modeling_agent.optimization_registry import SolverRegistry

    problem = _schedule_ir()
    formulation = problem.formulation
    assert isinstance(formulation, LinearFormulationIR)
    formulation.objective.terms[0].coefficient = 0.5

    result = SolverRegistry().solve(problem)

    assert result.backend_id == "ortools_scip"
    assert result.status in {"OPTIMAL", "FEASIBLE"}
    assert result.objective_value == pytest.approx(0.5)


def test_cp_sat_rejects_integer_objective_range_that_exceeds_int64() -> None:
    from math_modeling_agent.optimization_registry import _cp_sat_compatible

    problem = _schedule_ir()
    formulation = problem.formulation
    assert isinstance(formulation, LinearFormulationIR)
    formulation.objective.terms[0].coefficient = float(2**62)
    formulation.objective.terms[1].coefficient = float(2**62)

    assert not _cp_sat_compatible(formulation)


def test_registry_selects_simple_min_cost_flow_backend() -> None:
    from math_modeling_agent.optimization_registry import SolverRegistry

    result = SolverRegistry().solve(_flow_ir())

    assert result.backend_id == "ortools_simple_min_cost_flow"
    assert result.status == "OPTIMAL"
    assert result.variable_values == {"route-1": 5}
    assert type(result.variable_values["route-1"]) is int
    assert result.objective_value == 20
    assert type(result.objective_value) is int


def test_non_feasible_solver_result_does_not_include_a_fake_solution() -> None:
    from math_modeling_agent.optimization_registry import SolverRegistry

    problem = _linear_ir()
    formulation = problem.formulation
    assert isinstance(formulation, LinearFormulationIR)
    formulation.constraints.append(
        IRConstraint(
            constraint_id="contradiction",
            name="矛盾约束",
            terms=[LinearTermIR(variable_id="x", coefficient=1)],
            relation=">=",
            rhs=10,
        )
    )

    result = SolverRegistry().solve(problem)

    assert result.status == "INFEASIBLE"
    assert result.objective_value is None
    assert result.variable_values == {}


def test_glop_distinguishes_unbounded_objective_from_infeasible_constraints() -> None:
    from math_modeling_agent.optimization_registry import SolverRegistry

    problem = _linear_ir()
    formulation = problem.formulation
    assert isinstance(formulation, LinearFormulationIR)
    formulation.constraints.clear()

    result = SolverRegistry().solve(problem)

    assert result.status == "UNBOUNDED"
    assert result.objective_value is None
    assert result.variable_values == {}


def test_missing_compatible_backend_returns_unsupported_without_a_solution() -> None:
    from math_modeling_agent.optimization_registry import SolverRegistry

    result = SolverRegistry(backends=[]).solve(_linear_ir())

    assert result.status == "UNSUPPORTED_MODEL"
    assert result.backend_id == "none"
    assert result.objective_value is None
    assert result.variable_values == {}


def test_registry_uses_scip_for_non_schedule_binary_linear_model() -> None:
    from math_modeling_agent.optimization_registry import SolverRegistry

    problem = _linear_ir(variable_domain="binary", lower_bound=None)
    result = SolverRegistry().solve(problem)

    assert result.backend_id == "ortools_scip"
    assert result.status == "OPTIMAL"
