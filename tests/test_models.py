import pytest
from pydantic import ValidationError

from math_modeling_agent.models import (
    CoverageRequirement,
    Employee,
    SchedulingProblem,
    Shift,
)


def make_valid_problem() -> SchedulingProblem:
    return SchedulingProblem(
        employees=[
            Employee(
                employee_id="E1",
                name="员工一",
                skills={"急救"},
                max_hours=40,
            )
        ],
        shifts=[
            Shift(
                shift_id="S1",
                duration_hours=8,
            )
        ],
        coverage_requirements=[
            CoverageRequirement(
                shift_id="S1",
                minimum_employees=1,
                required_skill_counts={"急救": 1},
            )
        ],
    )


def test_valid_problem_keeps_employee_skills() -> None:
    problem = make_valid_problem()

    assert problem.employees[0].skills == {"急救"}


def test_employee_max_hours_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Employee(
            employee_id="E1",
            name="员工一",
            skills=set(),
            max_hours=0,
        )


def test_each_shift_must_have_coverage_requirement() -> None:
    with pytest.raises(ValidationError):
        SchedulingProblem(
            employees=[
                Employee(
                    employee_id="E1",
                    name="员工一",
                    skills=set(),
                    max_hours=40,
                )
            ],
            shifts=[
                Shift(
                    shift_id="S1",
                    duration_hours=8,
                )
            ],
            coverage_requirements=[],
        )


def test_coverage_cannot_reference_unknown_shift() -> None:
    with pytest.raises(ValidationError):
        SchedulingProblem(
            employees=[
                Employee(
                    employee_id="E1",
                    name="员工一",
                    skills=set(),
                    max_hours=40,
                )
            ],
            shifts=[
                Shift(
                    shift_id="S1",
                    duration_hours=8,
                )
            ],
            coverage_requirements=[
                CoverageRequirement(
                    shift_id="S2",
                    minimum_employees=1,
                    required_skill_counts={},
                )
            ],
        )


def make_valid_linear_program():
    from math_modeling_agent.models import (
        LinearConstraint,
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )

    return LinearProgramProblem(
        variables=[
            LinearVariable(name="A", unit="件"),
            LinearVariable(name="B", unit="件"),
        ],
        objective=LinearObjective(
            direction="maximize",
            terms=[
                LinearTerm(variable="A", coefficient=40),
                LinearTerm(variable="B", coefficient=30),
            ],
        ),
        constraints=[
            LinearConstraint(
                name="工时",
                terms=[
                    LinearTerm(variable="A", coefficient=2),
                    LinearTerm(variable="B", coefficient=1),
                ],
                relation="<=",
                rhs=100,
            ),
            LinearConstraint(
                name="原料",
                terms=[
                    LinearTerm(variable="A", coefficient=1),
                    LinearTerm(variable="B", coefficient=1),
                ],
                relation="<=",
                rhs=80,
            ),
            LinearConstraint(
                name="产品A非负",
                terms=[LinearTerm(variable="A", coefficient=1)],
                relation=">=",
                rhs=0,
            ),
            LinearConstraint(
                name="产品B非负",
                terms=[LinearTerm(variable="B", coefficient=1)],
                relation=">=",
                rhs=0,
            ),
        ],
    )


def test_valid_linear_program_keeps_objective_and_constraints() -> None:
    problem = make_valid_linear_program()

    assert problem.objective.direction == "maximize"
    assert len(problem.variables) == 2
    assert len(problem.constraints) == 4


def test_linear_program_rejects_duplicate_variable_names() -> None:
    import pytest
    from pydantic import ValidationError

    from math_modeling_agent.models import (
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )

    with pytest.raises(ValidationError, match="变量名不能重复"):
        LinearProgramProblem(
            variables=[
                LinearVariable(name="A", unit="件"),
                LinearVariable(name="A", unit="件"),
            ],
            objective=LinearObjective(
                direction="maximize",
                terms=[LinearTerm(variable="A", coefficient=1)],
            ),
            constraints=[],
        )


def test_linear_program_rejects_unknown_variable_references() -> None:
    import pytest
    from pydantic import ValidationError

    from math_modeling_agent.models import (
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )

    with pytest.raises(ValidationError, match="未声明的变量"):
        LinearProgramProblem(
            variables=[LinearVariable(name="A", unit="件")],
            objective=LinearObjective(
                direction="maximize",
                terms=[LinearTerm(variable="B", coefficient=1)],
            ),
            constraints=[],
        )


def test_valid_min_cost_flow_preserves_nodes_arcs_and_units() -> None:
    from min_cost_flow_fixtures import EXPECTED_ARC_FLOWS, make_transport_problem

    problem = make_transport_problem()

    assert [node.supply for node in problem.nodes] == [20, 30, -25, -25]
    assert {arc.arc_id for arc in problem.arcs} == set(EXPECTED_ARC_FLOWS)
    assert problem.flow_unit == "箱"
    assert problem.cost_unit == "元/箱"


def test_min_cost_flow_rejects_duplicate_node_ids() -> None:
    from math_modeling_agent.models import MinCostFlowProblem
    from min_cost_flow_fixtures import make_transport_problem

    raw = make_transport_problem().model_dump()
    raw["nodes"][1]["node_id"] = raw["nodes"][0]["node_id"]

    with pytest.raises(ValidationError, match="节点编号不能重复"):
        MinCostFlowProblem.model_validate(raw)


def test_min_cost_flow_rejects_duplicate_arc_ids() -> None:
    from math_modeling_agent.models import MinCostFlowProblem
    from min_cost_flow_fixtures import make_transport_problem

    raw = make_transport_problem().model_dump()
    raw["arcs"][1]["arc_id"] = raw["arcs"][0]["arc_id"]

    with pytest.raises(ValidationError, match="路线编号不能重复"):
        MinCostFlowProblem.model_validate(raw)


def test_min_cost_flow_rejects_unknown_arc_endpoint() -> None:
    from math_modeling_agent.models import MinCostFlowProblem
    from min_cost_flow_fixtures import make_transport_problem

    raw = make_transport_problem().model_dump()
    raw["arcs"][0]["to_node"] = "UNKNOWN"

    with pytest.raises(ValidationError, match="路线引用了不存在的节点"):
        MinCostFlowProblem.model_validate(raw)


def test_min_cost_flow_rejects_self_loop() -> None:
    from math_modeling_agent.models import MinCostFlowProblem
    from min_cost_flow_fixtures import make_transport_problem

    raw = make_transport_problem().model_dump()
    raw["arcs"][0]["to_node"] = raw["arcs"][0]["from_node"]

    with pytest.raises(ValidationError, match="路线不能连接节点自身"):
        MinCostFlowProblem.model_validate(raw)


def test_min_cost_flow_rejects_negative_or_fractional_capacity() -> None:
    from math_modeling_agent.models import MinCostFlowProblem
    from min_cost_flow_fixtures import make_transport_problem

    negative_capacity = make_transport_problem().model_dump()
    negative_capacity["arcs"][0]["capacity"] = -1
    with pytest.raises(ValidationError):
        MinCostFlowProblem.model_validate(negative_capacity)

    fractional_capacity = make_transport_problem().model_dump()
    fractional_capacity["arcs"][0]["capacity"] = 1.5
    with pytest.raises(ValidationError):
        MinCostFlowProblem.model_validate(fractional_capacity)


def test_min_cost_flow_rejects_fractional_supply_and_unit_cost() -> None:
    from math_modeling_agent.models import MinCostFlowProblem
    from min_cost_flow_fixtures import make_transport_problem

    fractional_supply = make_transport_problem().model_dump()
    fractional_supply["nodes"][0]["supply"] = 1.5
    with pytest.raises(ValidationError):
        MinCostFlowProblem.model_validate(fractional_supply)

    fractional_cost = make_transport_problem().model_dump()
    fractional_cost["arcs"][0]["unit_cost"] = 1.5
    with pytest.raises(ValidationError):
        MinCostFlowProblem.model_validate(fractional_cost)


def test_min_cost_flow_allows_unbalanced_supply_for_solver_to_report() -> None:
    from math_modeling_agent.models import MinCostFlowProblem
    from min_cost_flow_fixtures import make_transport_problem

    raw = make_transport_problem().model_dump()
    raw["nodes"][3]["supply"] = -24
    problem = MinCostFlowProblem.model_validate(raw)

    assert sum(node.supply for node in problem.nodes) == 1


def test_min_cost_flow_allows_parallel_routes_with_distinct_ids() -> None:
    from math_modeling_agent.models import FlowArc, MinCostFlowProblem
    from min_cost_flow_fixtures import make_transport_problem

    problem = make_transport_problem()
    parallel_route = FlowArc(
        arc_id="W1_S1_alt",
        from_node="W1",
        to_node="S1",
        capacity=5,
        unit_cost=3,
    )
    expanded_problem = MinCostFlowProblem(
        nodes=problem.nodes,
        arcs=[*problem.arcs, parallel_route],
        flow_unit=problem.flow_unit,
        cost_unit=problem.cost_unit,
    )

    matching_routes = [
        arc
        for arc in expanded_problem.arcs
        if arc.from_node == "W1" and arc.to_node == "S1"
    ]
    assert len(matching_routes) == 2


def test_linear_variable_domain_defaults_to_continuous() -> None:
    from math_modeling_agent.models import LinearVariable

    variable = LinearVariable(name="x", unit="件")

    assert variable.domain == "continuous"


@pytest.mark.parametrize("domain", ["integer", "binary"])
def test_linear_variable_accepts_discrete_domains(domain: str) -> None:
    from math_modeling_agent.models import LinearVariable

    variable = LinearVariable(name="x", unit="件", domain=domain)

    assert variable.domain == domain


def test_linear_variable_rejects_unknown_domain() -> None:
    from math_modeling_agent.models import LinearVariable

    with pytest.raises(ValidationError):
        LinearVariable(name="x", unit="件", domain="semi_continuous")
