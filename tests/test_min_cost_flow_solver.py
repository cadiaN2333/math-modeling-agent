from min_cost_flow_fixtures import EXPECTED_ARC_FLOWS, make_transport_problem


def test_solver_finds_minimum_cost_warehouse_delivery() -> None:
    from math_modeling_agent.min_cost_flow_solver import solve_min_cost_flow

    result = solve_min_cost_flow(make_transport_problem())

    assert result.status == "OPTIMAL"
    assert result.arc_flows == EXPECTED_ARC_FLOWS
    assert result.total_cost == 80


def test_solver_reports_infeasible_when_total_supply_is_unbalanced() -> None:
    from math_modeling_agent.min_cost_flow_solver import solve_min_cost_flow
    from math_modeling_agent.models import MinCostFlowProblem

    raw = make_transport_problem().model_dump()
    raw["nodes"][3]["supply"] = -24
    problem = MinCostFlowProblem.model_validate(raw)

    result = solve_min_cost_flow(problem)

    assert result.status == "INFEASIBLE"
    assert result.arc_flows == {}
    assert result.total_cost is None


def test_solver_reports_infeasible_when_route_capacity_is_insufficient() -> None:
    from math_modeling_agent.min_cost_flow_solver import solve_min_cost_flow
    from math_modeling_agent.models import MinCostFlowProblem

    raw = make_transport_problem().model_dump()
    raw["arcs"][0]["capacity"] = 0
    raw["arcs"][1]["capacity"] = 0
    problem = MinCostFlowProblem.model_validate(raw)

    result = solve_min_cost_flow(problem)

    assert result.status == "INFEASIBLE"
    assert result.arc_flows == {}
    assert result.total_cost is None
