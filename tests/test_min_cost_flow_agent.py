from min_cost_flow_fixtures import (
    EXPECTED_ARC_FLOWS,
    make_transport_problem,
)


def test_agent_solves_and_independently_validates_transport_plan() -> None:
    from math_modeling_agent.min_cost_flow_agent import run_min_cost_flow_modeling

    result = run_min_cost_flow_modeling(make_transport_problem())

    assert result.solver_result.status == "OPTIMAL"
    assert result.solver_result.arc_flows == EXPECTED_ARC_FLOWS
    assert result.solver_result.total_cost == 80
    assert result.validation_report is not None
    assert result.validation_report.is_valid is True
    assert "minimum_cost_flow" in {
        item.method_id
        for item in result.method_recommendations
        if item.implementation_status == "已实现"
    }


def test_agent_does_not_validate_infeasible_problem() -> None:
    from math_modeling_agent.min_cost_flow_agent import run_min_cost_flow_modeling
    from math_modeling_agent.models import MinCostFlowProblem

    raw = make_transport_problem().model_dump()
    raw["arcs"][0]["capacity"] = 0
    raw["arcs"][1]["capacity"] = 0
    result = run_min_cost_flow_modeling(MinCostFlowProblem.model_validate(raw))

    assert result.solver_result.status == "INFEASIBLE"
    assert result.validation_report is None
