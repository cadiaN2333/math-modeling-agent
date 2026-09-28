from min_cost_flow_fixtures import EXPECTED_ARC_FLOWS, make_transport_problem


def test_validator_accepts_optimal_transport_plan() -> None:
    from math_modeling_agent.min_cost_flow_validator import (
        validate_min_cost_flow_solution,
    )

    report = validate_min_cost_flow_solution(
        make_transport_problem(), EXPECTED_ARC_FLOWS, reported_total_cost=80
    )

    assert report.is_valid is True
    assert report.errors == []
    assert report.recomputed_total_cost == 80


def test_validator_reports_route_capacity_violation() -> None:
    from math_modeling_agent.min_cost_flow_validator import (
        validate_min_cost_flow_solution,
    )

    flows = dict(EXPECTED_ARC_FLOWS)
    flows["W1_S1"] = 21
    report = validate_min_cost_flow_solution(
        make_transport_problem(), flows, reported_total_cost=82
    )

    assert report.is_valid is False
    assert any("容量" in error for error in report.errors)


def test_validator_reports_node_balance_violation() -> None:
    from math_modeling_agent.min_cost_flow_validator import (
        validate_min_cost_flow_solution,
    )

    flows = dict(EXPECTED_ARC_FLOWS)
    flows["W2_S1"] = 4
    report = validate_min_cost_flow_solution(
        make_transport_problem(), flows, reported_total_cost=77
    )

    assert report.is_valid is False
    assert any("供需守恒" in error for error in report.errors)


def test_validator_reports_negative_or_fractional_flow() -> None:
    from math_modeling_agent.min_cost_flow_validator import (
        validate_min_cost_flow_solution,
    )

    negative_flows = dict(EXPECTED_ARC_FLOWS)
    negative_flows["W1_S1"] = -1
    negative_report = validate_min_cost_flow_solution(
        make_transport_problem(), negative_flows, reported_total_cost=38
    )

    fractional_flows = dict(EXPECTED_ARC_FLOWS)
    fractional_flows["W1_S1"] = 1.5
    fractional_report = validate_min_cost_flow_solution(
        make_transport_problem(), fractional_flows, reported_total_cost=80
    )

    assert negative_report.is_valid is False
    assert any("非负整数" in error for error in negative_report.errors)
    assert fractional_report.is_valid is False
    assert any("非负整数" in error for error in fractional_report.errors)


def test_validator_reports_missing_and_unknown_arc_ids() -> None:
    from math_modeling_agent.min_cost_flow_validator import (
        validate_min_cost_flow_solution,
    )

    missing_flows = dict(EXPECTED_ARC_FLOWS)
    del missing_flows["W1_S2"]
    missing_report = validate_min_cost_flow_solution(
        make_transport_problem(), missing_flows, reported_total_cost=80
    )

    unknown_flows = dict(EXPECTED_ARC_FLOWS)
    unknown_flows["UNKNOWN"] = 0
    unknown_report = validate_min_cost_flow_solution(
        make_transport_problem(), unknown_flows, reported_total_cost=80
    )

    assert any("缺少路线流量" in error for error in missing_report.errors)
    assert any("未知路线" in error for error in unknown_report.errors)


def test_validator_reports_total_cost_mismatch() -> None:
    from math_modeling_agent.min_cost_flow_validator import (
        validate_min_cost_flow_solution,
    )

    report = validate_min_cost_flow_solution(
        make_transport_problem(), EXPECTED_ARC_FLOWS, reported_total_cost=81
    )

    assert report.is_valid is False
    assert report.recomputed_total_cost == 80
    assert any("总费用不一致" in error for error in report.errors)
