import json


def empty_min_cost_flow_draft():
    from math_modeling_agent.analysis_agent import MinCostFlowDraft

    return MinCostFlowDraft(
        nodes=[],
        arcs=[],
        flow_unit="not_applicable",
        cost_unit="not_applicable",
    )


def make_expected_transport_draft():
    return {
        "nodes": [
            {"node_id": "W1", "name": "仓库一", "supply": 20},
            {"node_id": "W2", "name": "仓库二", "supply": 30},
            {"node_id": "S1", "name": "门店一", "supply": -25},
            {"node_id": "S2", "name": "门店二", "supply": -25},
        ],
        "arcs": [
            {
                "arc_id": "W1_S1",
                "from_node": "W1",
                "to_node": "S1",
                "capacity": 20,
                "unit_cost": 2,
            },
            {
                "arc_id": "W1_S2",
                "from_node": "W1",
                "to_node": "S2",
                "capacity": 20,
                "unit_cost": 4,
            },
            {
                "arc_id": "W2_S1",
                "from_node": "W2",
                "to_node": "S1",
                "capacity": 25,
                "unit_cost": 3,
            },
            {
                "arc_id": "W2_S2",
                "from_node": "W2",
                "to_node": "S2",
                "capacity": 30,
                "unit_cost": 1,
            },
        ],
        "flow_unit": "箱",
        "cost_unit": "元/箱",
    }


def test_eval_case_fails_when_analysis_status_is_wrong() -> None:
    from math_modeling_agent.evals import evaluate_case

    case = {
        "id": "lp_not_supported",
        "family": "linear_programming",
        "expected_analysis_status": "unsupported",
    }
    payload = {
        "analysis": {"status": "ready"},
        "modeling_run": None,
    }

    result = evaluate_case(case, payload)

    assert result["passed"] is False
    assert result["checks"][0]["name"] == "analysis_status"
    assert result["checks"][0]["passed"] is False


def test_eval_case_checks_unsupported_boundary_keywords() -> None:
    from math_modeling_agent.evals import evaluate_case

    case = {
        "id": "lp_not_supported",
        "family": "linear_programming",
        "expected_analysis_status": "unsupported",
        "required_text_groups": [
            {
                "fields": ["known_facts"],
                "any_of": ["线性规划", "生产计划"],
            },
            {
                "fields": ["unsupported_reasons"],
                "any_of": ["只支持员工排班", "排班系统"],
            },
        ],
    }
    payload = {
        "analysis": {
            "status": "unsupported",
            "summary": "当前系统只支持员工排班。",
            "known_facts": ["用户提出生产计划线性规划问题。"],
            "missing_information": [],
            "clarifying_questions": [],
            "unsupported_reasons": ["当前系统只支持员工排班，线性规划尚未实现。"],
        },
        "modeling_run": None,
    }

    result = evaluate_case(case, payload)

    assert result["passed"] is True
    assert all(check["passed"] for check in result["checks"])


def test_eval_case_does_not_accept_user_facts_as_unsupported_reason() -> None:
    from math_modeling_agent.evals import evaluate_case

    case = {
        "id": "lp_not_supported",
        "family": "linear_programming",
        "expected_analysis_status": "unsupported",
        "required_text_groups": [
            {
                "fields": ["unsupported_reasons"],
                "any_of": ["只支持员工排班", "尚未支持线性规划"],
            }
        ],
    }
    payload = {
        "analysis": {
            "status": "unsupported",
            "summary": "生产计划问题。",
            "known_facts": ["用户提出线性规划生产计划。"],
            "missing_information": [],
            "clarifying_questions": [],
            "unsupported_reasons": ["需求信息已整理。"],
        },
        "modeling_run": None,
    }

    result = evaluate_case(case, payload)

    assert result["passed"] is False


def test_eval_case_requires_valid_solver_and_method_for_ready_schedule() -> None:
    from math_modeling_agent.evals import evaluate_case

    model = {
        "employees": [
            {
                "employee_id": "E1",
                "name": "林晓",
                "skills": ["急救"],
                "max_hours": 8.0,
            }
        ],
        "shifts": [{"shift_id": "S1", "duration_hours": 8.0}],
        "coverage_requirements": [
            {
                "shift_id": "S1",
                "minimum_employees": 1,
                "skill_requirements": [
                    {"skill": "急救", "minimum_count": 1}
                ],
            }
        ],
    }
    case = {
        "id": "ready_schedule",
        "family": "scheduling",
        "expected_analysis_status": "ready",
        "expected_model": model,
        "expected_solver_statuses": ["OPTIMAL", "FEASIBLE"],
        "expected_method_id": "cp_sat_scheduling",
        "expected_validation": "valid",
    }
    payload = {
        "analysis": {"status": "ready", "scheduling_draft": model},
        "modeling_run": {
            "solver_result": {"status": "OPTIMAL"},
            "validation_report": {"is_valid": True, "errors": []},
            "method_recommendations": [
                {
                    "method_id": "cp_sat_scheduling",
                    "implementation_status": "已实现",
                }
            ],
        },
    }

    result = evaluate_case(case, payload)

    assert result["passed"] is True
    assert {check["name"] for check in result["checks"]} == {
        "analysis_status",
        "structured_model",
        "solver_status",
        "implemented_method",
        "validator",
    }


def test_required_assignment_report_shows_assigned_and_missing_pairs() -> None:
    from math_modeling_agent.evals import evaluate_case

    case = {
        "id": "required_assignment_report",
        "family": "scheduling",
        "expected_analysis_status": "ready",
        "expected_solver_statuses": ["OPTIMAL"],
        "expected_method_id": "cp_sat_scheduling",
        "expected_validation": "valid",
        "required_assignment_pairs": [["E1", "S1"]],
    }
    payload = {
        "analysis": {"status": "ready"},
        "modeling_run": {
            "solver_result": {
                "status": "OPTIMAL",
                "assignments": [
                    {"employee_id": "E1", "shift_id": "S1"},
                    {"employee_id": "E2", "shift_id": "S2"},
                ],
            },
            "method_recommendations": [
                {
                    "method_id": "cp_sat_scheduling",
                    "implementation_status": "已实现",
                }
            ],
            "validation_report": {"is_valid": True, "errors": []},
        },
    }

    result = evaluate_case(case, payload)
    check = next(
        item for item in result["checks"] if item["name"] == "required_assignments"
    )

    assert check["passed"] is True
    assert check["actual"] == {
        "assigned": [["E1", "S1"], ["E2", "S2"]],
        "missing": [],
    }


def test_eval_case_compares_structured_model_without_list_order_sensitivity() -> None:
    from math_modeling_agent.evals import evaluate_case

    expected_model = {
        "employees": [
            {"employee_id": "E1", "name": "林晓", "skills": ["急救"], "max_hours": 8.0},
            {"employee_id": "E2", "name": "陈立", "skills": [], "max_hours": 8.0},
        ],
        "shifts": [
            {"shift_id": "S1", "duration_hours": 8.0},
            {"shift_id": "S2", "duration_hours": 8.0},
        ],
        "coverage_requirements": [
            {
                "shift_id": "S1",
                "minimum_employees": 1,
                "skill_requirements": [{"skill": "急救", "minimum_count": 1}],
            },
            {"shift_id": "S2", "minimum_employees": 1, "skill_requirements": []},
        ],
    }
    actual_model = {
        "employees": list(reversed(expected_model["employees"])),
        "shifts": list(reversed(expected_model["shifts"])),
        "coverage_requirements": list(reversed(expected_model["coverage_requirements"])),
    }
    result = evaluate_case(
        {"id": "order_independent", "family": "scheduling", "expected_analysis_status": "ready", "expected_model": expected_model},
        {
            "analysis": {"status": "ready", "scheduling_draft": actual_model},
            "modeling_run": None,
        },
    )

    model_check = next(check for check in result["checks"] if check["name"] == "structured_model")
    assert model_check["passed"] is True


def test_eval_case_scores_linear_program_structure_values_and_objective() -> None:
    from math_modeling_agent.evals import evaluate_case

    expected_model = {
        "variables": [
            {"name": "A", "unit": "件"},
            {"name": "B", "unit": "件"},
        ],
        "objective_direction": "maximize",
        "objective_terms": [
            {"variable": "A", "coefficient": 40.0},
            {"variable": "B", "coefficient": 30.0},
        ],
        "constraints": [
            {
                "constraint_id": "labor",
                "terms": [
                    {"variable": "A", "coefficient": 2.0},
                    {"variable": "B", "coefficient": 1.0},
                ],
                "relation": "<=",
                "rhs": 100.0,
            }
        ],
    }
    case = {
        "id": "lp_production",
        "family": "linear_programming",
        "expected_analysis_status": "ready",
        "expected_model": expected_model,
        "expected_solver_statuses": ["OPTIMAL"],
        "expected_method_id": "continuous_linear_programming",
        "expected_validation": "valid",
        "expected_variable_values": {"A": 20.0, "B": 60.0},
        "expected_objective_value": 2600.0,
    }
    actual_model = {
        **expected_model,
        "variables": [
            {**variable, "domain": "continuous"}
            for variable in reversed(expected_model["variables"])
        ],
        "objective_terms": list(reversed(expected_model["objective_terms"])),
        "constraints": [
            {
                **expected_model["constraints"][0],
                "terms": list(reversed(expected_model["constraints"][0]["terms"])),
            }
        ],
    }
    payload = {
        "analysis": {
            "status": "ready",
            "problem_family": "linear_programming",
            "linear_program_draft": actual_model,
        },
        "modeling_run": {
            "solver_result": {
                "status": "OPTIMAL",
                "variable_values": {"A": 20.0000001, "B": 59.9999999},
                "objective_value": 2600.0000005,
            },
            "validation_report": {"is_valid": True},
            "method_recommendations": [
                {
                    "method_id": "continuous_linear_programming",
                    "implementation_status": "已实现",
                }
            ],
        },
    }

    result = evaluate_case(case, payload)

    assert result["passed"] is True
    assert {check["name"] for check in result["checks"]} >= {
        "structured_model",
        "variable_values",
        "objective_value",
    }


def test_eval_scores_integer_linear_program_case() -> None:
    from math_modeling_agent.evals import evaluate_case, load_eval_cases

    case = next(
        case
        for case in load_eval_cases()
        if case["id"] == "mixed_integer_linear_programming_integer_product"
    )
    payload = {
        "analysis": {
            "status": "ready",
            "problem_family": "linear_programming",
            "linear_program_draft": case["expected_model"],
        },
        "modeling_run": {
            "solver_result": {
                "status": "OPTIMAL",
                "solver_name": "SCIP",
                "variable_values": {"x": 2.0},
                "objective_value": 2.0,
            },
            "validation_report": {"is_valid": True},
            "method_recommendations": [
                {
                    "method_id": "integer_programming",
                    "implementation_status": "已实现",
                }
            ],
        },
    }

    result = evaluate_case(case, payload)

    assert result["passed"] is True
    assert "solver_status" in {item["name"] for item in result["checks"]}


def test_eval_scores_network_flow_model_routes_and_total_cost() -> None:
    from math_modeling_agent.evals import evaluate_case
    from min_cost_flow_fixtures import make_transport_eval_payload

    case = {
        "id": "minimum_cost_flow_warehouse_delivery",
        "family": "transportation",
        "expected_analysis_status": "ready",
        "expected_model": make_expected_transport_draft(),
        "expected_solver_statuses": ["OPTIMAL"],
        "expected_method_id": "minimum_cost_flow",
        "expected_validation": "valid",
        "expected_route_flows": [
            ["W1", "S1", 20],
            ["W1", "S2", 0],
            ["W2", "S1", 5],
            ["W2", "S2", 25],
        ],
        "expected_total_cost": 80,
    }

    result = evaluate_case(case, make_transport_eval_payload())

    assert result["passed"] is True
    assert {check["name"] for check in result["checks"]} >= {
        "structured_model",
        "route_flows",
        "total_cost",
    }


def test_eval_ignores_min_cost_flow_arc_ids_and_order() -> None:
    from math_modeling_agent.evals import evaluate_case
    from min_cost_flow_fixtures import make_transport_eval_payload

    expected_flows = [
        ["W1", "S1", 20],
        ["W1", "S2", 0],
        ["W2", "S1", 5],
        ["W2", "S2", 25],
    ]
    case = {
        "id": "minimum_cost_flow_order_independent",
        "family": "transportation",
        "expected_analysis_status": "ready",
        "expected_model": make_expected_transport_draft(),
        "expected_solver_statuses": ["OPTIMAL"],
        "expected_method_id": "minimum_cost_flow",
        "expected_validation": "valid",
        "expected_route_flows": expected_flows,
        "expected_total_cost": 80,
    }
    payload = make_transport_eval_payload()
    draft = payload["analysis"]["minimum_cost_flow_draft"]
    old_flows = payload["modeling_run"]["solver_result"]["arc_flows"]
    reversed_arcs = list(reversed(draft["arcs"]))
    new_flows = {}
    for index, arc in enumerate(reversed_arcs, start=1):
        old_id = arc["arc_id"]
        new_id = f"R{index}"
        arc["arc_id"] = new_id
        new_flows[new_id] = old_flows[old_id]
    draft["arcs"] = reversed_arcs
    draft["nodes"] = list(reversed(draft["nodes"]))
    payload["modeling_run"]["solver_result"]["arc_flows"] = new_flows

    result = evaluate_case(case, payload)

    assert result["passed"] is True


def test_eval_rejects_wrong_route_flow_and_wrong_total_cost() -> None:
    from math_modeling_agent.evals import evaluate_case
    from min_cost_flow_fixtures import make_transport_eval_payload

    case = {
        "id": "minimum_cost_flow_wrong_result",
        "family": "transportation",
        "expected_analysis_status": "ready",
        "expected_model": make_expected_transport_draft(),
        "expected_solver_statuses": ["OPTIMAL"],
        "expected_method_id": "minimum_cost_flow",
        "expected_validation": "valid",
        "expected_route_flows": [
            ["W1", "S1", 20],
            ["W1", "S2", 0],
            ["W2", "S1", 5],
            ["W2", "S2", 25],
        ],
        "expected_total_cost": 80,
    }

    wrong_route = make_transport_eval_payload()
    wrong_route["modeling_run"]["solver_result"]["arc_flows"]["W2_S1"] = 4
    wrong_route_result = evaluate_case(case, wrong_route)
    route_check = next(
        check for check in wrong_route_result["checks"] if check["name"] == "route_flows"
    )
    assert wrong_route_result["passed"] is False
    assert route_check["passed"] is False

    wrong_cost = make_transport_eval_payload()
    wrong_cost["modeling_run"]["solver_result"]["total_cost"] = 81
    wrong_cost_result = evaluate_case(case, wrong_cost)
    cost_check = next(
        check for check in wrong_cost_result["checks"] if check["name"] == "total_cost"
    )
    assert wrong_cost_result["passed"] is False
    assert cost_check["passed"] is False


def test_transport_eval_payload_uses_an_independent_arc_flow_mapping() -> None:
    from min_cost_flow_fixtures import make_transport_eval_payload

    expected_flows = {
        "W1_S1": 20,
        "W1_S2": 0,
        "W2_S1": 5,
        "W2_S2": 25,
    }

    first_payload = make_transport_eval_payload()
    first_payload["modeling_run"]["solver_result"]["arc_flows"]["W2_S1"] = 4
    second_payload = make_transport_eval_payload()

    assert (
        second_payload["modeling_run"]["solver_result"]["arc_flows"]
        == expected_flows
    )


def test_live_eval_routes_ready_min_cost_flow_to_flow_adapter(monkeypatch, capsys) -> None:
    from math_modeling_agent import evals
    from min_cost_flow_fixtures import make_ready_analysis, make_transport_eval_payload

    case = {
        "id": "minimum_cost_flow_route_dispatch",
        "family": "transportation",
        "request": "最小化运输费用。",
        "expected_analysis_status": "ready",
        "expected_model": make_expected_transport_draft(),
        "expected_solver_statuses": ["OPTIMAL"],
        "expected_method_id": "minimum_cost_flow",
        "expected_validation": "valid",
        "expected_route_flows": [
            ["W1", "S1", 20],
            ["W1", "S2", 0],
            ["W2", "S1", 5],
            ["W2", "S2", 25],
        ],
        "expected_total_cost": 80,
    }
    analysis = make_ready_analysis()
    calls = []
    modeling_run = make_transport_eval_payload()["modeling_run"]
    monkeypatch.setattr(evals, "load_eval_cases", lambda: [case])
    monkeypatch.setattr(evals, "to_minimum_cost_flow_problem", lambda item: "flow-problem")
    monkeypatch.setattr(
        evals,
        "run_min_cost_flow_modeling",
        lambda problem: calls.append(problem) or object(),
    )
    monkeypatch.setattr(evals, "asdict", lambda result: modeling_run)

    exit_code = evals.main(
        ["--live", "--case-id", case["id"]],
        analyzer=lambda request: analysis,
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert calls == ["flow-problem"]
    assert report["results"][0]["passed"] is True


def test_live_eval_routes_ready_linear_program_to_lp_adapter(monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    from math_modeling_agent import evals

    case = next(
        item
        for item in evals.load_eval_cases()
        if item["id"] == "linear_programming_production_plan"
    )
    analysis_data = {
        "status": "ready",
        "problem_family": "linear_programming",
        "linear_program_draft": case["expected_model"],
    }
    analysis = SimpleNamespace(
        status="ready",
        problem_family="linear_programming",
        model_dump=lambda mode: analysis_data,
    )
    calls = []
    modeling_run = {
        "solver_result": {
            "status": "OPTIMAL",
            "variable_values": {"A": 20.0, "B": 60.0},
            "objective_value": 2600.0,
        },
        "validation_report": {"is_valid": True, "errors": []},
        "method_recommendations": [
            {
                "method_id": "continuous_linear_programming",
                "implementation_status": "已实现",
            }
        ],
    }
    monkeypatch.setattr(evals, "to_linear_program_problem", lambda item: "lp-problem")
    monkeypatch.setattr(
        evals,
        "run_linear_modeling",
        lambda problem: calls.append(problem) or object(),
    )
    monkeypatch.setattr(evals, "asdict", lambda result: modeling_run)

    exit_code = evals.main(
        ["--live", "--case-id", case["id"]],
        analyzer=lambda request: analysis,
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert calls == ["lp-problem"]
    assert report["results"][0]["passed"] is True


def test_eval_dataset_has_twelve_unique_valid_domain_cases() -> None:
    from math_modeling_agent.evals import load_eval_cases

    cases = load_eval_cases()
    case_ids = [case["id"] for case in cases]

    assert len(cases) == 12
    assert len(case_ids) == len(set(case_ids))
    assert {case["family"] for case in cases} == {
        "scheduling",
        "linear_programming",
        "transportation",
    }
    assert next(
        case for case in cases if case["id"] == "linear_programming_production_plan"
    )["expected_analysis_status"] == "ready"
    integer_case = next(
        case
        for case in cases
        if case["id"] == "mixed_integer_linear_programming_integer_product"
    )
    assert integer_case["expected_method_id"] == "integer_programming"
    assert "按件计" in integer_case["request"]
    assert next(
        case for case in cases if case["id"] == "minimum_cost_flow_warehouse_delivery"
    )["expected_analysis_status"] == "ready"
    assert next(
        case for case in cases if case["id"] == "minimum_cost_flow_infeasible_capacity"
    )["expected_solver_statuses"] == ["INFEASIBLE"]
    assert next(
        case for case in cases if case["id"] == "unsupported_multicommodity_flow"
    )["expected_analysis_status"] == "unsupported"
    assert all(
        case.get("expected_model")
        and case.get("expected_method_id")
        in {"continuous_linear_programming", "integer_programming"}
        for case in cases
        if case["family"] == "linear_programming"
        and case["expected_analysis_status"] == "ready"
    )
    assert all(
        case.get("expected_model")
        and case.get("expected_route_flows")
        and case.get("expected_total_cost") is not None
        and case.get("expected_method_id") == "minimum_cost_flow"
        for case in cases
        if case["family"] == "transportation"
        and case["expected_analysis_status"] == "ready"
        and case["expected_validation"] == "valid"
    )
    assert all(case["request"].strip() for case in cases)
    assert all(case["expected_analysis_status"] in {
        "ready",
        "needs_clarification",
        "unsupported",
    } for case in cases)
    assert all(
        "expected_solver_statuses" in case
        for case in cases
        if case["expected_analysis_status"] == "ready"
    )


def test_eval_loader_rejects_feasible_network_case_without_route_flows(tmp_path) -> None:
    import json
    from pathlib import Path
    import pytest

    from math_modeling_agent.evals import load_eval_cases

    cases_path = Path(__file__).resolve().parents[1] / "evals" / "cases.json"
    dataset = json.loads(cases_path.read_text(encoding="utf-8"))
    case = next(
        item
        for item in dataset["cases"]
        if item["id"] == "minimum_cost_flow_warehouse_delivery"
    )
    del case["expected_route_flows"]
    invalid_path = tmp_path / "网络流缺少路线期望.json"
    invalid_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="必须声明路线流量"):
        load_eval_cases(invalid_path)


def test_evals_cli_defaults_to_offline_without_calling_analyzer(capsys) -> None:
    from math_modeling_agent.evals import main

    def fail_if_called(_request: str):
        raise AssertionError("默认模式不得调用模型分析器")

    exit_code = main([], analyzer=fail_if_called)
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["mode"] == "offline_validation"
    assert report["case_count"] == 12
    assert report["api_calls"] == 0


def test_live_eval_uses_injected_analyzer_for_unsupported_case(capsys) -> None:
    from math_modeling_agent.analysis_agent import (
        LinearProgramDraft,
        ProblemAnalysis,
        SchedulingDraft,
    )
    from math_modeling_agent.evals import load_eval_cases, main

    case = next(
        item
        for item in load_eval_cases()
        if item["id"] == "unsupported_multicommodity_flow"
    )
    analysis = ProblemAnalysis(
        status="unsupported",
        problem_family="minimum_cost_flow",
        summary="这是多商品共享路线容量的运输网络流问题。",
        known_facts=["用户希望同时配送商品A和商品B。"],
        missing_information=[],
        clarifying_questions=[],
        unsupported_reasons=["当前版本尚未支持多商品流和共享路线容量的约束。"],
        subtasks=[],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
        linear_program_draft=LinearProgramDraft(
            variables=[],
            objective_direction="not_applicable",
            objective_terms=[],
            constraints=[],
        ),
        minimum_cost_flow_draft=empty_min_cost_flow_draft(),
    )
    received_requests = []

    def fake_analyzer(request: str):
        received_requests.append(request)
        return analysis

    exit_code = main(
        ["--live", "--case-id", case["id"]],
        analyzer=fake_analyzer,
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert received_requests == [case["request"]]
    assert report["api_calls"] == 1
    assert report["passed_count"] == 1
    assert report["results"][0]["passed"] is True


def test_live_eval_runs_local_solver_and_reports_ready_metrics(monkeypatch, capsys) -> None:
    import json
    from types import SimpleNamespace

    from math_modeling_agent import evals

    case = next(
        item
        for item in evals.load_eval_cases()
        if item["id"] == "schedule_three_staff_feasible"
    )
    analysis_data = {
        "status": "ready",
        "scheduling_draft": case["expected_model"],
    }
    analysis = SimpleNamespace(
        status="ready",
        problem_family="employee_scheduling",
        model_dump=lambda mode: analysis_data,
    )
    calls = []
    modeling_run = {
        "solver_result": {
            "status": "OPTIMAL",
            "assignments": [
                {"employee_id": "E1", "shift_id": "S1"},
                {"employee_id": "E3", "shift_id": "S2"},
            ],
        },
        "validation_report": {"is_valid": True, "errors": []},
        "method_recommendations": [
            {
                "method_id": "cp_sat_scheduling",
                "implementation_status": "已实现",
            }
        ],
    }
    monkeypatch.setattr(evals, "to_scheduling_problem", lambda item: "test-problem")
    monkeypatch.setattr(
        evals,
        "run_modeling",
        lambda problem: calls.append(problem) or object(),
    )
    monkeypatch.setattr(evals, "asdict", lambda result: modeling_run)

    exit_code = evals.main(
        ["--live", "--case-id", case["id"]],
        analyzer=lambda request: analysis,
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert calls == ["test-problem"]
    assert report["results"][0]["passed"] is True
    assert report["analysis_status_accuracy"] == 1.0
    assert report["solver_status_accuracy"] == 1.0
    assert report["implemented_method_hit_rate"] == 1.0
    assert report["validator_pass_rate"] == 1.0


def test_live_eval_does_not_print_exception_text(capsys) -> None:
    import json

    from math_modeling_agent.evals import main

    def fail_with_sensitive_text(_request: str):
        raise RuntimeError("test-secret-must-not-be-printed")

    exit_code = main(
        ["--live", "--case-id", "clarify_missing_employee_max_hours"],
        analyzer=fail_with_sensitive_text,
    )
    output = capsys.readouterr().out
    report = json.loads(output)

    assert exit_code == 1
    assert "test-secret-must-not-be-printed" not in output
    assert report["error_count"] == 1
    assert report["results"][0]["error_type"] == "RuntimeError"
