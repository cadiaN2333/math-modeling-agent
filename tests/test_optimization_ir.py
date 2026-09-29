from copy import deepcopy

import pytest
from pydantic import ValidationError

from math_modeling_agent.optimization_ir import OptimizationIR


def _linear_payload() -> dict:
    return {
        "schema_version": "1",
        "problem_id": "staff-plan-001",
        "problem_family": "employee_scheduling",
        "formulation": {
            "kind": "linear",
            "variables": [
                {
                    "variable_id": "var-x",
                    "name": "regular_hours",
                    "domain": "continuous",
                    "lower_bound": 0,
                    "upper_bound": 40,
                    "unit": "hour",
                    "source_fact_ids": ["fact-1"],
                },
                {
                    "variable_id": "var-y",
                    "name": "overtime_hours",
                    "domain": "integer",
                    "lower_bound": 0,
                    "upper_bound": 10,
                    "unit": "hour",
                },
                {
                    "variable_id": "var-z",
                    "name": "assigned",
                    "domain": "binary",
                    "lower_bound": 0,
                    "upper_bound": 1,
                    "unit": "boolean",
                },
            ],
            "objective": {
                "direction": "minimize",
                "terms": [
                    {"variable_id": "var-x", "coefficient": 12.5},
                    {"variable_id": "var-y", "coefficient": 18},
                    {"variable_id": "var-z", "coefficient": 3},
                ],
                "constant": 2.5,
                "unit": "currency",
            },
            "constraints": [
                {
                    "constraint_id": "constraint-coverage",
                    "name": "weekly_coverage",
                    "terms": [
                        {"variable_id": "var-x", "coefficient": 1},
                        {"variable_id": "var-y", "coefficient": 1},
                    ],
                    "relation": ">=",
                    "rhs": 35,
                    "constant": 1,
                    "unit": "hour",
                    "source_fact_ids": ["fact-1"],
                },
                {
                    "constraint_id": "constraint-budget",
                    "name": "budget",
                    "terms": [{"variable_id": "var-x", "coefficient": 2}],
                    "relation": "<=",
                    "rhs": 80,
                    "unit": "currency",
                },
                {
                    "constraint_id": "constraint-fixed",
                    "name": "fixed assignment",
                    "terms": [{"variable_id": "var-z", "coefficient": 1}],
                    "relation": "==",
                    "rhs": 1,
                    "unit": "boolean",
                },
            ],
        },
        "evidence": [
            {
                "source_id": "fact-1",
                "locator": "policy.md#weekly-hours",
                "title": "每周工时规则",
            }
        ],
        "assumptions": [
            {
                "assumption_id": "assumption-1",
                "statement": "加班工时按整小时计",
                "source_fact_ids": ["fact-1"],
            }
        ],
    }


def _network_payload() -> dict:
    return {
        "schema_version": "1",
        "problem_id": "shipment-001",
        "problem_family": "minimum_cost_flow",
        "formulation": {
            "kind": "network_flow",
            "nodes": [
                {"node_id": "source", "name": "仓库", "supply": 5},
                {"node_id": "sink", "name": "门店", "supply": -5},
            ],
            "arcs": [
                {
                    "arc_id": "route-1",
                    "from_node": "source",
                    "to_node": "sink",
                    "capacity": 8,
                    "unit_cost": 4,
                }
            ],
            "flow_unit": "箱",
            "cost_unit": "元/箱",
        },
        "evidence": [],
        "assumptions": [],
    }


def test_linear_ir_round_trips_with_affine_objective_and_stable_ids() -> None:
    ir = OptimizationIR.model_validate(_linear_payload())
    restored = OptimizationIR.model_validate_json(ir.model_dump_json())

    assert restored == ir
    assert restored.schema_version == "1"
    assert restored.formulation.kind == "linear"
    assert [item.domain for item in restored.formulation.variables] == [
        "continuous",
        "integer",
        "binary",
    ]
    assert restored.formulation.objective.constant == 2.5
    assert {item.relation for item in restored.formulation.constraints} == {
        "<=",
        ">=",
        "==",
    }


def test_network_flow_ir_round_trips_without_losing_network_structure() -> None:
    ir = OptimizationIR.model_validate(_network_payload())
    restored = OptimizationIR.model_validate_json(ir.model_dump_json())

    assert restored == ir
    assert restored.formulation.kind == "network_flow"
    assert restored.formulation.nodes[0].supply == 5
    assert restored.formulation.arcs[0].capacity == 8
    assert restored.formulation.arcs[0].unit_cost == 4
    assert restored.formulation.flow_unit == "箱"
    assert restored.formulation.cost_unit == "元/箱"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(schema_version="2"), "schema_version"),
        (
            lambda p: p["formulation"]["variables"].append(
                deepcopy(p["formulation"]["variables"][0])
            ),
            "决策变量 ID 不能重复",
        ),
        (
            lambda p: p["formulation"]["constraints"].append(
                deepcopy(p["formulation"]["constraints"][0])
            ),
            "约束 ID 不能重复",
        ),
        (
            lambda p: p["formulation"]["objective"]["terms"].append(
                {"variable_id": "missing-variable", "coefficient": 1}
            ),
            "未声明变量",
        ),
        (
            lambda p: p["formulation"]["variables"][0].update(domain="semi-continuous"),
            "domain",
        ),
        (
            lambda p: p["formulation"]["variables"][0].update(
                lower_bound=12, upper_bound=3
            ),
            "上下界",
        ),
        (
            lambda p: p["formulation"]["objective"]["terms"][0].update(
                coefficient=float("nan")
            ),
            "有限",
        ),
        (
            lambda p: p["formulation"]["constraints"][0].update(rhs=float("inf")),
            "有限",
        ),
        (
            lambda p: p["formulation"]["variables"][0].update(
                source_fact_ids=["missing-fact"]
            ),
            "证据",
        ),
        (
            lambda p: p["assumptions"][0].update(source_fact_ids=["missing-fact"]),
            "证据",
        ),
        (
            lambda p: p["formulation"].update(unexpected_field=True),
            "extra",
        ),
    ],
)
def test_linear_ir_rejects_invalid_structure(mutate, message: str) -> None:
    payload = _linear_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        OptimizationIR.model_validate(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda p: p["formulation"]["nodes"].append(
                deepcopy(p["formulation"]["nodes"][0])
            ),
            "节点编号不能重复",
        ),
        (
            lambda p: p["formulation"]["arcs"][0].update(to_node="unknown"),
            "不存在的节点",
        ),
        (
            lambda p: p["formulation"]["arcs"][0].update(to_node="source"),
            "自身",
        ),
        (
            lambda p: p["formulation"]["arcs"][0].update(capacity=-1),
            "capacity",
        ),
        (
            lambda p: p["formulation"].update(unexpected_field=True),
            "extra",
        ),
        (
            lambda p: p.update(problem_family="linear_programming"),
            "问题类型",
        ),
    ],
)
def test_network_flow_ir_rejects_invalid_structure(mutate, message: str) -> None:
    payload = _network_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        OptimizationIR.model_validate(payload)


def test_top_level_rejects_extra_fields_and_duplicate_evidence_ids() -> None:
    payload = _linear_payload()
    payload["unexpected"] = "no"
    with pytest.raises(ValidationError, match="extra"):
        OptimizationIR.model_validate(payload)

    payload = _linear_payload()
    payload["evidence"].append(deepcopy(payload["evidence"][0]))
    with pytest.raises(ValidationError, match="证据"):
        OptimizationIR.model_validate(payload)


def test_assumption_ids_must_be_unique() -> None:
    payload = _linear_payload()
    payload["assumptions"].append(deepcopy(payload["assumptions"][0]))

    with pytest.raises(ValidationError, match="假设"):
        OptimizationIR.model_validate(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["formulation"]["variables"][0].update(unexpected=True),
        lambda p: p["formulation"]["objective"].update(unexpected=True),
        lambda p: p["formulation"]["objective"]["terms"][0].update(unexpected=True),
        lambda p: p["formulation"]["constraints"][0].update(unexpected=True),
        lambda p: p["evidence"][0].update(unexpected=True),
        lambda p: p["assumptions"][0].update(unexpected=True),
    ],
)
def test_each_linear_ir_input_model_forbids_extra_fields(mutate) -> None:
    payload = _linear_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match="extra"):
        OptimizationIR.model_validate(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["formulation"]["nodes"][0].update(unexpected=True),
        lambda p: p["formulation"]["arcs"][0].update(unexpected=True),
    ],
)
def test_each_network_ir_input_model_forbids_extra_fields(mutate) -> None:
    payload = _network_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match="extra"):
        OptimizationIR.model_validate(payload)
