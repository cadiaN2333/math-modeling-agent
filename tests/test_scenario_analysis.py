import pytest
from pydantic import ValidationError


def _linear_problem(
    capacity: float,
    *,
    objective_coefficient: float = 3,
    minimum: float = 0,
):
    from math_modeling_agent.models import (
        LinearConstraint,
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )

    return LinearProgramProblem(
        variables=[LinearVariable(name="x", unit="件")],
        objective=LinearObjective(
            direction="maximize",
            terms=[LinearTerm(variable="x", coefficient=objective_coefficient)],
        ),
        constraints=[
            LinearConstraint(
                name="capacity",
                terms=[LinearTerm(variable="x", coefficient=1)],
                relation="<=",
                rhs=capacity,
            ),
            LinearConstraint(
                name="minimum",
                terms=[LinearTerm(variable="x", coefficient=1)],
                relation=">=",
                rhs=minimum,
            ),
        ],
    )


def test_scenario_request_rejects_duplicate_ids_and_changed_variable_signature() -> None:
    from math_modeling_agent.models import (
        LinearConstraint,
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )
    from math_modeling_agent.scenario_models import (
        LinearScenario,
        ScenarioAnalysisRequest,
    )

    base = _linear_problem(10)
    with pytest.raises(ValidationError, match="scenario_id"):
        ScenarioAnalysisRequest(
            base_problem=base,
            scenarios=[
                LinearScenario(
                    scenario_id="low_capacity",
                    description="降低容量",
                    problem=_linear_problem(8),
                ),
                LinearScenario(
                    scenario_id="low_capacity",
                    description="再降低容量",
                    problem=_linear_problem(6),
                ),
            ],
        )

    changed_domain = LinearProgramProblem(
        variables=[LinearVariable(name="x", unit="件", domain="integer")],
        objective=LinearObjective(
            direction="maximize",
            terms=[LinearTerm(variable="x", coefficient=3)],
        ),
        constraints=[
            LinearConstraint(
                name="capacity",
                terms=[LinearTerm(variable="x", coefficient=1)],
                relation="<=",
                rhs=8,
            ),
            LinearConstraint(
                name="minimum",
                terms=[LinearTerm(variable="x", coefficient=1)],
                relation=">=",
                rhs=0,
            ),
        ],
    )
    with pytest.raises(ValidationError, match="变量签名"):
        ScenarioAnalysisRequest(
            base_problem=base,
            scenarios=[
                LinearScenario(
                    scenario_id="integer_domain",
                    description="改变变量域",
                    problem=changed_domain,
                )
            ],
        )


def test_scenario_analysis_reports_objective_and_variable_deltas() -> None:
    from math_modeling_agent.scenario_analysis import run_scenario_analysis
    from math_modeling_agent.scenario_models import (
        LinearScenario,
        ScenarioAnalysisRequest,
    )

    request = ScenarioAnalysisRequest(
        base_problem=_linear_problem(10),
        scenarios=[
            LinearScenario(
                scenario_id="lower_capacity",
                description="产能上限降为8件",
                problem=_linear_problem(8),
            )
        ],
    )

    result = run_scenario_analysis(request)
    scenario = result.scenario_runs[0]

    assert result.base_run.solver_result.objective_value == pytest.approx(30)
    assert scenario.modeling_run.solver_result.objective_value == pytest.approx(24)
    assert scenario.objective_delta_from_base == pytest.approx(-6)
    assert scenario.relative_objective_delta_percent == pytest.approx(-20)
    assert scenario.variable_deltas_from_base == pytest.approx({"x": -2})
    assert scenario.objective_comparison_note is None


def test_scenario_with_changed_objective_does_not_report_objective_delta() -> None:
    from math_modeling_agent.scenario_analysis import run_scenario_analysis
    from math_modeling_agent.scenario_models import (
        LinearScenario,
        ScenarioAnalysisRequest,
    )

    request = ScenarioAnalysisRequest(
        base_problem=_linear_problem(10),
        scenarios=[
            LinearScenario(
                scenario_id="changed_price",
                description="单件价值改为4",
                problem=_linear_problem(8, objective_coefficient=4),
            )
        ],
    )

    scenario = run_scenario_analysis(request).scenario_runs[0]

    assert scenario.modeling_run.solver_result.status == "OPTIMAL"
    assert scenario.objective_delta_from_base is None
    assert scenario.relative_objective_delta_percent is None
    assert scenario.variable_deltas_from_base == pytest.approx({"x": -2})
    assert "目标函数" in scenario.objective_comparison_note


def test_infeasible_scenario_has_status_but_no_fabricated_deltas() -> None:
    from math_modeling_agent.scenario_analysis import run_scenario_analysis
    from math_modeling_agent.scenario_models import (
        LinearScenario,
        ScenarioAnalysisRequest,
    )

    request = ScenarioAnalysisRequest(
        base_problem=_linear_problem(10),
        scenarios=[
            LinearScenario(
                scenario_id="infeasible",
                description="最低产量超过容量",
                problem=_linear_problem(4, minimum=10),
            )
        ],
    )

    scenario = run_scenario_analysis(request).scenario_runs[0]

    assert scenario.modeling_run.solver_result.status == "INFEASIBLE"
    assert scenario.objective_delta_from_base is None
    assert scenario.variable_deltas_from_base == {}
    assert "情景没有通过 validator" in scenario.objective_comparison_note
