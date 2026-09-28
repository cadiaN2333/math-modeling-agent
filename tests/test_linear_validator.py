import pytest


def make_production_plan_problem():
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
                name="A非负",
                terms=[LinearTerm(variable="A", coefficient=1)],
                relation=">=",
                rhs=0,
            ),
            LinearConstraint(
                name="B非负",
                terms=[LinearTerm(variable="B", coefficient=1)],
                relation=">=",
                rhs=0,
            ),
        ],
    )


def test_linear_validator_accepts_and_recomputes_valid_solution() -> None:
    from math_modeling_agent.linear_validator import validate_linear_solution

    report = validate_linear_solution(
        make_production_plan_problem(),
        variable_values={"A": 20, "B": 60},
        reported_objective=2600,
    )

    assert report.is_valid
    assert report.errors == []
    assert report.recomputed_objective == pytest.approx(2600.0)


def test_linear_validator_rejects_constraint_violation() -> None:
    from math_modeling_agent.linear_validator import validate_linear_solution

    report = validate_linear_solution(
        make_production_plan_problem(),
        variable_values={"A": 50, "B": 50},
        reported_objective=3500,
    )

    assert not report.is_valid
    assert any("工时" in error for error in report.errors)
    assert any("原料" in error for error in report.errors)


def test_linear_validator_rejects_objective_value_mismatch() -> None:
    from math_modeling_agent.linear_validator import validate_linear_solution

    report = validate_linear_solution(
        make_production_plan_problem(),
        variable_values={"A": 20, "B": 60},
        reported_objective=2500,
    )

    assert not report.is_valid
    assert any("目标值" in error for error in report.errors)
