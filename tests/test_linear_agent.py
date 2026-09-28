import pytest


def test_linear_agent_solves_and_independently_validates_production_plan() -> None:
    from math_modeling_agent.linear_agent import run_linear_modeling
    from math_modeling_agent.models import (
        LinearConstraint,
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )

    problem = LinearProgramProblem(
        variables=[LinearVariable(name="A", unit="件"), LinearVariable(name="B", unit="件")],
        objective=LinearObjective(
            direction="maximize",
            terms=[LinearTerm(variable="A", coefficient=40), LinearTerm(variable="B", coefficient=30)],
        ),
        constraints=[
            LinearConstraint(
                name="labor",
                terms=[LinearTerm(variable="A", coefficient=2), LinearTerm(variable="B", coefficient=1)],
                relation="<=",
                rhs=100,
            ),
            LinearConstraint(
                name="material",
                terms=[LinearTerm(variable="A", coefficient=1), LinearTerm(variable="B", coefficient=1)],
                relation="<=",
                rhs=80,
            ),
            LinearConstraint(
                name="A_nonnegative",
                terms=[LinearTerm(variable="A", coefficient=1)],
                relation=">=",
                rhs=0,
            ),
            LinearConstraint(
                name="B_nonnegative",
                terms=[LinearTerm(variable="B", coefficient=1)],
                relation=">=",
                rhs=0,
            ),
        ],
    )

    result = run_linear_modeling(problem)

    assert result.solver_result.status == "OPTIMAL"
    assert result.solver_result.variable_values == pytest.approx({"A": 20.0, "B": 60.0})
    assert result.solver_result.objective_value == pytest.approx(2600.0)
    assert result.validation_report is not None
    assert result.validation_report.is_valid is True
    assert "continuous_linear_programming" in {
        method.method_id
        for method in result.method_recommendations
        if method.implementation_status == "已实现"
    }
