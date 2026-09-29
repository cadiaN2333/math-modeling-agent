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


def test_glop_solves_production_plan_to_expected_optimum() -> None:
    from math_modeling_agent.linear_solver import solve_linear_program

    result = solve_linear_program(make_production_plan_problem())

    assert result.status == "OPTIMAL"
    assert result.solver_name == "GLOP"
    assert result.variable_values["A"] == pytest.approx(20.0, abs=1e-6)
    assert result.variable_values["B"] == pytest.approx(60.0, abs=1e-6)
    assert result.objective_value == pytest.approx(2600.0, abs=1e-6)


def test_glop_reports_infeasible_lp_without_solution_values() -> None:
    from math_modeling_agent.linear_solver import solve_linear_program
    from math_modeling_agent.models import (
        LinearConstraint,
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )

    problem = LinearProgramProblem(
        variables=[LinearVariable(name="x", unit="件")],
        objective=LinearObjective(
            direction="maximize",
            terms=[LinearTerm(variable="x", coefficient=1)],
        ),
        constraints=[
            LinearConstraint(
                name="最低产量",
                terms=[LinearTerm(variable="x", coefficient=1)],
                relation=">=",
                rhs=10,
            ),
            LinearConstraint(
                name="容量上限",
                terms=[LinearTerm(variable="x", coefficient=1)],
                relation="<=",
                rhs=5,
            ),
        ],
    )

    result = solve_linear_program(problem)

    assert result.status == "INFEASIBLE"
    assert result.variable_values == {}
    assert result.objective_value is None


def test_glop_reports_unbounded_lp_without_solution_values() -> None:
    from math_modeling_agent.linear_solver import solve_linear_program
    from math_modeling_agent.models import (
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )

    problem = LinearProgramProblem(
        variables=[LinearVariable(name="x", unit="件")],
        objective=LinearObjective(
            direction="maximize",
            terms=[LinearTerm(variable="x", coefficient=1)],
        ),
        constraints=[],
    )

    result = solve_linear_program(problem)

    assert result.status == "UNBOUNDED"
    assert result.variable_values == {}
    assert result.objective_value is None


def test_scip_solves_integer_linear_program() -> None:
    from math_modeling_agent.linear_solver import solve_linear_program
    from math_modeling_agent.models import (
        LinearConstraint,
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )

    problem = LinearProgramProblem(
        variables=[LinearVariable(name="x", unit="件", domain="integer")],
        objective=LinearObjective(
            direction="maximize",
            terms=[LinearTerm(variable="x", coefficient=1)],
        ),
        constraints=[
            LinearConstraint(
                name="容量",
                terms=[LinearTerm(variable="x", coefficient=1)],
                relation="<=",
                rhs=2.5,
            ),
            LinearConstraint(
                name="非负",
                terms=[LinearTerm(variable="x", coefficient=1)],
                relation=">=",
                rhs=0,
            ),
        ],
    )

    result = solve_linear_program(problem)

    assert result.status == "OPTIMAL"
    assert result.solver_name == "SCIP"
    assert result.variable_values["x"] == pytest.approx(2)
    assert result.objective_value == pytest.approx(2)


def test_scip_solves_mixed_continuous_and_binary_variables() -> None:
    from math_modeling_agent.linear_solver import solve_linear_program
    from math_modeling_agent.models import (
        LinearConstraint,
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )

    problem = LinearProgramProblem(
        variables=[
            LinearVariable(name="x", unit="件"),
            LinearVariable(name="y", unit="是否启用", domain="binary"),
        ],
        objective=LinearObjective(
            direction="maximize",
            terms=[
                LinearTerm(variable="x", coefficient=3),
                LinearTerm(variable="y", coefficient=5),
            ],
        ),
        constraints=[
            LinearConstraint(
                name="资源一",
                terms=[
                    LinearTerm(variable="x", coefficient=1),
                    LinearTerm(variable="y", coefficient=2),
                ],
                relation="<=",
                rhs=4,
            ),
            LinearConstraint(
                name="资源二",
                terms=[
                    LinearTerm(variable="x", coefficient=4),
                    LinearTerm(variable="y", coefficient=2),
                ],
                relation="<=",
                rhs=12,
            ),
            LinearConstraint(
                name="产量非负",
                terms=[LinearTerm(variable="x", coefficient=1)],
                relation=">=",
                rhs=0,
            ),
        ],
    )

    result = solve_linear_program(problem)

    assert result.status == "OPTIMAL"
    assert result.solver_name == "SCIP"
    assert result.variable_values == pytest.approx({"x": 2, "y": 1})
    assert result.objective_value == pytest.approx(11)


def test_milp_reports_unavailable_scip_without_fake_solution(monkeypatch) -> None:
    from ortools.linear_solver import pywraplp

    from math_modeling_agent.linear_solver import solve_linear_program
    from math_modeling_agent.models import (
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )

    problem = LinearProgramProblem(
        variables=[LinearVariable(name="x", unit="件", domain="integer")],
        objective=LinearObjective(
            direction="maximize",
            terms=[LinearTerm(variable="x", coefficient=1)],
        ),
        constraints=[],
    )
    monkeypatch.setattr(pywraplp.Solver, "CreateSolver", lambda _name: None)

    result = solve_linear_program(problem)

    assert result.status == "SOLVER_UNAVAILABLE"
    assert result.solver_name == "SCIP"
    assert result.variable_values == {}
    assert result.objective_value is None
