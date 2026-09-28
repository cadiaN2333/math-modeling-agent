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
