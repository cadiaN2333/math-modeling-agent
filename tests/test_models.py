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
