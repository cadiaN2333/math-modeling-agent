from math_modeling_agent.models import (
    CoverageRequirement,
    Employee,
    SchedulingProblem,
    Shift,
)
from math_modeling_agent.solver import Assignment


def make_problem() -> SchedulingProblem:
    return SchedulingProblem(
        employees=[
            Employee(
                employee_id="E1",
                name="有急救技能的员工",
                skills={"急救"},
                max_hours=8,
            ),
            Employee(
                employee_id="E2",
                name="普通员工",
                skills=set(),
                max_hours=8,
            ),
        ],
        shifts=[
            Shift(shift_id="S1", duration_hours=8),
            Shift(shift_id="S2", duration_hours=8),
        ],
        coverage_requirements=[
            CoverageRequirement(
                shift_id="S1",
                minimum_employees=1,
                required_skill_counts={"急救": 1},
            ),
            CoverageRequirement(
                shift_id="S2",
                minimum_employees=1,
                required_skill_counts={},
            ),
        ],
    )


def test_validator_accepts_a_valid_schedule() -> None:
    from math_modeling_agent.validator import validate_solution

    problem = make_problem()
    assignments = [
        Assignment(employee_id="E1", shift_id="S1"),
        Assignment(employee_id="E2", shift_id="S2"),
    ]

    report = validate_solution(problem, assignments)

    assert report.is_valid
    assert report.errors == []


def test_validator_rejects_uncovered_shift() -> None:
    from math_modeling_agent.validator import validate_solution

    problem = make_problem()
    assignments = [
        Assignment(employee_id="E1", shift_id="S1"),
    ]

    report = validate_solution(problem, assignments)

    assert not report.is_valid
    assert report.errors


def test_validator_rejects_employee_without_required_skill() -> None:
    from math_modeling_agent.validator import validate_solution

    problem = make_problem()
    assignments = [
        Assignment(employee_id="E2", shift_id="S1"),
        Assignment(employee_id="E1", shift_id="S2"),
    ]

    report = validate_solution(problem, assignments)

    assert not report.is_valid
    assert report.errors


def test_validator_rejects_excessive_work_hours() -> None:
    from math_modeling_agent.validator import validate_solution

    problem = make_problem()
    assignments = [
        Assignment(employee_id="E1", shift_id="S1"),
        Assignment(employee_id="E1", shift_id="S2"),
    ]

    report = validate_solution(problem, assignments)

    assert not report.is_valid
    assert report.errors