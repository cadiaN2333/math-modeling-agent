"""提供用于本地演示的排班样例。"""

from .models import CoverageRequirement, Employee, SchedulingProblem, Shift


def make_sample_problem() -> SchedulingProblem:
    """创建包含人数和技能要求的三班次样例问题。"""

    return SchedulingProblem(
        employees=[
            Employee(
                employee_id="E1",
                name="林晓",
                skills={"急救"},
                max_hours=8,
            ),
            Employee(
                employee_id="E2",
                name="陈立",
                skills={"收银"},
                max_hours=16,
            ),
            Employee(
                employee_id="E3",
                name="周宁",
                skills=set(),
                max_hours=16,
            ),
        ],
        shifts=[
            Shift(shift_id="S1", duration_hours=8),
            Shift(shift_id="S2", duration_hours=8),
            Shift(shift_id="S3", duration_hours=8),
        ],
        coverage_requirements=[
            CoverageRequirement(
                shift_id="S1",
                minimum_employees=2,
                required_skill_counts={"急救": 1},
            ),
            CoverageRequirement(
                shift_id="S2",
                minimum_employees=1,
                required_skill_counts={"收银": 1},
            ),
            CoverageRequirement(
                shift_id="S3",
                minimum_employees=2,
                required_skill_counts={},
            ),
        ],
    )
