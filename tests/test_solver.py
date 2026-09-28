from math_modeling_agent.models import (
    CoverageRequirement,
    Employee,
    SchedulingProblem,
    Shift,
)


def test_solver_covers_shifts_and_required_skills() -> None:
    # 延迟导入，让测试在求解器尚未实现时也能正常收集
    from math_modeling_agent.solver import solve_schedule

    problem = SchedulingProblem(
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

    result = solve_schedule(problem)

    # 求解器应该找到最优解或可行解
    assert result.status in {"OPTIMAL", "FEASIBLE"}

    assignments = {
        (item.employee_id, item.shift_id)
        for item in result.assignments
    }

    # 有急救技能的员工必须排到 S1，两个班次都必须有人
    assert ("E1", "S1") in assignments
    assert ("E2", "S2") in assignments


def test_solver_reports_infeasible_when_skill_is_unavailable() -> None:
    from math_modeling_agent.solver import solve_schedule

    problem = SchedulingProblem(
        employees=[
            Employee(
                employee_id="E1",
                name="普通员工",
                skills=set(),
                max_hours=8,
            )
        ],
        shifts=[Shift(shift_id="S1", duration_hours=8)],
        coverage_requirements=[
            CoverageRequirement(
                shift_id="S1",
                minimum_employees=1,
                required_skill_counts={"急救": 1},
            )
        ],
    )

    result = solve_schedule(problem)

    # 没有员工具备急救技能，因此问题不可行
    assert result.status == "INFEASIBLE"

    # 无解时不应返回任何排班
    assert result.assignments == []
