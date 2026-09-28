from math_modeling_agent.models import (
    CoverageRequirement,
    Employee,
    SchedulingProblem,
    Shift,
)


def test_run_modeling_solves_and_validates() -> None:
    # 延迟导入，先让测试因流程模块尚未实现而失败
    from math_modeling_agent.agent import run_modeling

    problem = SchedulingProblem(
        employees=[
            Employee(
                employee_id="E1",
                name="急救员",
                skills={"急救"},
                max_hours=8,
            )
        ],
        shifts=[
            Shift(shift_id="S1", duration_hours=8),
        ],
        coverage_requirements=[
            CoverageRequirement(
                shift_id="S1",
                minimum_employees=1,
                required_skill_counts={"急救": 1},
            )
        ],
    )

    result = run_modeling(problem)

    # 求解成功后，流程还应把方案交给独立验证器
    assert result.solver_result.status in {"OPTIMAL", "FEASIBLE"}
    assert result.validation_report is not None
    assert result.validation_report.is_valid
    # 建模流程应先按问题和目标检索方法，并优先给出当前已实现的 CP-SAT
    assert result.method_recommendations
    assert result.method_recommendations[0].method_id == "cp_sat_scheduling"
    assert result.method_recommendations[0].implementation_status == "已实现"
