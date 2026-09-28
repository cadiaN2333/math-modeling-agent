"""独立检查排班求解结果是否满足问题约束。"""

from dataclasses import dataclass

from .models import SchedulingProblem
from .solver import Assignment


@dataclass
class ValidationReport:
    """保存验证是否通过，以及发现的问题。"""

    is_valid: bool
    errors: list[str]


def _hours_to_minutes(hours: float) -> int:
    """将小时换算成分钟，和求解器保持相同精度。"""
    return int(round(hours * 60))


def validate_solution(
    problem: SchedulingProblem,
    assignments: list[Assignment],
) -> ValidationReport:
    """重新检查排班人数、技能要求和员工工时。"""

    errors: list[str] = []
    employee_by_id = {
        employee.employee_id: employee
        for employee in problem.employees
    }
    shift_by_id = {
        shift.shift_id: shift
        for shift in problem.shifts
    }

    # 按班次和员工分别汇总排班记录
    employees_by_shift: dict[str, list[str]] = {
        shift.shift_id: []
        for shift in problem.shifts
    }
    minutes_by_employee: dict[str, int] = {
        employee.employee_id: 0
        for employee in problem.employees
    }
    seen_assignments: set[tuple[str, str]] = set()

    for assignment in assignments:
        employee_id = assignment.employee_id
        shift_id = assignment.shift_id

        # 确认结果中引用的员工和班次都存在
        if employee_id not in employee_by_id:
            errors.append(f"排班结果引用了不存在的员工：{employee_id}")
            continue

        if shift_id not in shift_by_id:
            errors.append(f"排班结果引用了不存在的班次：{shift_id}")
            continue

        # 同一员工不能在同一班次重复出现
        key = (employee_id, shift_id)
        if key in seen_assignments:
            errors.append(
                f"员工 {employee_id} 在班次 {shift_id} 中被重复安排"
            )
            continue
        seen_assignments.add(key)

        employees_by_shift[shift_id].append(employee_id)
        minutes_by_employee[employee_id] += _hours_to_minutes(
            shift_by_id[shift_id].duration_hours
        )

    # 检查每个班次的最低人数和技能人数
    for requirement in problem.coverage_requirements:
        assigned_employee_ids = employees_by_shift[requirement.shift_id]

        if len(assigned_employee_ids) < requirement.minimum_employees:
            errors.append(
                f"班次 {requirement.shift_id} 人数不足："
                f"需要 {requirement.minimum_employees} 人，"
                f"实际 {len(assigned_employee_ids)} 人"
            )

        for skill, required_count in requirement.required_skill_counts.items():
            skilled_count = sum(
                skill in employee_by_id[employee_id].skills
                for employee_id in assigned_employee_ids
            )

            if skilled_count < required_count:
                errors.append(
                    f"班次 {requirement.shift_id} 的技能 {skill} 人数不足："
                    f"需要 {required_count} 人，实际 {skilled_count} 人"
                )

    # 检查每名员工的总工时是否超过上限
    for employee in problem.employees:
        actual_minutes = minutes_by_employee[employee.employee_id]
        maximum_minutes = _hours_to_minutes(employee.max_hours)

        if actual_minutes > maximum_minutes:
            errors.append(
                f"员工 {employee.employee_id} 工时超限："
                f"上限 {maximum_minutes} 分钟，"
                f"实际 {actual_minutes} 分钟"
            )

    return ValidationReport(
        is_valid=not errors,
        errors=errors,
    )