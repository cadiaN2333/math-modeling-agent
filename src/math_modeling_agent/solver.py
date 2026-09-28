"""使用 OR-Tools 求解排班问题。"""

from dataclasses import dataclass

from ortools.sat.python import cp_model

from .models import SchedulingProblem


@dataclass
class Assignment:
    """表示一名员工被安排到一个班次。"""

    employee_id: str
    shift_id: str


@dataclass
class SolverResult:
    """保存求解状态、排班结果和目标值。"""

    status: str
    assignments: list[Assignment]
    objective_minutes: int | None


def _hours_to_minutes(hours: float) -> int:
    """将小时换算成分钟，求解时精确到分钟。"""
    return int(round(hours * 60))


def solve_schedule(problem: SchedulingProblem) -> SolverResult:
    """根据排班问题创建模型、求解并返回结果。"""

    # 创建一个空的 CP-SAT 优化模型
    model = cp_model.CpModel()

    # 保存每个“员工—班次”组合对应的 0/1 决策变量
    assigned = {}

    for employee in problem.employees:
        for shift in problem.shifts:
            key = (employee.employee_id, shift.shift_id)
            variable_name = f"assign_{employee.employee_id}_{shift.shift_id}"
            assigned[key] = model.NewBoolVar(variable_name)

    # 添加每个班次的最低人数约束
    for requirement in problem.coverage_requirements:
        assigned_count = sum(
            assigned[(employee.employee_id, requirement.shift_id)]
            for employee in problem.employees
        )
        model.Add(assigned_count >= requirement.minimum_employees)

        # 添加每个班次的技能人数约束
        for skill, required_count in requirement.required_skill_counts.items():
            skilled_employee_count = sum(
                assigned[(employee.employee_id, requirement.shift_id)]
                for employee in problem.employees
                if skill in employee.skills
            )
            model.Add(skilled_employee_count >= required_count)

    # 添加每名员工的最大工时约束
    for employee in problem.employees:
        total_minutes = sum(
            assigned[(employee.employee_id, shift.shift_id)]
            * _hours_to_minutes(shift.duration_hours)
            for shift in problem.shifts
        )
        model.Add(total_minutes <= _hours_to_minutes(employee.max_hours))

    # 以满足需求的前提下尽量减少总排班分钟数
    total_scheduled_minutes = sum(
        assigned[(employee.employee_id, shift.shift_id)]
        * _hours_to_minutes(shift.duration_hours)
        for employee in problem.employees
        for shift in problem.shifts
    )
    model.Minimize(total_scheduled_minutes)

    # 创建求解器；单线程和固定随机种子有助于复现实验
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0

    # 开始求解
    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)

    # 没有可行解时，不返回虚构的分配
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SolverResult(
            status=status,
            assignments=[],
            objective_minutes=None,
        )

    # 只把值为 1 的变量转换成实际排班记录
    assignments = [
        Assignment(employee_id=employee.employee_id, shift_id=shift.shift_id)
        for employee in problem.employees
        for shift in problem.shifts
        if solver.Value(assigned[(employee.employee_id, shift.shift_id)]) == 1
    ]

    return SolverResult(
        status=status,
        assignments=assignments,
        objective_minutes=int(round(solver.ObjectiveValue())),
    )