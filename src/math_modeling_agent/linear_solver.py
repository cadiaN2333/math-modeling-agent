"""使用 OR-Tools GLOP 或 SCIP 求解连续/混合整数线性规划。"""

from dataclasses import dataclass

from ortools.linear_solver import pywraplp

from .models import LinearProgramProblem


@dataclass
class LinearSolverResult:
    """保存 LP/MILP 求解状态、后端、目标值和变量解。"""

    status: str
    objective_value: float | None
    variable_values: dict[str, float]
    solver_name: str


def _select_solver_name(problem: LinearProgramProblem) -> str:
    """根据变量域为连续 LP 或含离散变量模型选择后端。"""

    if any(variable.domain != "continuous" for variable in problem.variables):
        return "SCIP"
    return "GLOP"


def _build_linear_model(
    problem: LinearProgramProblem,
    *,
    include_objective: bool,
) -> tuple[pywraplp.Solver | None, dict[str, pywraplp.Variable], str]:
    """建立连续或混合整数模型，也可建立只含约束的可行性模型。"""

    solver_name = _select_solver_name(problem)
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        return None, {}, solver_name

    infinity = solver.infinity()
    variables: dict[str, pywraplp.Variable] = {}
    for item in problem.variables:
        if item.domain == "binary":
            variables[item.name] = solver.BoolVar(item.name)
        elif item.domain == "integer":
            variables[item.name] = solver.IntVar(-infinity, infinity, item.name)
        else:
            variables[item.name] = solver.NumVar(-infinity, infinity, item.name)

    for constraint in problem.constraints:
        expression = sum(
            term.coefficient * variables[term.variable]
            for term in constraint.terms
        )
        if constraint.relation == "<=":
            solver.Add(expression <= constraint.rhs, constraint.name)
        elif constraint.relation == ">=":
            solver.Add(expression >= constraint.rhs, constraint.name)
        else:
            solver.Add(expression == constraint.rhs, constraint.name)

    objective = solver.Objective()
    if include_objective:
        for term in problem.objective.terms:
            objective.SetCoefficient(variables[term.variable], term.coefficient)
        if problem.objective.direction == "maximize":
            objective.SetMaximization()
        else:
            objective.SetMinimization()
    else:
        # 零目标模型只检查约束集合是否存在可行点。
        objective.SetMinimization()

    return solver, variables, solver_name


def _status_name(status_code: int) -> str:
    """将 MPSolver 返回码转换为稳定的状态字符串。"""

    return {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }.get(status_code, f"UNKNOWN_{status_code}")


def solve_linear_program(problem: LinearProgramProblem) -> LinearSolverResult:
    """按变量域建立 GLOP/SCIP 模型；非可行状态不返回伪造解。"""

    solver, variables, solver_name = _build_linear_model(
        problem,
        include_objective=True,
    )
    if solver is None:
        return LinearSolverResult(
            status="SOLVER_UNAVAILABLE",
            objective_value=None,
            variable_values={},
            solver_name=solver_name,
        )

    status_code = solver.Solve()
    status = _status_name(status_code)

    if status == "INFEASIBLE" and solver_name == "GLOP":
        # GLOP 的 MPSolver 包装会把“不可行或无界”归为 INFEASIBLE；
        # 用同一组约束和零目标再求一次，区分确实无解与目标无界。
        feasibility_solver, _, _ = _build_linear_model(
            problem,
            include_objective=False,
        )
        if feasibility_solver is None:
            status = "INFEASIBLE_OR_UNBOUNDED"
        else:
            feasibility_status = _status_name(feasibility_solver.Solve())
            if feasibility_status in {"OPTIMAL", "FEASIBLE"}:
                status = "UNBOUNDED"
            elif feasibility_status != "INFEASIBLE":
                status = "INFEASIBLE_OR_UNBOUNDED"

    if status not in {"OPTIMAL", "FEASIBLE"}:
        return LinearSolverResult(
            status=status,
            objective_value=None,
            variable_values={},
            solver_name=solver_name,
        )

    return LinearSolverResult(
        status=status,
        objective_value=solver.Objective().Value(),
        variable_values={
            name: variable.solution_value()
            for name, variable in variables.items()
        },
        solver_name=solver_name,
    )
