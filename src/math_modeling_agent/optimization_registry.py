"""依据统一 IR 的能力与数值特征选择并调用优化后端。"""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from pydantic import ValidationError

from .min_cost_flow_solver import solve_min_cost_flow
from .models import MinCostFlowProblem
from .optimization_ir import (
    LinearFormulationIR,
    LinearTermIR,
    LinearVariableIR,
    NetworkFlowFormulationIR,
    OptimizationIR,
)


_FEASIBLE_STATUSES = {"OPTIMAL", "FEASIBLE"}
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _is_finite_number(value: object) -> bool:
    """检查统一结果中的数值，避免布尔值被误当作整数。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isinstance(value, int) or isfinite(value)


@dataclass(frozen=True)
class OptimizationResult:
    """保存后端状态、目标值和按 IR 编号组织的变量或弧流解。"""

    status: str
    objective_value: float | int | None
    variable_values: dict[str, float | int]
    backend_id: str
    message: str | None = None

    def __post_init__(self) -> None:
        """防止不可行或失败状态携带伪造的数值解。"""

        if self.status in _FEASIBLE_STATUSES:
            if not _is_finite_number(self.objective_value):
                raise ValueError("可行求解结果必须包含有限目标值")
            if not self.variable_values:
                raise ValueError("可行求解结果必须包含变量或弧流值")
            if any(
                not _is_finite_number(value)
                for value in self.variable_values.values()
            ):
                raise ValueError("可行求解结果不能包含非有限变量值")
        elif self.objective_value is not None or self.variable_values:
            raise ValueError("非可行求解结果不得携带目标值或变量解")


class SolverBackend(Protocol):
    """统一 IR 求解后端必须提供的能力接口。"""

    backend_id: str

    def supports(self, problem: OptimizationIR) -> bool: ...

    def solve(self, problem: OptimizationIR) -> OptimizationResult: ...


def _new_linear_solver(solver_name: str) -> pywraplp.Solver | None:
    try:
        return pywraplp.Solver.CreateSolver(solver_name)
    except RuntimeError:
        return None


def _make_mpsolver_model(
    formulation: LinearFormulationIR,
    solver_name: str,
    *,
    include_objective: bool = True,
) -> tuple[
    pywraplp.Solver | None,
    dict[str, pywraplp.Variable],
]:
    """把仿射 IR 编译为 OR-Tools MPSolver 模型。"""

    solver = _new_linear_solver(solver_name)
    if solver is None:
        return None, {}

    infinity = solver.infinity()
    variables: dict[str, pywraplp.Variable] = {}
    for item in formulation.variables:
        lower = item.lower_bound
        upper = item.upper_bound
        if item.domain == "binary":
            lower = max(0, lower) if lower is not None else 0
            upper = min(1, upper) if upper is not None else 1
        lower_bound = lower if lower is not None else -infinity
        upper_bound = upper if upper is not None else infinity

        if item.domain == "continuous":
            variable = solver.NumVar(lower_bound, upper_bound, item.variable_id)
        else:
            variable = solver.IntVar(lower_bound, upper_bound, item.variable_id)
        variables[item.variable_id] = variable

    for item in formulation.constraints:
        expression = sum(
            term.coefficient * variables[term.variable_id]
            for term in item.terms
        )
        adjusted_rhs = item.rhs - item.constant
        if item.relation == "<=":
            solver.Add(expression <= adjusted_rhs, item.constraint_id)
        elif item.relation == ">=":
            solver.Add(expression >= adjusted_rhs, item.constraint_id)
        else:
            solver.Add(expression == adjusted_rhs, item.constraint_id)

    objective = solver.Objective()
    if include_objective:
        for term in formulation.objective.terms:
            objective.SetCoefficient(variables[term.variable_id], term.coefficient)
        objective.SetOffset(formulation.objective.constant)
        if formulation.objective.direction == "maximize":
            objective.SetMaximization()
        else:
            objective.SetMinimization()
    else:
        objective.SetMinimization()

    return solver, variables


def _mpsolver_status(status_code: int) -> str:
    """将 MPSolver 状态码转换为稳定字符串。"""

    return {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }.get(status_code, f"UNKNOWN_{status_code}")


class _MPSolverBackend:
    """为 GLOP 和 SCIP 共享连续/混合整数线性模型的转换。"""

    def __init__(self, backend_id: str, solver_name: str) -> None:
        self.backend_id = backend_id
        self._solver_name = solver_name

    def supports(self, problem: OptimizationIR) -> bool:
        return (
            isinstance(problem.formulation, LinearFormulationIR)
            and _new_linear_solver(self._solver_name) is not None
        )

    def solve(self, problem: OptimizationIR) -> OptimizationResult:
        formulation = problem.formulation
        if not isinstance(formulation, LinearFormulationIR):
            return _unsupported_backend_result(self.backend_id)

        solver, variables = _make_mpsolver_model(
            formulation,
            self._solver_name,
        )
        if solver is None:
            return OptimizationResult(
                status="SOLVER_UNAVAILABLE",
                objective_value=None,
                variable_values={},
                backend_id=self.backend_id,
                message=f"OR-Tools {self._solver_name} 后端不可用",
            )

        status = _mpsolver_status(solver.Solve())
        if status == "INFEASIBLE" and self._solver_name == "GLOP":
            # GLOP 可能将不可行与无界统一报告为 INFEASIBLE；零目标可行性复核用于区分。
            feasibility_solver, _ = _make_mpsolver_model(
                formulation,
                self._solver_name,
                include_objective=False,
            )
            if feasibility_solver is None:
                status = "INFEASIBLE_OR_UNBOUNDED"
            else:
                feasibility_status = _mpsolver_status(feasibility_solver.Solve())
                if feasibility_status in _FEASIBLE_STATUSES:
                    status = "UNBOUNDED"
                elif feasibility_status != "INFEASIBLE":
                    status = "INFEASIBLE_OR_UNBOUNDED"

        if status not in _FEASIBLE_STATUSES:
            return OptimizationResult(
                status=status,
                objective_value=None,
                variable_values={},
                backend_id=self.backend_id,
            )

        return OptimizationResult(
            status=status,
            objective_value=solver.Objective().Value(),
            variable_values={
                variable_id: variable.solution_value()
                for variable_id, variable in variables.items()
            },
            backend_id=self.backend_id,
        )


class _GlopBackend(_MPSolverBackend):
    """处理连续线性模型的 GLOP 后端。"""

    def __init__(self) -> None:
        super().__init__("ortools_glop", "GLOP")

    def supports(self, problem: OptimizationIR) -> bool:
        formulation = problem.formulation
        return (
            isinstance(formulation, LinearFormulationIR)
            and problem.problem_family != "employee_scheduling"
            and all(variable.domain == "continuous" for variable in formulation.variables)
            and super().supports(problem)
        )


class _ScipBackend(_MPSolverBackend):
    """处理混合整数及 GLOP 不支持的线性模型的 SCIP 后端。"""

    def __init__(self) -> None:
        super().__init__("ortools_scip", "SCIP")


def _integer_value(value: float) -> int | None:
    """仅接受 CP-SAT 可精确表达的有限 64 位整数。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int):
        result = value
    else:
        if not isfinite(value) or not value.is_integer():
            return None
        result = int(value)
    if not _INT64_MIN <= result <= _INT64_MAX:
        return None
    return result


def _cp_sat_variable_bounds(variable: LinearVariableIR) -> tuple[int, int] | None:
    """返回 CP-SAT 整数变量的有限闭区间。"""

    if variable.domain == "continuous":
        return None

    if variable.domain == "binary":
        raw_lower = variable.lower_bound if variable.lower_bound is not None else 0
        raw_upper = variable.upper_bound if variable.upper_bound is not None else 1
        lower = _integer_value(raw_lower)
        upper = _integer_value(raw_upper)
        if lower is None or upper is None or not 0 <= lower <= upper <= 1:
            return None
        return lower, upper

    if variable.lower_bound is None or variable.upper_bound is None:
        return None
    lower = _integer_value(variable.lower_bound)
    upper = _integer_value(variable.upper_bound)
    if lower is None or upper is None or lower > upper:
        return None
    return lower, upper


def _cp_sat_expression_range_is_safe(
    terms: list[LinearTermIR],
    constant: float,
    variable_bounds: dict[str, tuple[int, int]],
) -> bool:
    """保证线性表达式在变量界内的整个取值范围都不超出 int64。"""

    integer_constant = _integer_value(constant)
    if integer_constant is None:
        return False

    minimum = integer_constant
    maximum = integer_constant
    for term in terms:
        coefficient = _integer_value(term.coefficient)
        if coefficient is None:
            return False
        lower, upper = variable_bounds[term.variable_id]
        lower_product = coefficient * lower
        upper_product = coefficient * upper
        minimum += min(lower_product, upper_product)
        maximum += max(lower_product, upper_product)
        if minimum < _INT64_MIN or maximum > _INT64_MAX:
            return False
    return True


def _cp_sat_compatible(formulation: LinearFormulationIR) -> bool:
    """检查变量域、变量界与表达式常数是否完全满足 CP-SAT 整数要求。"""

    variable_bounds: dict[str, tuple[int, int]] = {}
    for variable in formulation.variables:
        bounds = _cp_sat_variable_bounds(variable)
        if bounds is None:
            return False
        variable_bounds[variable.variable_id] = bounds

    if not _cp_sat_expression_range_is_safe(
        formulation.objective.terms,
        formulation.objective.constant,
        variable_bounds,
    ):
        return False

    for constraint in formulation.constraints:
        rhs = _integer_value(constraint.rhs)
        constant = _integer_value(constraint.constant)
        if rhs is None or constant is None:
            return False
        if not _INT64_MIN <= rhs - constant <= _INT64_MAX:
            return False
        if not _cp_sat_expression_range_is_safe(
            constraint.terms,
            constraint.constant,
            variable_bounds,
        ):
            return False
    return True


class _CpSatBackend:
    """处理有限整数域、整数系数排班/线性模型的 CP-SAT 后端。"""

    backend_id = "ortools_cp_sat"

    def supports(self, problem: OptimizationIR) -> bool:
        formulation = problem.formulation
        return (
            isinstance(formulation, LinearFormulationIR)
            and problem.problem_family == "employee_scheduling"
            and _cp_sat_compatible(formulation)
        )

    def solve(self, problem: OptimizationIR) -> OptimizationResult:
        formulation = problem.formulation
        if (
            not isinstance(formulation, LinearFormulationIR)
            or not _cp_sat_compatible(formulation)
        ):
            return _unsupported_backend_result(self.backend_id)

        model = cp_model.CpModel()
        variables: dict[str, cp_model.IntVar] = {}
        for item in formulation.variables:
            bounds = _cp_sat_variable_bounds(item)
            if bounds is None:
                return _unsupported_backend_result(self.backend_id)
            variables[item.variable_id] = model.NewIntVar(
                bounds[0],
                bounds[1],
                item.variable_id,
            )

        for constraint in formulation.constraints:
            expression = sum(
                int(term.coefficient) * variables[term.variable_id]
                for term in constraint.terms
            )
            adjusted_rhs = int(constraint.rhs - constraint.constant)
            if constraint.relation == "<=":
                model.Add(expression <= adjusted_rhs)
            elif constraint.relation == ">=":
                model.Add(expression >= adjusted_rhs)
            else:
                model.Add(expression == adjusted_rhs)

        objective = sum(
            int(term.coefficient) * variables[term.variable_id]
            for term in formulation.objective.terms
        ) + int(formulation.objective.constant)
        if formulation.objective.direction == "maximize":
            model.Maximize(objective)
        else:
            model.Minimize(objective)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        status_code = solver.Solve(model)
        status = solver.StatusName(status_code)
        if status not in _FEASIBLE_STATUSES:
            return OptimizationResult(
                status=status,
                objective_value=None,
                variable_values={},
                backend_id=self.backend_id,
            )

        return OptimizationResult(
            status=status,
            objective_value=solver.ObjectiveValue(),
            variable_values={
                variable_id: float(solver.Value(variable))
                for variable_id, variable in variables.items()
            },
            backend_id=self.backend_id,
        )


class _SimpleMinCostFlowBackend:
    """直接把结构化网络 IR 交给专用最小费用流后端。"""

    backend_id = "ortools_simple_min_cost_flow"

    def supports(self, problem: OptimizationIR) -> bool:
        return isinstance(problem.formulation, NetworkFlowFormulationIR)

    def solve(self, problem: OptimizationIR) -> OptimizationResult:
        formulation = problem.formulation
        if not isinstance(formulation, NetworkFlowFormulationIR):
            return _unsupported_backend_result(self.backend_id)

        try:
            domain_problem = MinCostFlowProblem.model_validate(
                {
                    "nodes": [node.model_dump() for node in formulation.nodes],
                    "arcs": [arc.model_dump() for arc in formulation.arcs],
                    "flow_unit": formulation.flow_unit,
                    "cost_unit": formulation.cost_unit,
                }
            )
        except ValidationError as error:
            return OptimizationResult(
                status="MODEL_INVALID",
                objective_value=None,
                variable_values={},
                backend_id=self.backend_id,
                message=str(error),
            )

        result = solve_min_cost_flow(domain_problem)
        if result.status != "OPTIMAL" or result.total_cost is None:
            return OptimizationResult(
                status=result.status,
                objective_value=None,
                variable_values={},
                backend_id=self.backend_id,
            )
        return OptimizationResult(
            status=result.status,
            objective_value=result.total_cost,
            variable_values={
                arc_id: flow
                for arc_id, flow in result.arc_flows.items()
            },
            backend_id=self.backend_id,
        )


def _unsupported_backend_result(backend_id: str) -> OptimizationResult:
    return OptimizationResult(
        status="UNSUPPORTED_MODEL",
        objective_value=None,
        variable_values={},
        backend_id=backend_id,
        message="该后端不支持当前 IR 或其数值特征",
    )


class SolverRegistry:
    """按优先级选择精确兼容的后端，不自动删除或松弛任何约束。"""

    def __init__(self, backends: list[SolverBackend] | None = None) -> None:
        self.backends = list(
            backends
            if backends is not None
            else [
                _SimpleMinCostFlowBackend(),
                _GlopBackend(),
                _CpSatBackend(),
                _ScipBackend(),
            ]
        )

    def select(self, problem: OptimizationIR) -> SolverBackend | None:
        """按声明顺序返回第一个完全支持当前模型的后端。"""

        return next(
            (backend for backend in self.backends if backend.supports(problem)),
            None,
        )

    def solve(self, problem: OptimizationIR) -> OptimizationResult:
        """求解 IR；无兼容后端时返回无解值的 UNSUPPORTED_MODEL。"""

        backend = self.select(problem)
        if backend is None:
            return OptimizationResult(
                status="UNSUPPORTED_MODEL",
                objective_value=None,
                variable_values={},
                backend_id="none",
                message="没有已注册的后端支持当前优化模型",
            )
        return backend.solve(problem)
