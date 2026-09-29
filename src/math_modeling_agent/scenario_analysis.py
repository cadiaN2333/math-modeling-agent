"""重算命名线性情景并安全比较求解结果。"""

from dataclasses import dataclass
from math import isclose

from .linear_agent import LinearModelingRun, run_linear_modeling
from .models import LinearProgramProblem
from .scenario_models import ScenarioAnalysisRequest


@dataclass
class ScenarioRunResult:
    """保存一个情景的求解报告以及相对基准的差异。"""

    scenario_id: str
    description: str
    modeling_run: LinearModelingRun
    objective_delta_from_base: float | None
    relative_objective_delta_percent: float | None
    variable_deltas_from_base: dict[str, float]
    objective_comparison_note: str | None


@dataclass
class ScenarioAnalysisResult:
    """保存基准求解和所有情景比较结果。"""

    base_run: LinearModelingRun
    scenario_runs: list[ScenarioRunResult]


def _has_valid_solution(run: LinearModelingRun) -> bool:
    """仅接受有可行状态且通过独立 validator 的结果用于比较。"""

    return (
        run.solver_result.status in {"OPTIMAL", "FEASIBLE"}
        and run.validation_report is not None
        and run.validation_report.is_valid
    )


def _objective_is_unchanged(
    base_problem: LinearProgramProblem,
    scenario_problem: LinearProgramProblem,
) -> bool:
    """判断两个模型是否使用相同方向和系数的目标函数。"""

    if base_problem.objective.direction != scenario_problem.objective.direction:
        return False
    base_terms = {
        term.variable: term.coefficient for term in base_problem.objective.terms
    }
    scenario_terms = {
        term.variable: term.coefficient
        for term in scenario_problem.objective.terms
    }
    if base_terms.keys() != scenario_terms.keys():
        return False
    return all(
        isclose(base_terms[name], scenario_terms[name], rel_tol=1e-12, abs_tol=1e-12)
        for name in base_terms
    )


def run_scenario_analysis(
    request: ScenarioAnalysisRequest,
) -> ScenarioAnalysisResult:
    """独立求解基准与情景，并只在结果可比时计算差值。"""

    base_run = run_linear_modeling(request.base_problem)
    base_is_valid = _has_valid_solution(base_run)
    results: list[ScenarioRunResult] = []

    for scenario in request.scenarios:
        run = run_linear_modeling(scenario.problem)
        scenario_is_valid = _has_valid_solution(run)
        objective_is_unchanged = _objective_is_unchanged(
            request.base_problem,
            scenario.problem,
        )
        objective_delta: float | None = None
        relative_delta: float | None = None
        variable_deltas: dict[str, float] = {}
        note: str | None = None

        if not base_is_valid:
            note = "基准模型没有通过 validator 的可行解，无法计算差值。"
        elif not scenario_is_valid:
            note = "该情景没有通过 validator 的可行解，无法计算差值。"
        else:
            base_values = base_run.solver_result.variable_values
            scenario_values = run.solver_result.variable_values
            variable_deltas = {
                name: scenario_values[name] - base_values[name]
                for name in base_values
            }
            if not objective_is_unchanged:
                note = "情景目标函数与基准不同，目标值不可直接比较；仍报告变量变化。"
            else:
                base_objective = base_run.solver_result.objective_value
                scenario_objective = run.solver_result.objective_value
                if base_objective is None or scenario_objective is None:
                    note = "求解结果缺少目标值，无法计算目标差。"
                else:
                    objective_delta = scenario_objective - base_objective
                    if abs(base_objective) > 1e-12:
                        relative_delta = objective_delta / abs(base_objective) * 100
                    if (
                        base_run.solver_result.status != "OPTIMAL"
                        or run.solver_result.status != "OPTIMAL"
                    ):
                        note = "目标差基于当前可行解；求解器未对两边都证明最优。"
                    elif relative_delta is None:
                        note = "基准目标值为0，无法计算相对百分比变化。"

        results.append(
            ScenarioRunResult(
                scenario_id=scenario.scenario_id,
                description=scenario.description,
                modeling_run=run,
                objective_delta_from_base=objective_delta,
                relative_objective_delta_percent=relative_delta,
                variable_deltas_from_base=variable_deltas,
                objective_comparison_note=note,
            )
        )

    return ScenarioAnalysisResult(base_run=base_run, scenario_runs=results)
