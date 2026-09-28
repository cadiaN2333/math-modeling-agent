"""使用 SCIP 优化电工杯问题三的连续装置功率。"""

from itertools import product
from statistics import fmean

from ortools.linear_solver import pywraplp
from pydantic import BaseModel, Field

from .energy_park import TypicalDayResult, _purchase_price_yuan_per_kwh, compute_energy_schedule
from .energy_park_discrete import (
    AnnualProductionSummary,
    DiscreteQuestionTwoResult,
    classify_green_metrics,
    run_discrete_question_two,
)
from .energy_park_continuous_validator import validate_continuous_schedule
from .energy_park_models import EnergyParkDataset, HourlyProfile
from .energy_park_validator import EnergyParkValidationReport


RATED_CAPACITY_SCALE = 2.0
MINIMUM_LOAD_FRACTION = 0.1


class ContinuousScheduleRun(BaseModel):
    """保存单一日产量和风光场景的连续调度结果。"""

    scenario_id: str
    target_tons_per_day: float
    capacity_scale: float
    minimum_load_fraction: float
    solver_status: str
    process_fractions: list[float] = Field(min_length=0, max_length=24)
    solver_grid_purchase_mw: list[float] = Field(min_length=0, max_length=24)
    solver_grid_export_mw: list[float] = Field(min_length=0, max_length=24)
    operation: TypicalDayResult | None
    validation_report: EnergyParkValidationReport | None


class ContinuousQuestionThreeResult(BaseModel):
    """保存问题三典型日、多场景和与问题二的比较。"""

    target_levels_tons_per_day: list[float]
    rated_capacity_scale: float
    minimum_load_fraction: float
    days_per_scenario: int
    representative_year_days: int
    typical_runs: list[ContinuousScheduleRun]
    scenario_runs: list[ContinuousScheduleRun]
    annual_summaries: list[AnnualProductionSummary]
    discrete_comparison_by_target: list[dict[str, float]]
    best_typical_target_tons_per_day: float
    best_annual_cost_per_ton_target_tons_per_day: float
    modeling_assumptions: list[str]


def _weighted_export_price(
    data: EnergyParkDataset,
    wind_generation_mw: float,
    pv_generation_mw: float,
) -> float:
    """按当小时风光出力结构折算单一上网变量的边际电价。"""

    total_generation = wind_generation_mw + pv_generation_mw
    if total_generation <= 0:
        return 0.0
    return (
        wind_generation_mw * data.costs.wind_export_price_yuan_per_kwh
        + pv_generation_mw * data.costs.pv_export_price_yuan_per_kwh
    ) / total_generation


def _normalize_solver_fraction(value: float, minimum: float) -> float:
    """收敛 SCIP 在边界处产生的浮点误差，不掩盖实质越界。"""

    tolerance = 1e-7
    if value < minimum - tolerance or value > 1.0 + tolerance:
        raise RuntimeError(f"求解器返回越界运行比例：{value}")
    return min(1.0, max(minimum, value))


def solve_continuous_schedule(
    data: EnergyParkDataset,
    *,
    target_tons_per_day: float,
    scenario_id: str,
    wind_profile: HourlyProfile | None = None,
    pv_profile: HourlyProfile | None = None,
    capacity_scale: float = RATED_CAPACITY_SCALE,
    minimum_load_fraction: float = MINIMUM_LOAD_FRACTION,
) -> ContinuousScheduleRun:
    """以连续负荷比例变量最小化单场景日电力及运维成本。"""

    if not scenario_id.strip():
        raise ValueError("场景编号不能为空")
    if capacity_scale <= 0:
        raise ValueError("额定产能倍数必须为正数")
    if not 0 < minimum_load_fraction <= 1:
        raise ValueError("最低运行比例必须在 (0, 1] 范围内")

    wind_profile = wind_profile or data.typical_wind
    pv_profile = pv_profile or data.typical_pv
    full_load_tons_per_hour = (
        data.technical.ammonia_tons_per_hour * capacity_scale
    )
    fraction_sum = target_tons_per_day / full_load_tons_per_hour
    minimum_daily_output = 24 * minimum_load_fraction * full_load_tons_per_hour
    maximum_daily_output = 24 * full_load_tons_per_hour
    if not minimum_daily_output - 1e-9 <= target_tons_per_day <= maximum_daily_output + 1e-9:
        raise ValueError(
            f"目标日产量必须在 {minimum_daily_output:g} 至 "
            f"{maximum_daily_output:g} 吨之间（10% 最低负荷）"
        )

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        return ContinuousScheduleRun(
            scenario_id=scenario_id,
            target_tons_per_day=target_tons_per_day,
            capacity_scale=capacity_scale,
            minimum_load_fraction=minimum_load_fraction,
            solver_status="SOLVER_UNAVAILABLE",
            process_fractions=[],
            solver_grid_purchase_mw=[],
            solver_grid_export_mw=[],
            operation=None,
            validation_report=None,
        )

    fractions = [
        solver.NumVar(minimum_load_fraction, 1.0, f"fraction_{hour:02d}")
        for hour in range(24)
    ]
    purchases = [
        solver.NumVar(0.0, solver.infinity(), f"purchase_{hour:02d}")
        for hour in range(24)
    ]
    exports = [
        solver.NumVar(0.0, solver.infinity(), f"export_{hour:02d}")
        for hour in range(24)
    ]
    import_direction = [
        solver.IntVar(0, 1, f"import_direction_{hour:02d}")
        for hour in range(24)
    ]
    solver.Add(solver.Sum(fractions) == fraction_sum)

    parameters = data.technical
    process_power_mw = (
        parameters.alkaline_power_mw
        + parameters.pem_power_mw
        + parameters.ammonia_power_mw
    ) * capacity_scale
    maximum_import_mw = parameters.ordinary_load_peak_mw + process_power_mw
    maximum_export_mw = parameters.wind_capacity_mw + parameters.pv_capacity_mw
    objective = solver.Objective()
    objective.SetMinimization()

    for hour in range(24):
        ordinary_load_mw = (
            parameters.ordinary_load_peak_mw * data.ordinary_load.values[hour]
        )
        wind_generation_mw = (
            parameters.wind_capacity_mw * wind_profile.values[hour]
        )
        pv_generation_mw = parameters.pv_capacity_mw * pv_profile.values[hour]
        process_om_per_fraction = capacity_scale * 1000 * (
            parameters.alkaline_power_mw * data.costs.alkaline_om_yuan_per_kwh
            + parameters.pem_power_mw * data.costs.pem_om_yuan_per_kwh
            + parameters.ammonia_power_mw * data.costs.ammonia_om_yuan_per_kwh
        )
        solver.Add(
            wind_generation_mw
            + pv_generation_mw
            + purchases[hour]
            == ordinary_load_mw
            + process_power_mw * fractions[hour]
            + exports[hour]
        )
        solver.Add(purchases[hour] <= maximum_import_mw * import_direction[hour])
        solver.Add(exports[hour] <= maximum_export_mw * (1 - import_direction[hour]))
        purchase_price = _purchase_price_yuan_per_kwh(
            data,
            int(data.ordinary_load.periods[hour].split(":", maxsplit=1)[0]),
        )
        export_price = _weighted_export_price(
            data,
            wind_generation_mw,
            pv_generation_mw,
        )
        objective.SetCoefficient(purchases[hour], 1000 * purchase_price)
        objective.SetCoefficient(exports[hour], -1000 * export_price)
        objective.SetCoefficient(fractions[hour], process_om_per_fraction)

    status_code = solver.Solve()
    status_names = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    solver_status = status_names.get(status_code, "UNKNOWN")
    if status_code not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        return ContinuousScheduleRun(
            scenario_id=scenario_id,
            target_tons_per_day=target_tons_per_day,
            capacity_scale=capacity_scale,
            minimum_load_fraction=minimum_load_fraction,
            solver_status=solver_status,
            process_fractions=[],
            solver_grid_purchase_mw=[],
            solver_grid_export_mw=[],
            operation=None,
            validation_report=None,
        )

    process_values = [
        _normalize_solver_fraction(variable.solution_value(), minimum_load_fraction)
        for variable in fractions
    ]
    purchase_values = [variable.solution_value() for variable in purchases]
    export_values = [variable.solution_value() for variable in exports]
    operation = compute_energy_schedule(
        data,
        process_values,
        capacity_scale=capacity_scale,
        wind_profile=wind_profile,
        pv_profile=pv_profile,
    )
    validation_report = validate_continuous_schedule(
        data,
        target_tons_per_day=target_tons_per_day,
        process_fractions=process_values,
        solver_grid_purchase_mw=purchase_values,
        solver_grid_export_mw=export_values,
        operation=operation,
        capacity_scale=capacity_scale,
        minimum_load_fraction=minimum_load_fraction,
        wind_profile=wind_profile,
        pv_profile=pv_profile,
    )
    return ContinuousScheduleRun(
        scenario_id=scenario_id,
        target_tons_per_day=target_tons_per_day,
        capacity_scale=capacity_scale,
        minimum_load_fraction=minimum_load_fraction,
        solver_status=solver_status,
        process_fractions=process_values,
        solver_grid_purchase_mw=purchase_values,
        solver_grid_export_mw=export_values,
        operation=operation,
        validation_report=validation_report,
    )


def run_continuous_question_three(
    data: EnergyParkDataset,
    *,
    days_per_scenario: int = 15,
    target_levels_tons_per_day: list[int] | None = None,
    discrete_result: DiscreteQuestionTwoResult | None = None,
) -> ContinuousQuestionThreeResult:
    """求解问题三并按相同场景、产量与成本口径对比问题二。"""

    if days_per_scenario <= 0:
        raise ValueError("每个场景代表的天数必须为正数")
    targets = target_levels_tons_per_day or [72, 63, 54, 45, 36]
    if not targets or any(target <= 0 for target in targets):
        raise ValueError("日产量档位必须是正数列表")

    discrete_result = discrete_result or run_discrete_question_two(
        data,
        days_per_scenario=days_per_scenario,
        target_levels_tons_per_day=targets,
    )
    expected_targets = [float(target) for target in targets]
    if discrete_result.target_levels_tons_per_day != expected_targets:
        raise ValueError("问题二对照结果的日产量档位必须与问题三一致")
    if discrete_result.days_per_scenario != days_per_scenario:
        raise ValueError("问题二与问题三的场景天数权重必须一致")

    scenarios = [("typical", data.typical_wind, data.typical_pv)]
    scenarios.extend(
        (
            f"wind_{wind_index}_pv_{pv_index}",
            data.wind_scenarios[wind_index - 1],
            data.pv_scenarios[pv_index - 1],
        )
        for wind_index, pv_index in product(
            range(1, len(data.wind_scenarios) + 1),
            range(1, len(data.pv_scenarios) + 1),
        )
    )

    typical_runs: list[ContinuousScheduleRun] = []
    scenario_runs: list[ContinuousScheduleRun] = []
    scenario_runs_by_target: dict[float, list[ContinuousScheduleRun]] = {}
    for target in targets:
        runs_for_target: list[ContinuousScheduleRun] = []
        for scenario_id, wind_profile, pv_profile in scenarios:
            run = solve_continuous_schedule(
                data,
                target_tons_per_day=target,
                scenario_id=scenario_id,
                wind_profile=wind_profile,
                pv_profile=pv_profile,
            )
            if run.solver_status != "OPTIMAL":
                raise RuntimeError(
                    f"连续调度日产量 {target:g} 吨、场景 {scenario_id} "
                    f"未证明最优：{run.solver_status}"
                )
            if run.operation is None or run.validation_report is None:
                raise RuntimeError(
                    f"连续调度日产量 {target:g} 吨、场景 {scenario_id} 缺少校验结果"
                )
            if not run.validation_report.is_valid:
                raise RuntimeError(
                    f"连续调度日产量 {target:g} 吨、场景 {scenario_id} "
                    f"校验失败：{run.validation_report.errors}"
                )
            runs_for_target.append(run)
            if scenario_id == "typical":
                typical_runs.append(run)
            else:
                scenario_runs.append(run)
        scenario_runs_by_target[float(target)] = runs_for_target[1:]

    representative_year_days = (
        len(data.wind_scenarios) * len(data.pv_scenarios) * days_per_scenario
    )
    annual_summaries: list[AnnualProductionSummary] = []
    for target in targets:
        runs = scenario_runs_by_target[float(target)]
        costs_per_ton = [
            run.operation.costs.cost_including_ammonia_capex_per_ton_yuan
            for run in runs
            if run.operation is not None
        ]
        annual_total_cost = sum(
            run.operation.costs.cost_including_ammonia_capex_yuan
            * days_per_scenario
            for run in runs
            if run.operation is not None
        )
        green_status_contest = {"全满足": 0, "部分满足": 0, "全不满足": 0}
        green_status_physical = {"全满足": 0, "部分满足": 0, "全不满足": 0}
        for run in runs:
            if run.operation is None:
                continue
            metrics = run.operation.green_metrics
            green_status_contest[
                classify_green_metrics(metrics, use_physical_self_use=False)
            ] += days_per_scenario
            green_status_physical[
                classify_green_metrics(metrics, use_physical_self_use=True)
            ] += days_per_scenario

        annual_production = float(target) * representative_year_days
        annual_summaries.append(
            AnnualProductionSummary(
                target_tons_per_day=float(target),
                representative_year_days=representative_year_days,
                annual_production_tons=annual_production,
                annual_total_cost_yuan=annual_total_cost,
                annual_cost_per_ton_yuan=annual_total_cost / annual_production,
                annual_grid_purchase_mwh=sum(
                    run.operation.grid_purchase_mwh * days_per_scenario
                    for run in runs
                    if run.operation is not None
                ),
                annual_grid_export_mwh=sum(
                    run.operation.grid_export_mwh * days_per_scenario
                    for run in runs
                    if run.operation is not None
                ),
                daily_cost_per_ton_min_yuan=min(costs_per_ton),
                daily_cost_per_ton_max_yuan=max(costs_per_ton),
                daily_cost_per_ton_average_yuan=fmean(costs_per_ton),
                green_status_days_contest_formula=green_status_contest,
                green_status_days_physical_self_use=green_status_physical,
            )
        )

    discrete_summary_by_target = {
        summary.target_tons_per_day: summary
        for summary in discrete_result.annual_summaries
    }
    continuous_summary_by_target = {
        summary.target_tons_per_day: summary for summary in annual_summaries
    }
    comparison_by_target: list[dict[str, float]] = []
    for target in expected_targets:
        discrete_summary = discrete_summary_by_target[target]
        continuous_summary = continuous_summary_by_target[target]
        comparison_by_target.append(
            {
                "target_tons_per_day": target,
                "discrete_annual_cost_yuan": discrete_summary.annual_total_cost_yuan,
                "continuous_annual_cost_yuan": continuous_summary.annual_total_cost_yuan,
                "annual_cost_difference_yuan": (
                    continuous_summary.annual_total_cost_yuan
                    - discrete_summary.annual_total_cost_yuan
                ),
                "discrete_annual_cost_per_ton_yuan": (
                    discrete_summary.annual_cost_per_ton_yuan
                ),
                "continuous_annual_cost_per_ton_yuan": (
                    continuous_summary.annual_cost_per_ton_yuan
                ),
                "annual_cost_per_ton_difference_yuan": (
                    continuous_summary.annual_cost_per_ton_yuan
                    - discrete_summary.annual_cost_per_ton_yuan
                ),
                "annual_grid_purchase_difference_mwh": (
                    continuous_summary.annual_grid_purchase_mwh
                    - discrete_summary.annual_grid_purchase_mwh
                ),
                "annual_grid_export_difference_mwh": (
                    continuous_summary.annual_grid_export_mwh
                    - discrete_summary.annual_grid_export_mwh
                ),
            }
        )

    best_typical = min(
        typical_runs,
        key=lambda run: run.operation.costs.cost_including_ammonia_capex_per_ton_yuan,
    )
    best_annual = min(
        annual_summaries,
        key=lambda summary: summary.annual_cost_per_ton_yuan,
    )
    return ContinuousQuestionThreeResult(
        target_levels_tons_per_day=expected_targets,
        rated_capacity_scale=RATED_CAPACITY_SCALE,
        minimum_load_fraction=MINIMUM_LOAD_FRACTION,
        days_per_scenario=days_per_scenario,
        representative_year_days=representative_year_days,
        typical_runs=typical_runs,
        scenario_runs=scenario_runs,
        annual_summaries=annual_summaries,
        discrete_comparison_by_target=comparison_by_target,
        best_typical_target_tons_per_day=best_typical.target_tons_per_day,
        best_annual_cost_per_ton_target_tons_per_day=best_annual.target_tons_per_day,
        modeling_assumptions=[
            "将 10% 解释为每个小时都保持的最低连续负荷，不包含完全停机状态。",
            "问题二和问题三使用相同产量、场景权重及成本口径后再比较。",
            "绿电指标在最低成本调度后计算，不作为硬约束。",
            "代表年按 24 个场景各 15 天共 360 天汇总。",
        ],
    )
