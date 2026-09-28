"""用 CP-SAT 求解电工杯问题二的离散开停排程。"""

from math import isclose
from itertools import product
from statistics import fmean

from ortools.sat.python import cp_model
from pydantic import BaseModel, Field

from .energy_park import (
    TypicalDayResult,
    _purchase_price_yuan_per_kwh,
    compute_energy_schedule,
)
from .energy_park_models import EnergyParkDataset, HourlyProfile
from .energy_park_discrete_validator import validate_discrete_schedule
from .energy_park_validator import EnergyParkValidationReport


RATED_CAPACITY_SCALE = 2.0
COST_INTEGER_SCALE = 100_000


class DiscreteScheduleRun(BaseModel):
    """保存单一产量和单一风光场景的离散调度结果。"""

    scenario_id: str
    target_tons_per_day: float
    solver_status: str
    hourly_on: list[bool] = Field(min_length=0, max_length=24)
    operation: TypicalDayResult | None
    validation_report: EnergyParkValidationReport | None


class AnnualProductionSummary(BaseModel):
    """汇总一个日产量档位在 24 个代表场景下的全年结果。"""

    target_tons_per_day: float
    representative_year_days: int
    annual_production_tons: float
    annual_total_cost_yuan: float
    annual_cost_per_ton_yuan: float
    annual_grid_purchase_mwh: float
    annual_grid_export_mwh: float
    daily_cost_per_ton_min_yuan: float
    daily_cost_per_ton_max_yuan: float
    daily_cost_per_ton_average_yuan: float
    green_status_days_contest_formula: dict[str, int]
    green_status_days_physical_self_use: dict[str, int]


class DiscreteQuestionTwoResult(BaseModel):
    """保存问题二典型日解、场景解及代表年统计。"""

    target_levels_tons_per_day: list[float]
    rated_capacity_scale: float
    days_per_scenario: int
    representative_year_days: int
    typical_runs: list[DiscreteScheduleRun]
    scenario_runs: list[DiscreteScheduleRun]
    annual_summaries: list[AnnualProductionSummary]
    best_typical_target_tons_per_day: float
    best_annual_cost_per_ton_target_tons_per_day: float
    modeling_assumptions: list[str]


def _hourly_net_operating_cost(
    data: EnergyParkDataset,
    hour_index: int,
    *,
    process_is_on: bool,
    capacity_scale: float,
    wind_profile: HourlyProfile,
    pv_profile: HourlyProfile,
) -> float:
    """计算某小时开机或停机时的购电、售电和设备运维净成本。"""

    parameters = data.technical
    costs = data.costs
    ordinary_load_mw = (
        parameters.ordinary_load_peak_mw
        * data.ordinary_load.values[hour_index]
    )
    wind_mw = parameters.wind_capacity_mw * wind_profile.values[hour_index]
    pv_mw = parameters.pv_capacity_mw * pv_profile.values[hour_index]
    generation_mw = wind_mw + pv_mw
    operating_fraction = 1.0 if process_is_on else 0.0
    process_mw = (
        parameters.alkaline_power_mw
        + parameters.pem_power_mw
        + parameters.ammonia_power_mw
    ) * capacity_scale * operating_fraction
    load_mw = ordinary_load_mw + process_mw
    purchase_mw = max(load_mw - generation_mw, 0.0)
    export_mw = max(generation_mw - load_mw, 0.0)

    period = data.ordinary_load.periods[hour_index]
    hour = int(period.split(":", maxsplit=1)[0])
    grid_cost = purchase_mw * 1000 * _purchase_price_yuan_per_kwh(data, hour)
    if generation_mw > 0:
        wind_export_mw = export_mw * wind_mw / generation_mw
        pv_export_mw = export_mw * pv_mw / generation_mw
    else:
        wind_export_mw = 0.0
        pv_export_mw = 0.0
    export_revenue = (
        wind_export_mw * 1000 * costs.wind_export_price_yuan_per_kwh
        + pv_export_mw * 1000 * costs.pv_export_price_yuan_per_kwh
    )
    process_om = (
        parameters.alkaline_power_mw * capacity_scale * operating_fraction * 1000
        * costs.alkaline_om_yuan_per_kwh
        + parameters.pem_power_mw * capacity_scale * operating_fraction * 1000
        * costs.pem_om_yuan_per_kwh
        + parameters.ammonia_power_mw * capacity_scale * operating_fraction * 1000
        * costs.ammonia_om_yuan_per_kwh
    )
    return grid_cost - export_revenue + process_om


def _to_integer_cost(coefficient_yuan: float) -> int:
    """把购售电和运维的线性系数缩放成 CP-SAT 可用整数。"""

    scaled = coefficient_yuan * COST_INTEGER_SCALE
    integer_value = int(round(scaled))
    if not isclose(scaled, integer_value, rel_tol=0, abs_tol=1e-5):
        raise ValueError("成本精度超过 CP-SAT 整数目标的 0.00001 元分辨率")
    return integer_value


def solve_discrete_schedule(
    data: EnergyParkDataset,
    *,
    target_tons_per_day: float,
    scenario_id: str,
    wind_profile: HourlyProfile | None = None,
    pv_profile: HourlyProfile | None = None,
    capacity_scale: float = RATED_CAPACITY_SCALE,
) -> DiscreteScheduleRun:
    """在逐小时全开或全停的条件下，求最低成本生产时段。"""

    if not scenario_id.strip():
        raise ValueError("场景编号不能为空")
    if capacity_scale <= 0:
        raise ValueError("额定产能倍数必须为正数")
    wind_profile = wind_profile or data.typical_wind
    pv_profile = pv_profile or data.typical_pv

    tons_per_full_load_hour = (
        data.technical.ammonia_tons_per_hour * capacity_scale
    )
    raw_on_hours = target_tons_per_day / tons_per_full_load_hour
    on_hours = round(raw_on_hours)
    if not isclose(raw_on_hours, on_hours, rel_tol=0, abs_tol=1e-9):
        raise ValueError("目标日产量不能由整数个额定开机小时实现")
    if not 1 <= on_hours <= 24:
        raise ValueError("目标日产量必须在 1 至 24 个额定开机小时范围内")

    model = cp_model.CpModel()
    on = [model.NewBoolVar(f"on_{hour:02d}") for hour in range(24)]
    model.Add(sum(on) == on_hours)

    # 每小时只有开/停两种负荷状态，先计算两种确定的净成本再线性化。
    objective_terms = []
    for hour in range(24):
        off_cost = _hourly_net_operating_cost(
            data,
            hour,
            process_is_on=False,
            capacity_scale=capacity_scale,
            wind_profile=wind_profile,
            pv_profile=pv_profile,
        )
        on_cost = _hourly_net_operating_cost(
            data,
            hour,
            process_is_on=True,
            capacity_scale=capacity_scale,
            wind_profile=wind_profile,
            pv_profile=pv_profile,
        )
        objective_terms.append(_to_integer_cost(on_cost - off_cost) * on[hour])
    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return DiscreteScheduleRun(
            scenario_id=scenario_id,
            target_tons_per_day=target_tons_per_day,
            solver_status=status,
            hourly_on=[],
            operation=None,
            validation_report=None,
        )

    hourly_on = [bool(solver.Value(variable)) for variable in on]
    fractions = [1.0 if running else 0.0 for running in hourly_on]
    operation = compute_energy_schedule(
        data,
        fractions,
        capacity_scale=capacity_scale,
        wind_profile=wind_profile,
        pv_profile=pv_profile,
        include_cost_sensitivity=False,
    )
    validation_report = validate_discrete_schedule(
        data,
        target_tons_per_day=target_tons_per_day,
        hourly_on=hourly_on,
        operation=operation,
        capacity_scale=capacity_scale,
        wind_profile=wind_profile,
        pv_profile=pv_profile,
    )
    return DiscreteScheduleRun(
        scenario_id=scenario_id,
        target_tons_per_day=target_tons_per_day,
        solver_status=status,
        hourly_on=hourly_on,
        operation=operation,
        validation_report=validation_report,
    )


def classify_green_metrics(metrics, *, use_physical_self_use: bool) -> str:
    """按所选自用率口径将三项政策指标分类。"""

    self_use_passes = (
        metrics.physical_self_use_passes
        if use_physical_self_use
        else metrics.contest_formula_self_use_passes
    )
    statuses = [
        self_use_passes,
        metrics.green_share_of_load_passes,
        metrics.renewable_export_passes,
    ]
    passed = sum(status is True for status in statuses)
    if passed == len(statuses):
        return "全满足"
    if passed > 0:
        return "部分满足"
    return "全不满足"


def run_discrete_question_two(
    data: EnergyParkDataset,
    *,
    days_per_scenario: int = 15,
    target_levels_tons_per_day: list[int] | None = None,
) -> DiscreteQuestionTwoResult:
    """求解典型日及 24 种风光组合下的五档离散日产量。"""

    if days_per_scenario <= 0:
        raise ValueError("每个场景代表的天数必须为正数")
    targets = target_levels_tons_per_day or [72, 63, 54, 45, 36]
    if not targets or any(target <= 0 for target in targets):
        raise ValueError("日产量档位必须是正数列表")

    scenarios = [("typical", data.typical_wind, data.typical_pv)]
    scenarios.extend(
        (
            f"wind_{wind_index}_pv_{pv_index}",
            wind_profile,
            pv_profile,
        )
        for wind_index, wind_profile in enumerate(data.wind_scenarios, start=1)
        for pv_index, pv_profile in enumerate(data.pv_scenarios, start=1)
    )

    typical_runs: list[DiscreteScheduleRun] = []
    scenario_runs: list[DiscreteScheduleRun] = []
    scenario_runs_by_target: dict[float, list[DiscreteScheduleRun]] = {}
    for target in targets:
        runs_for_target: list[DiscreteScheduleRun] = []
        for scenario_id, wind_profile, pv_profile in scenarios:
            run = solve_discrete_schedule(
                data,
                target_tons_per_day=target,
                scenario_id=scenario_id,
                wind_profile=wind_profile,
                pv_profile=pv_profile,
            )
            if run.solver_status != "OPTIMAL":
                raise RuntimeError(
                    f"日产量 {target:g} 吨、场景 {scenario_id} 未证明最优："
                    f"{run.solver_status}"
                )
            if run.operation is None or run.validation_report is None:
                raise RuntimeError(
                    f"日产量 {target:g} 吨、场景 {scenario_id} 缺少可核验结果"
                )
            if not run.validation_report.is_valid:
                raise RuntimeError(
                    f"日产量 {target:g} 吨、场景 {scenario_id} 校验失败："
                    f"{run.validation_report.errors}"
                )
            runs_for_target.append(run)
            if scenario_id == "typical":
                typical_runs.append(run)
            else:
                scenario_runs.append(run)
        scenario_runs_by_target[float(target)] = runs_for_target[1:]

    representative_year_days = len(data.wind_scenarios) * len(data.pv_scenarios) * days_per_scenario
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
        annual_production = float(target) * representative_year_days
        green_status_days_contest = {"全满足": 0, "部分满足": 0, "全不满足": 0}
        green_status_days_physical = {"全满足": 0, "部分满足": 0, "全不满足": 0}
        for run in runs:
            if run.operation is None:
                continue
            metrics = run.operation.green_metrics
            contest_status = classify_green_metrics(
                metrics,
                use_physical_self_use=False,
            )
            physical_status = classify_green_metrics(
                metrics,
                use_physical_self_use=True,
            )
            green_status_days_contest[contest_status] += days_per_scenario
            green_status_days_physical[physical_status] += days_per_scenario
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
                green_status_days_contest_formula=green_status_days_contest,
                green_status_days_physical_self_use=green_status_days_physical,
            )
        )

    best_typical_run = min(
        typical_runs,
        key=lambda run: run.operation.costs.cost_including_ammonia_capex_per_ton_yuan,
    )
    best_annual_summary = min(
        annual_summaries,
        key=lambda summary: summary.annual_cost_per_ton_yuan,
    )
    return DiscreteQuestionTwoResult(
        target_levels_tons_per_day=[float(target) for target in targets],
        rated_capacity_scale=RATED_CAPACITY_SCALE,
        days_per_scenario=days_per_scenario,
        representative_year_days=representative_year_days,
        typical_runs=typical_runs,
        scenario_runs=scenario_runs,
        annual_summaries=annual_summaries,
        best_typical_target_tons_per_day=best_typical_run.target_tons_per_day,
        best_annual_cost_per_ton_target_tons_per_day=(
            best_annual_summary.target_tons_per_day
        ),
        modeling_assumptions=[
            "额定装置对应 72 吨/日；开机时三套装置同步满负荷，停机时同步为零。",
            "每档产量按 3 吨/小时及整数开机小时实现。",
            "绿电指标作为最低成本排程的事后评价，不作为优化硬约束。",
            "24 个风光组合各代表 15 天，代表年按 360 天计算。",
            "资本成本采用附件中已声明的直线摊销估算口径。",
        ],
    )
