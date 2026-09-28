"""从原始输入独立重算能源园区基准结果。"""

from math import isclose, isfinite
import re

from pydantic import BaseModel

from .energy_park import TypicalDayResult
from .energy_park_models import EnergyParkDataset


class EnergyParkValidationReport(BaseModel):
    """报告逐时平衡及汇总结果的独立核验状态。"""

    is_valid: bool
    errors: list[str]
    maximum_power_balance_error_mw: float


def _close(actual: float, expected: float) -> bool:
    return isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6)


def _price_for_hour(data: EnergyParkDataset, hour: int) -> float:
    """独立按附件电价时段重新选择购电价格。"""

    costs = data.costs
    if (10 <= hour < 15) or (18 <= hour < 21):
        return costs.purchase_price_peak_yuan_per_kwh
    if (7 <= hour < 10) or (15 <= hour < 18) or (21 <= hour < 23):
        return costs.purchase_price_flat_yuan_per_kwh
    return costs.purchase_price_valley_yuan_per_kwh


def _hour(period: str) -> int | None:
    match = re.match(r"^\s*(\d{1,2}):\d{2}\s*-", period)
    return int(match.group(1)) if match is not None else None


def _check_value(errors: list[str], label: str, actual: float, expected: float) -> None:
    if not isfinite(actual) or not _close(actual, expected):
        errors.append(f"{label}不一致：结果为 {actual}，重算为 {expected}")


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0 else None


def validate_typical_day(
    data: EnergyParkDataset,
    result: TypicalDayResult,
) -> EnergyParkValidationReport:
    """不复用求解结果的汇总字段，逐时重算并比较报告。"""

    errors: list[str] = []
    parameters = data.technical
    costs = data.costs
    expected_process_load_mw = (
        parameters.alkaline_power_mw
        + parameters.pem_power_mw
        + parameters.ammonia_power_mw
    )
    expected_rows: list[dict[str, float | str]] = []
    maximum_balance_error = 0.0
    expected_grid_purchase_cost = 0.0
    expected_wind_export_revenue = 0.0
    expected_pv_export_revenue = 0.0

    if len(result.hourly_balances) != 24:
        errors.append(f"逐时结果应有 24 行，实际为 {len(result.hourly_balances)} 行")

    for index, period in enumerate(data.ordinary_load.periods):
        ordinary_load = (
            parameters.ordinary_load_peak_mw * data.ordinary_load.values[index]
        )
        wind_generation = (
            parameters.wind_capacity_mw * data.typical_wind.values[index]
        )
        pv_generation = parameters.pv_capacity_mw * data.typical_pv.values[index]
        generation = wind_generation + pv_generation
        load = ordinary_load + expected_process_load_mw
        purchase = max(load - generation, 0.0)
        export = max(generation - load, 0.0)
        residual = generation + purchase - load - export
        maximum_balance_error = max(maximum_balance_error, abs(residual))

        expected_rows.append(
            {
                "period": period,
                "ordinary_load_mw": ordinary_load,
                "process_load_mw": expected_process_load_mw,
                "total_load_mw": load,
                "wind_generation_mw": wind_generation,
                "pv_generation_mw": pv_generation,
                "renewable_generation_mw": generation,
                "grid_purchase_mw": purchase,
                "grid_export_mw": export,
                "power_balance_residual_mw": residual,
            }
        )
        hour = _hour(period)
        if hour is None:
            errors.append(f"无法核验未知小时标签：{period}")
        else:
            expected_grid_purchase_cost += (
                purchase * 1000 * _price_for_hour(data, hour)
            )
        if generation > 0:
            expected_wind_export_revenue += (
                export * wind_generation / generation * 1000
                * costs.wind_export_price_yuan_per_kwh
            )
            expected_pv_export_revenue += (
                export * pv_generation / generation * 1000
                * costs.pv_export_price_yuan_per_kwh
            )

    for index, expected_row in enumerate(expected_rows[: len(result.hourly_balances)]):
        actual_row = result.hourly_balances[index]
        if actual_row.period != expected_row["period"]:
            errors.append(
                f"第 {index + 1} 行时段标签不一致："
                f"结果为 {actual_row.period}，输入为 {expected_row['period']}"
            )
        for field in (
            "ordinary_load_mw",
            "process_load_mw",
            "total_load_mw",
            "wind_generation_mw",
            "pv_generation_mw",
            "renewable_generation_mw",
            "grid_purchase_mw",
            "grid_export_mw",
            "power_balance_residual_mw",
        ):
            _check_value(
                errors,
                f"{actual_row.period} 的 {field}",
                getattr(actual_row, field),
                float(expected_row[field]),
            )

    total_load = sum(float(row["total_load_mw"]) for row in expected_rows)
    wind_energy = sum(float(row["wind_generation_mw"]) for row in expected_rows)
    pv_energy = sum(float(row["pv_generation_mw"]) for row in expected_rows)
    renewable_energy = wind_energy + pv_energy
    purchased_energy = sum(float(row["grid_purchase_mw"]) for row in expected_rows)
    exported_energy = sum(float(row["grid_export_mw"]) for row in expected_rows)

    for label, actual, expected in (
        ("日电量：总用电", result.total_load_mwh, total_load),
        ("日电量：风电", result.wind_generation_mwh, wind_energy),
        ("日电量：光伏", result.pv_generation_mwh, pv_energy),
        ("日电量：新能源", result.renewable_generation_mwh, renewable_energy),
        ("日电量：网购", result.grid_purchase_mwh, purchased_energy),
        ("日电量：上网", result.grid_export_mwh, exported_energy),
    ):
        _check_value(errors, label, actual, expected)

    expected_metrics = {
        "contest_formula_self_use_ratio": _ratio(
            total_load - exported_energy - purchased_energy,
            renewable_energy,
        ),
        "physical_self_use_ratio": _ratio(
            renewable_energy - exported_energy,
            renewable_energy,
        ),
        "green_share_of_load_ratio": _ratio(
            renewable_energy - exported_energy,
            total_load,
        ),
        "renewable_export_ratio": _ratio(exported_energy, renewable_energy),
    }
    for field, expected in expected_metrics.items():
        actual = getattr(result.green_metrics, field)
        if actual is None or expected is None:
            if actual is not expected:
                errors.append(f"绿电指标 {field} 的不可计算状态不一致")
        else:
            _check_value(errors, f"绿电指标 {field}", actual, expected)

    expected_metric_statuses = {
        "contest_formula_self_use_passes": (
            expected_metrics["contest_formula_self_use_ratio"] > 0.60
            if expected_metrics["contest_formula_self_use_ratio"] is not None
            else None
        ),
        "physical_self_use_passes": (
            expected_metrics["physical_self_use_ratio"] > 0.60
            if expected_metrics["physical_self_use_ratio"] is not None
            else None
        ),
        "green_share_of_load_passes": (
            expected_metrics["green_share_of_load_ratio"] > 0.30
            if expected_metrics["green_share_of_load_ratio"] is not None
            else None
        ),
        "renewable_export_passes": (
            expected_metrics["renewable_export_ratio"] < 0.20
            if expected_metrics["renewable_export_ratio"] is not None
            else None
        ),
    }
    for field, expected in expected_metric_statuses.items():
        if getattr(result.green_metrics, field) is not expected:
            errors.append(f"绿电指标 {field} 与题面阈值判定不一致")

    if result.source_files != data.source_files:
        errors.append("结果来源文件清单与输入数据不一致")

    expected_wind_cost = wind_energy * 1000 * costs.wind_lcoe_yuan_per_kwh
    expected_pv_cost = pv_energy * 1000 * costs.pv_lcoe_yuan_per_kwh
    expected_alkaline_om = parameters.alkaline_power_mw * 24 * 1000 * costs.alkaline_om_yuan_per_kwh
    expected_pem_om = parameters.pem_power_mw * 24 * 1000 * costs.pem_om_yuan_per_kwh
    expected_ammonia_om = parameters.ammonia_power_mw * 24 * 1000 * costs.ammonia_om_yuan_per_kwh
    expected_export_revenue = expected_wind_export_revenue + expected_pv_export_revenue
    expected_variable_cost = (
        expected_wind_cost
        + expected_pv_cost
        + expected_grid_purchase_cost
        - expected_export_revenue
        + expected_alkaline_om
        + expected_pem_om
        + expected_ammonia_om
    )
    expected_production = parameters.ammonia_tons_per_hour * 24
    expected_capex_daily = (
        costs.ammonia_capex_yuan_per_kg_h2
        * (parameters.alkaline_hydrogen_kg_per_hour + parameters.pem_hydrogen_kg_per_hour)
        / costs.ammonia_lifetime_years
        / 365
    )
    expected_costs = {
        "wind_generation_cost_yuan": expected_wind_cost,
        "pv_generation_cost_yuan": expected_pv_cost,
        "renewable_generation_cost_yuan": expected_wind_cost + expected_pv_cost,
        "grid_purchase_cost_yuan": expected_grid_purchase_cost,
        "wind_export_revenue_yuan": expected_wind_export_revenue,
        "pv_export_revenue_yuan": expected_pv_export_revenue,
        "export_revenue_yuan": expected_export_revenue,
        "alkaline_om_yuan": expected_alkaline_om,
        "pem_om_yuan": expected_pem_om,
        "ammonia_om_yuan": expected_ammonia_om,
        "variable_operating_cost_yuan": expected_variable_cost,
        "variable_cost_per_ton_yuan": expected_variable_cost / expected_production,
        "ammonia_capex_daily_yuan": expected_capex_daily,
        "cost_including_ammonia_capex_yuan": expected_variable_cost + expected_capex_daily,
        "cost_including_ammonia_capex_per_ton_yuan": (
            expected_variable_cost + expected_capex_daily
        ) / expected_production,
    }
    for field, expected in expected_costs.items():
        _check_value(errors, f"成本 {field}", getattr(result.costs, field), expected)

    return EnergyParkValidationReport(
        is_valid=not errors,
        errors=errors,
        maximum_power_balance_error_mw=maximum_balance_error,
    )
