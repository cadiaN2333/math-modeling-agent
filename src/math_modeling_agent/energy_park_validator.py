"""从原始输入独立重算能源园区基准结果。"""

from math import isclose, isfinite
import re

from pydantic import BaseModel

from .energy_park import TypicalDayResult
from .energy_park_models import EnergyParkDataset, HourlyProfile


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


def _purchase_tier(hour: int) -> str:
    """独立确定小时所属的峰、平、谷电价类别。"""

    if (10 <= hour < 15) or (18 <= hour < 21):
        return "peak"
    if (7 <= hour < 10) or (15 <= hour < 18) or (21 <= hour < 23):
        return "flat"
    return "valley"


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
    *,
    wind_profile: HourlyProfile | None = None,
    pv_profile: HourlyProfile | None = None,
    require_cost_sensitivity: bool = True,
) -> EnergyParkValidationReport:
    """不复用结果汇总字段，按指定风光曲线和运行比例逐时复算。"""

    errors: list[str] = []
    parameters = data.technical
    costs = data.costs
    wind_profile = wind_profile or data.typical_wind
    pv_profile = pv_profile or data.typical_pv
    capacity_scale = result.capacity_scale
    process_fractions = result.process_fractions
    if len(process_fractions) != 24:
        errors.append("逐时运行比例必须有 24 项")
        process_fractions = [0.0] * 24
    if (
        wind_profile.periods != data.ordinary_load.periods
        or pv_profile.periods != data.ordinary_load.periods
    ):
        errors.append("validator 收到的风光曲线与常规负荷时段不一致")
    expected_process_load_mw = (
        parameters.alkaline_power_mw
        + parameters.pem_power_mw
        + parameters.ammonia_power_mw
    )
    expected_rows: list[dict[str, float | str]] = []
    maximum_balance_error = 0.0
    expected_grid_purchase_cost = 0.0
    expected_purchase_kwh_by_tier = {"peak": 0.0, "flat": 0.0, "valley": 0.0}
    expected_wind_export_revenue = 0.0
    expected_pv_export_revenue = 0.0

    if len(result.hourly_balances) != 24:
        errors.append(f"逐时结果应有 24 行，实际为 {len(result.hourly_balances)} 行")

    for index, period in enumerate(data.ordinary_load.periods):
        ordinary_load = (
            parameters.ordinary_load_peak_mw * data.ordinary_load.values[index]
        )
        wind_generation = parameters.wind_capacity_mw * wind_profile.values[index]
        pv_generation = parameters.pv_capacity_mw * pv_profile.values[index]
        generation = wind_generation + pv_generation
        process_load = (
            expected_process_load_mw
            * capacity_scale
            * process_fractions[index]
        )
        load = ordinary_load + process_load
        purchase = max(load - generation, 0.0)
        export = max(generation - load, 0.0)
        residual = generation + purchase - load - export
        maximum_balance_error = max(maximum_balance_error, abs(residual))

        expected_rows.append(
            {
                "period": period,
                "ordinary_load_mw": ordinary_load,
                "process_load_mw": process_load,
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
            tier = _purchase_tier(hour)
            expected_purchase_kwh_by_tier[tier] += purchase * 1000
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
    expected_production = (
        parameters.ammonia_tons_per_hour
        * capacity_scale
        * sum(process_fractions)
    )
    _check_value(
        errors,
        "日产氨量",
        result.ammonia_production_tons,
        expected_production,
    )

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
    total_scaled_fraction = capacity_scale * sum(process_fractions)
    expected_alkaline_om = (
        parameters.alkaline_power_mw
        * total_scaled_fraction
        * 1000
        * costs.alkaline_om_yuan_per_kwh
    )
    expected_pem_om = (
        parameters.pem_power_mw
        * total_scaled_fraction
        * 1000
        * costs.pem_om_yuan_per_kwh
    )
    expected_ammonia_om = (
        parameters.ammonia_power_mw
        * total_scaled_fraction
        * 1000
        * costs.ammonia_om_yuan_per_kwh
    )
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
    expected_capex_daily = (
        costs.ammonia_capex_yuan_per_kg_h2
        * (parameters.alkaline_hydrogen_kg_per_hour + parameters.pem_hydrogen_kg_per_hour)
        * capacity_scale
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

    expected_sensitivity_components = [
        (
            "wind_lcoe",
            "风电度电成本",
            costs.wind_lcoe_yuan_per_kwh,
            "元/kWh",
            "费用",
            expected_wind_cost,
            False,
        ),
        (
            "pv_lcoe",
            "光伏度电成本",
            costs.pv_lcoe_yuan_per_kwh,
            "元/kWh",
            "费用",
            expected_pv_cost,
            False,
        ),
        (
            "purchase_price_peak",
            "峰时网购电",
            costs.purchase_price_peak_yuan_per_kwh,
            "元/kWh",
            "费用",
            expected_purchase_kwh_by_tier["peak"]
            * costs.purchase_price_peak_yuan_per_kwh,
            False,
        ),
        (
            "purchase_price_flat",
            "平时网购电",
            costs.purchase_price_flat_yuan_per_kwh,
            "元/kWh",
            "费用",
            expected_purchase_kwh_by_tier["flat"]
            * costs.purchase_price_flat_yuan_per_kwh,
            False,
        ),
        (
            "purchase_price_valley",
            "谷时网购电",
            costs.purchase_price_valley_yuan_per_kwh,
            "元/kWh",
            "费用",
            expected_purchase_kwh_by_tier["valley"]
            * costs.purchase_price_valley_yuan_per_kwh,
            False,
        ),
        (
            "wind_export_price",
            "风电上网收入",
            costs.wind_export_price_yuan_per_kwh,
            "元/kWh",
            "收入",
            expected_wind_export_revenue,
            False,
        ),
        (
            "pv_export_price",
            "光伏上网收入",
            costs.pv_export_price_yuan_per_kwh,
            "元/kWh",
            "收入",
            expected_pv_export_revenue,
            False,
        ),
        (
            "alkaline_om",
            "ALK 电解槽运维费",
            costs.alkaline_om_yuan_per_kwh,
            "元/kWh",
            "费用",
            expected_alkaline_om,
            False,
        ),
        (
            "pem_om",
            "PEM 电解槽运维费",
            costs.pem_om_yuan_per_kwh,
            "元/kWh",
            "费用",
            expected_pem_om,
            False,
        ),
        (
            "ammonia_om",
            "合成氨装置运维费",
            costs.ammonia_om_yuan_per_kwh,
            "元/kWh",
            "费用",
            expected_ammonia_om,
            False,
        ),
        (
            "ammonia_capex",
            "合成氨装置资本摊销",
            costs.ammonia_capex_yuan_per_kg_h2,
            "元/(kgH₂/h)",
            "费用",
            expected_capex_daily,
            True,
        ),
    ]
    expected_sensitivity_assumption = (
        "分别将每个成本或收入参数相对上调 1%，保持当前运行计划、购售电量及产量不变；"
        "该敏感度是固定调度下的成本核算影响，不代表重新优化后的结果。"
    )
    if (
        require_cost_sensitivity
        and result.costs.sensitivity_assumption != expected_sensitivity_assumption
    ):
        errors.append("成本敏感度的扰动及固定调度假设不一致")
    if (
        require_cost_sensitivity
        and len(result.costs.sensitivity_analysis)
        != len(expected_sensitivity_components)
    ):
        errors.append(
            "成本敏感度分项数量不一致："
            f"结果为 {len(result.costs.sensitivity_analysis)}，"
            f"重算为 {len(expected_sensitivity_components)}"
        )
    for index, expected in enumerate(expected_sensitivity_components):
        if not require_cost_sensitivity:
            break
        if index >= len(result.costs.sensitivity_analysis):
            break
        (
            component_id,
            name,
            parameter_value,
            parameter_unit,
            amount_type,
            amount_yuan,
            is_capex,
        ) = expected
        actual = result.costs.sensitivity_analysis[index]
        for field, expected_value in (
            ("component_id", component_id),
            ("component_name", name),
            ("base_parameter_value", parameter_value),
            ("parameter_unit", parameter_unit),
            ("amount_type", amount_type),
        ):
            if getattr(actual, field) != expected_value:
                errors.append(
                    f"成本敏感度第 {index + 1} 项的 {field} 不一致："
                    f"结果为 {getattr(actual, field)}，重算为 {expected_value}"
                )
        _check_value(
            errors,
            f"成本敏感度 {component_id} 的基准金额",
            actual.base_amount_yuan,
            amount_yuan,
        )
        direction = -1.0 if amount_type == "收入" else 1.0
        expected_delta = direction * amount_yuan * 0.01 / expected_production
        _check_value(
            errors,
            f"成本敏感度 {component_id} 的运行成本影响",
            actual.variable_cost_delta_yuan_per_ton_for_one_percent_increase,
            0.0 if is_capex else expected_delta,
        )
        _check_value(
            errors,
            f"成本敏感度 {component_id} 的含资本摊销成本影响",
            actual.capex_included_cost_delta_yuan_per_ton_for_one_percent_increase,
            expected_delta,
        )

    return EnergyParkValidationReport(
        is_valid=not errors,
        errors=errors,
        maximum_power_balance_error_mw=maximum_balance_error,
    )
