"""独立检查电工杯问题三的连续功率计划。"""

from math import isclose

from .energy_park import TypicalDayResult
from .energy_park_models import EnergyParkDataset, HourlyProfile
from .energy_park_validator import EnergyParkValidationReport, validate_typical_day


def validate_continuous_schedule(
    data: EnergyParkDataset,
    *,
    target_tons_per_day: float,
    process_fractions: list[float],
    solver_grid_purchase_mw: list[float],
    solver_grid_export_mw: list[float],
    operation: TypicalDayResult,
    capacity_scale: float,
    minimum_load_fraction: float,
    wind_profile: HourlyProfile | None = None,
    pv_profile: HourlyProfile | None = None,
) -> EnergyParkValidationReport:
    """重算日产量、连续负荷上下限及无同时购售电条件。"""

    base_report = validate_typical_day(
        data,
        operation,
        wind_profile=wind_profile,
        pv_profile=pv_profile,
    )
    errors = list(base_report.errors)
    expected_hour_count = 24
    if len(process_fractions) != expected_hour_count:
        errors.append("连续运行比例应有 24 项")
    if len(solver_grid_purchase_mw) != expected_hour_count:
        errors.append("求解器购电功率应有 24 项")
    if len(solver_grid_export_mw) != expected_hour_count:
        errors.append("求解器上网功率应有 24 项")

    if len(process_fractions) == expected_hour_count:
        if any(
            not minimum_load_fraction - 1e-8 <= fraction <= 1 + 1e-8
            for fraction in process_fractions
        ):
            errors.append("连续运行比例超出 10% 至 100% 设备边界")
        expected_sum = target_tons_per_day / (
            data.technical.ammonia_tons_per_hour * capacity_scale
        )
        if not isclose(sum(process_fractions), expected_sum, rel_tol=0, abs_tol=1e-6):
            errors.append("连续运行比例合计未满足目标日产量")
        if operation.process_fractions != process_fractions:
            errors.append("运行结果记录的连续负荷比例与求解变量不一致")

    if len(solver_grid_purchase_mw) == expected_hour_count and len(
        solver_grid_export_mw
    ) == expected_hour_count:
        for index, (purchase, export) in enumerate(
            zip(solver_grid_purchase_mw, solver_grid_export_mw)
        ):
            if purchase > 1e-7 and export > 1e-7:
                errors.append(f"第 {index + 1} 小时同时购电和上网")
            if not isclose(
                purchase,
                operation.hourly_balances[index].grid_purchase_mw,
                rel_tol=0,
                abs_tol=1e-6,
            ):
                errors.append(f"第 {index + 1} 小时求解器购电与平衡重算不一致")
            if not isclose(
                export,
                operation.hourly_balances[index].grid_export_mw,
                rel_tol=0,
                abs_tol=1e-6,
            ):
                errors.append(f"第 {index + 1} 小时求解器上网与平衡重算不一致")

    if not isclose(
        operation.ammonia_production_tons,
        target_tons_per_day,
        rel_tol=0,
        abs_tol=1e-6,
    ):
        errors.append("连续计划复算日产量与目标不一致")

    return base_report.model_copy(update={"is_valid": not errors, "errors": errors})
