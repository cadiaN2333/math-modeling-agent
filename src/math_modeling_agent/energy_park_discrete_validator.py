"""独立检查电工杯问题二的离散开停方案。"""

from math import isclose

from .energy_park import TypicalDayResult
from .energy_park_models import EnergyParkDataset, HourlyProfile
from .energy_park_validator import (
    EnergyParkValidationReport,
    validate_typical_day,
)


def validate_discrete_schedule(
    data: EnergyParkDataset,
    *,
    target_tons_per_day: float,
    hourly_on: list[bool],
    operation: TypicalDayResult,
    capacity_scale: float,
    wind_profile: HourlyProfile | None = None,
    pv_profile: HourlyProfile | None = None,
) -> EnergyParkValidationReport:
    """核对日产量、开机小时和操作计划中的功率/成本结果。"""

    base_report = validate_typical_day(
        data,
        operation,
        wind_profile=wind_profile,
        pv_profile=pv_profile,
    )
    errors = list(base_report.errors)
    if len(hourly_on) != 24:
        errors.append(f"离散开停状态应有 24 项，实际为 {len(hourly_on)} 项")
    if capacity_scale <= 0:
        errors.append("装置产能倍数必须为正数")
    elif not isclose(
        operation.capacity_scale,
        capacity_scale,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        errors.append("运行结果记录的产能倍数与求解输入不一致")

    full_load_tons_per_hour = (
        data.technical.ammonia_tons_per_hour * capacity_scale
    )
    if full_load_tons_per_hour <= 0:
        errors.append("满负荷产氨速率必须为正数")
        expected_on_hours = None
    else:
        exact_hours = target_tons_per_day / full_load_tons_per_hour
        rounded_hours = round(exact_hours)
        expected_on_hours = (
            rounded_hours
            if isclose(exact_hours, rounded_hours, rel_tol=0, abs_tol=1e-9)
            else None
        )
        if expected_on_hours is None:
            errors.append("目标日产量不能由整数个满负荷小时实现")

    actual_on_hours = sum(flag is True for flag in hourly_on)
    if expected_on_hours is not None and actual_on_hours != expected_on_hours:
        errors.append(
            f"目标日产量 {target_tons_per_day:g} 吨需要开机 {expected_on_hours} 小时，"
            f"实际为 {actual_on_hours} 小时"
        )

    if len(operation.process_fractions) == 24 and len(hourly_on) == 24:
        expected_fractions = [1.0 if flag is True else 0.0 for flag in hourly_on]
        if operation.process_fractions != expected_fractions:
            errors.append("运行结果中的设备负荷比例与开停状态不一致")

    if not isclose(
        operation.ammonia_production_tons,
        target_tons_per_day,
        rel_tol=0,
        abs_tol=1e-6,
    ):
        errors.append(
            f"目标日产量为 {target_tons_per_day:g} 吨，"
            f"运行结果为 {operation.ammonia_production_tons:g} 吨"
        )

    return base_report.model_copy(
        update={"is_valid": not errors, "errors": errors}
    )
