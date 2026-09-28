import pytest


def _dataset_with_sunny_half_day():
    from math_modeling_agent.energy_park_models import (
        EnergyParkCostParameters,
        EnergyParkDataset,
        HourlyProfile,
    )

    periods = [f"{hour}:00-{hour + 1}:00" for hour in range(24)]

    def profile(values):
        return HourlyProfile(periods=periods, values=values)

    # 中午光伏恰好覆盖满负荷，避免把高价余电售出比自用更划算。
    solar_profile = profile([0.0] * 12 + [42.5 / 64] * 12)
    return EnergyParkDataset(
        ordinary_load=profile([0.0] * 24),
        typical_wind=profile([0.0] * 24),
        typical_pv=solar_profile,
        wind_scenarios=[profile([0.0] * 24) for _ in range(6)],
        pv_scenarios=[solar_profile for _ in range(4)],
        costs=EnergyParkCostParameters(
            wind_lcoe_yuan_per_kwh=0.15,
            pv_lcoe_yuan_per_kwh=0.12,
            alkaline_om_yuan_per_kwh=0.1,
            pem_om_yuan_per_kwh=0.15,
            alkaline_lifetime_years=30,
            pem_lifetime_years=30,
            alkaline_efficiency_percent=70,
            pem_efficiency_percent=80,
            hydrogen_energy_kwh_per_kg=50,
            storage_capex_yuan_per_kwh=1000,
            storage_om_yuan_per_kwh=0.01,
            storage_lifetime_years=15,
            storage_charge_efficiency_percent=90,
            storage_discharge_efficiency_percent=90,
            storage_self_loss_percent=0.2,
            ammonia_capex_yuan_per_kg_h2=60000,
            ammonia_om_yuan_per_kwh=0.002,
            ammonia_lifetime_years=30,
            ammonia_energy_kwh_per_kg=0.5,
            ammonia_hydrogen_kg_per_kg=0.2,
            purchase_price_peak_yuan_per_kwh=0.8024,
            purchase_price_flat_yuan_per_kwh=0.6074,
            purchase_price_valley_yuan_per_kwh=0.3424,
            wind_export_price_yuan_per_kwh=0,
            pv_export_price_yuan_per_kwh=0,
        ),
    )


def test_discrete_schedule_meets_target_and_uses_sunny_hours() -> None:
    from math_modeling_agent.energy_park_discrete import solve_discrete_schedule

    dataset = _dataset_with_sunny_half_day()
    run = solve_discrete_schedule(
        dataset,
        target_tons_per_day=36,
        scenario_id="sunny-half-day",
    )

    assert run.solver_status == "OPTIMAL"
    assert sum(run.hourly_on) == 12
    assert run.operation.ammonia_production_tons == pytest.approx(36)
    assert run.hourly_on[:12] == [False] * 12
    assert run.hourly_on[12:] == [True] * 12
    assert run.validation_report.is_valid is True


def test_question_two_runs_five_targets_for_24_scenarios_and_360_days() -> None:
    from math_modeling_agent.energy_park_discrete import run_discrete_question_two

    result = run_discrete_question_two(_dataset_with_sunny_half_day())

    assert result.target_levels_tons_per_day == [72, 63, 54, 45, 36]
    assert len(result.typical_runs) == 5
    assert len(result.scenario_runs) == 24 * 5
    assert len({run.scenario_id for run in result.scenario_runs}) == 24
    assert len(result.annual_summaries) == 5
    assert any("事后评价" in item for item in result.modeling_assumptions)
    for summary in result.annual_summaries:
        assert summary.representative_year_days == 360
        assert sum(summary.green_status_days_contest_formula.values()) == 360
        assert sum(summary.green_status_days_physical_self_use.values()) == 360
        assert summary.annual_production_tons == pytest.approx(
            summary.target_tons_per_day * 360
        )


def test_discrete_schedule_rejects_target_not_representable_by_full_load_hours() -> None:
    from math_modeling_agent.energy_park_discrete import solve_discrete_schedule

    with pytest.raises(ValueError, match="整数个额定开机小时"):
        solve_discrete_schedule(
            _dataset_with_sunny_half_day(),
            target_tons_per_day=37,
            scenario_id="invalid-target",
        )


def test_discrete_validator_rejects_schedule_target_mismatch() -> None:
    from math_modeling_agent.energy_park import compute_energy_schedule
    from math_modeling_agent.energy_park_discrete_validator import (
        validate_discrete_schedule,
    )

    dataset = _dataset_with_sunny_half_day()
    hourly_on = [False] * 12 + [True] * 12
    operation = compute_energy_schedule(
        dataset,
        [float(value) for value in hourly_on],
        capacity_scale=2.0,
    )

    report = validate_discrete_schedule(
        dataset,
        target_tons_per_day=45,
        hourly_on=hourly_on,
        operation=operation,
        capacity_scale=2.0,
    )

    assert report.is_valid is False
    assert any("目标日产量" in error for error in report.errors)
