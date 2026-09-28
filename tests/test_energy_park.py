import pytest


def _dataset_for_synthetic_typical_day():
    from math_modeling_agent.energy_park_models import (
        EnergyParkCostParameters,
        EnergyParkDataset,
        HourlyProfile,
    )

    periods = [f"{hour}:00-{hour + 1}:00" for hour in range(24)]

    def profile(values):
        return HourlyProfile(periods=periods, values=values)

    costs = EnergyParkCostParameters(
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
        wind_export_price_yuan_per_kwh=0.3779,
        pv_export_price_yuan_per_kwh=0.3779,
    )
    return EnergyParkDataset(
        ordinary_load=profile([1 / 6] * 24),
        typical_wind=profile([0.0] * 12 + [0.5] * 12),
        typical_pv=profile([0.0] * 12 + [0.25] * 12),
        wind_scenarios=[profile([0.5] * 24) for _ in range(6)],
        pv_scenarios=[profile([0.25] * 24) for _ in range(4)],
        costs=costs,
        source_files=["附件1.xlsx", "附件2.xlsx"],
    )


def test_typical_day_balances_buy_and_export_power() -> None:
    from math_modeling_agent.energy_park import compute_typical_day

    result = compute_typical_day(_dataset_for_synthetic_typical_day())

    assert result.total_load_mwh == pytest.approx(522)
    assert result.renewable_generation_mwh == pytest.approx(432)
    assert result.grid_purchase_mwh == pytest.approx(261)
    assert result.grid_export_mwh == pytest.approx(171)
    assert result.source_files == ["附件1.xlsx", "附件2.xlsx"]
    assert max(abs(item.power_balance_residual_mw) for item in result.hourly_balances) == pytest.approx(0)


def test_typical_day_reports_contest_and_physical_self_use_ratios() -> None:
    from math_modeling_agent.energy_park import compute_typical_day

    result = compute_typical_day(_dataset_for_synthetic_typical_day())

    assert result.green_metrics.contest_formula_self_use_ratio == pytest.approx(90 / 432)
    assert result.green_metrics.physical_self_use_ratio == pytest.approx(261 / 432)
    assert result.green_metrics.green_share_of_load_ratio == pytest.approx(0.5)
    assert result.green_metrics.renewable_export_ratio == pytest.approx(171 / 432)
    assert result.green_metrics.contest_formula_self_use_passes is False
    assert result.green_metrics.physical_self_use_passes is True


def test_typical_day_returns_cost_components_and_explicit_capex_basis() -> None:
    from math_modeling_agent.energy_park import compute_typical_day

    result = compute_typical_day(_dataset_for_synthetic_typical_day())

    assert result.costs.variable_operating_cost_yuan == pytest.approx(
        result.costs.renewable_generation_cost_yuan
        + result.costs.grid_purchase_cost_yuan
        - result.costs.export_revenue_yuan
        + result.costs.alkaline_om_yuan
        + result.costs.pem_om_yuan
        + result.costs.ammonia_om_yuan
    )
    assert "铭牌氢处理能力" in result.costs.ammonia_capex_assumption


def test_validator_detects_tampered_power_balance() -> None:
    from math_modeling_agent.energy_park import compute_typical_day
    from math_modeling_agent.energy_park_validator import validate_typical_day

    data = _dataset_for_synthetic_typical_day()
    result = compute_typical_day(data)
    first_hour = result.hourly_balances[0].model_copy(
        update={"grid_export_mw": 0.1}
    )
    tampered_result = result.model_copy(
        update={"hourly_balances": [first_hour, *result.hourly_balances[1:]]}
    )

    report = validate_typical_day(data, tampered_result)

    assert report.is_valid is False
    assert any("0:00-1:00" in error for error in report.errors), report.errors


def test_validator_detects_tampered_metric_status() -> None:
    from math_modeling_agent.energy_park import compute_typical_day
    from math_modeling_agent.energy_park_validator import validate_typical_day

    data = _dataset_for_synthetic_typical_day()
    result = compute_typical_day(data)
    altered_metrics = result.green_metrics.model_copy(
        update={"physical_self_use_passes": False}
    )
    tampered_result = result.model_copy(update={"green_metrics": altered_metrics})

    report = validate_typical_day(data, tampered_result)

    assert report.is_valid is False
    assert any("physical_self_use_passes" in error for error in report.errors)


def test_validator_detects_changed_source_file_list() -> None:
    from math_modeling_agent.energy_park import compute_typical_day
    from math_modeling_agent.energy_park_validator import validate_typical_day

    data = _dataset_for_synthetic_typical_day()
    result = compute_typical_day(data)
    tampered_result = result.model_copy(update={"source_files": ["未核实.xlsx"]})

    report = validate_typical_day(data, tampered_result)

    assert report.is_valid is False
    assert any("来源文件" in error for error in report.errors)
