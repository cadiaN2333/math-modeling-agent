import pytest
from pydantic import ValidationError


def test_hourly_profile_rejects_23_values() -> None:
    from math_modeling_agent.energy_park_models import HourlyProfile

    periods = [f"{hour}:00-{hour + 1}:00" for hour in range(24)]

    with pytest.raises(ValidationError):
        HourlyProfile(periods=periods, values=[0.5] * 23)


def test_hourly_profile_rejects_duplicate_periods() -> None:
    from math_modeling_agent.energy_park_models import HourlyProfile

    with pytest.raises(ValidationError, match="时段标签不能重复"):
        HourlyProfile(periods=["00:00-01:00"] * 24, values=[0.5] * 24)


@pytest.mark.parametrize("invalid_value", [-0.01, 1.01, float("nan")])
def test_hourly_profile_rejects_invalid_per_unit_values(
    invalid_value: float,
) -> None:
    from math_modeling_agent.energy_park_models import HourlyProfile

    periods = [f"{hour}:00-{hour + 1}:00" for hour in range(24)]
    values = [0.5] * 24
    values[0] = invalid_value

    with pytest.raises(ValidationError):
        HourlyProfile(periods=periods, values=values)


def _valid_dataset_data() -> dict[str, object]:
    from math_modeling_agent.energy_park_models import EnergyParkCostParameters

    periods = [f"{hour}:00-{hour + 1}:00" for hour in range(24)]
    profile = {"periods": periods, "values": [0.5] * 24}
    return {
        "ordinary_load": profile,
        "typical_wind": profile,
        "typical_pv": profile,
        "wind_scenarios": [profile.copy() for _ in range(6)],
        "pv_scenarios": [profile.copy() for _ in range(4)],
        "costs": EnergyParkCostParameters(
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
        ),
    }


def test_energy_park_dataset_requires_six_wind_and_four_pv_scenarios() -> None:
    from math_modeling_agent.energy_park_models import EnergyParkDataset

    data = _valid_dataset_data()
    data["wind_scenarios"] = data["wind_scenarios"][:5]

    with pytest.raises(ValidationError):
        EnergyParkDataset.model_validate(data)


def test_energy_park_dataset_requires_identical_hour_labels() -> None:
    from math_modeling_agent.energy_park_models import EnergyParkDataset

    data = _valid_dataset_data()
    solar = data["typical_pv"]
    assert isinstance(solar, dict)
    solar["periods"] = ["错位时段", *solar["periods"][1:]]

    with pytest.raises(ValidationError, match="所有曲线的小时标签必须完全一致"):
        EnergyParkDataset.model_validate(data)


def test_energy_park_costs_reject_efficiency_over_100_percent() -> None:
    from math_modeling_agent.energy_park_models import EnergyParkCostParameters

    data = _valid_dataset_data()["costs"].model_dump()
    data["storage_charge_efficiency_percent"] = 101

    with pytest.raises(ValidationError):
        EnergyParkCostParameters.model_validate(data)


def test_energy_park_dataset_rejects_inconsistent_hydrogen_capacity() -> None:
    from math_modeling_agent.energy_park_models import (
        EnergyParkDataset,
        EnergyParkTechnicalParameters,
    )

    data = _valid_dataset_data()
    data["technical"] = EnergyParkTechnicalParameters(
        alkaline_hydrogen_kg_per_hour=141
    )

    with pytest.raises(ValidationError, match="碱性电解槽额定产氢量与功率效率不一致"):
        EnergyParkDataset.model_validate(data)
