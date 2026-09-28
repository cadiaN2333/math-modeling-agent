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
    periods = [f"{hour}:00-{hour + 1}:00" for hour in range(24)]
    profile = {"periods": periods, "values": [0.5] * 24}
    return {
        "ordinary_load": profile,
        "typical_wind": profile,
        "typical_pv": profile,
        "wind_scenarios": [profile.copy() for _ in range(6)],
        "pv_scenarios": [profile.copy() for _ in range(4)],
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
