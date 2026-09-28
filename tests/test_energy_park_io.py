import pytest


def _valid_hourly_rows() -> list[tuple[str, object]]:
    periods = [f"{hour}:00-{hour + 1}:00" for hour in range(24)]
    return [("时段", "标幺值"), *[(period, 0.5) for period in periods]]


def test_parse_hourly_table_preserves_period_order() -> None:
    from math_modeling_agent.energy_park_io import parse_hourly_table

    rows = _valid_hourly_rows()
    profile = parse_hourly_table(rows, source_name="附件1")

    assert profile.periods == [row[0] for row in rows[1:]]
    assert profile.values == [0.5] * 24


def test_parse_hourly_table_names_missing_source_value() -> None:
    from math_modeling_agent.energy_park_io import parse_hourly_table

    rows = _valid_hourly_rows()
    rows[5] = (rows[5][0], None)

    with pytest.raises(ValueError, match="附件1.*6"):
        parse_hourly_table(rows, source_name="附件1")


def test_parse_hourly_table_reads_selected_numeric_column() -> None:
    from math_modeling_agent.energy_park_io import parse_hourly_table

    rows = [("时段", "风电", "光伏")]
    rows.extend(
        (f"{hour}:00-{hour + 1}:00", 0.25, 0.75)
        for hour in range(24)
    )

    pv_profile = parse_hourly_table(
        rows,
        source_name="附件2",
        value_column=2,
    )

    assert pv_profile.values == [0.75] * 24
