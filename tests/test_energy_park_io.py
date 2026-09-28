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


def test_parse_hourly_table_checks_value_header_when_requested() -> None:
    from math_modeling_agent.energy_park_io import parse_hourly_table

    with pytest.raises(ValueError, match="附件2.*风电标幺值"):
        parse_hourly_table(
            _valid_hourly_rows(),
            source_name="附件2",
            expected_value_header="风电标幺值",
        )


def _hourly_sheet(*headings: str) -> list[tuple[object, ...]]:
    periods = [f"{hour}:00-{hour + 1}:00" for hour in range(24)]
    rows = [("时段", *headings)]
    rows.extend((period, *([0.5] * len(headings))) for period in periods)
    return rows


def _attachment_tables() -> dict[int, list[tuple[object, ...]]]:
    return {
        1: _hourly_sheet("典型日常规电负荷标幺功率"),
        2: _hourly_sheet("风电标幺值", "光伏标幺值"),
        3: _hourly_sheet(*(f"风电场景{i}" for i in range(1, 7))),
        4: _hourly_sheet(*(f"光伏场景{i}" for i in range(1, 5))),
        5: [
            (None, "风机", "光伏", "碱性电解槽", "质子交换膜电解槽"),
            ("度电成本（元/kWh）", 0.15, 0.12, "—", "—"),
            ("运维系数(元/kWh)", "—", "—", 0.1, 0.15),
            ("使用寿命(年)", 25, 25, 30, 30),
            ("效率(%)", "—", "—", 70, 80),
            ("制氢耗电量", "—", "—", "50kWh/kg", "50kWh/kg"),
        ],
        6: [
            (None, "电储能", "合成氨装置"),
            ("投资成本", "1000 元/kWh", "60000 元/kgH2"),
            ("运维系数(元/kWh)", 0.01, 0.002),
            ("使用寿命(年)", 15, 30),
            ("效率(%)", "90（充电）/90（放电）", "—"),
            ("自损耗率(%)", 0.2, 0),
            ("用电/用氢需求", "—", "0.5 kWh/kgNH3和0.2kgH2/kgNH3"),
        ],
        7: [
            ("时段", "分时电价（元/kWh）"),
            ("高峰", 0.8024),
            ("平时", 0.6074),
            ("低谷", 0.3424),
        ],
        8: [
            ("电源类型", "上网电价（元/kWh）"),
            ("风电", 0.3779),
            ("光伏", 0.3779),
        ],
    }


def test_load_energy_park_directory_maps_all_eight_attachments(
    tmp_path,
    monkeypatch,
) -> None:
    from math_modeling_agent import energy_park_io

    names = [
        "附件1：园区典型日常规电负荷标幺功率曲线.xlsx",
        "附件2：典型日风电、光伏标幺功率表.xlsx",
        "附件3：园区6种场景的风电标幺功率表.xlsx",
        "附件4：园区4种场景的光伏标幺功率表.xlsx",
        "附件5：风光发电与制氢设备技术参数.xlsx",
        "附件6：储能设备和合成氨装置技术参数.xlsx",
        "附件7：分时电价表.xlsx",
        "附件8：风电、光伏余电上网电价.xlsx",
    ]
    for name in names:
        (tmp_path / name).touch()

    tables = _attachment_tables()

    def read_rows(path):
        attachment_number = int(path.name[2])
        return tables[attachment_number]

    monkeypatch.setattr(energy_park_io, "_read_workbook_rows", read_rows)
    dataset = energy_park_io.load_energy_park_directory(tmp_path)

    assert len(dataset.ordinary_load.values) == 24
    assert len(dataset.wind_scenarios) == 6
    assert len(dataset.pv_scenarios) == 4
    assert dataset.costs.wind_lcoe_yuan_per_kwh == pytest.approx(0.15)
    assert dataset.costs.purchase_price_peak_yuan_per_kwh == pytest.approx(0.8024)
    assert dataset.costs.ammonia_hydrogen_kg_per_kg == pytest.approx(0.2)
    assert dataset.source_files == names


def test_load_energy_park_directory_reports_missing_attachments(tmp_path) -> None:
    from math_modeling_agent.energy_park_io import load_energy_park_directory

    with pytest.raises(ValueError, match="附件1"):
        load_energy_park_directory(tmp_path)


def test_load_energy_park_directory_rejects_unknown_tariff_label(
    tmp_path,
    monkeypatch,
) -> None:
    from math_modeling_agent import energy_park_io

    names = [
        "附件1：园区典型日常规电负荷标幺功率曲线.xlsx",
        "附件2：典型日风电、光伏标幺功率表.xlsx",
        "附件3：园区6种场景的风电标幺功率表.xlsx",
        "附件4：园区4种场景的光伏标幺功率表.xlsx",
        "附件5：风光发电与制氢设备技术参数.xlsx",
        "附件6：储能设备和合成氨装置技术参数.xlsx",
        "附件7：分时电价表.xlsx",
        "附件8：风电、光伏余电上网电价.xlsx",
    ]
    for name in names:
        (tmp_path / name).touch()
    tables = _attachment_tables()
    tables[7][1] = ("未知时段", 0.8024)
    monkeypatch.setattr(
        energy_park_io,
        "_read_workbook_rows",
        lambda path: tables[int(path.name[2])],
    )

    with pytest.raises(ValueError, match="附件7.*高峰"):
        energy_park_io.load_energy_park_directory(tmp_path)
