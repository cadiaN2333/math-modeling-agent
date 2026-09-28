"""读取并校验能源园区问题的 Excel 曲线数据。"""

from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from .energy_park_models import HourlyProfile


def parse_hourly_table(
    rows: Sequence[Sequence[Any]],
    *,
    source_name: str,
    value_column: int = 1,
) -> HourlyProfile:
    """从含表头的矩阵读取一条 24 小时标幺功率曲线。"""

    if not rows:
        raise ValueError(f"{source_name} 工作表为空")
    if value_column <= 0:
        raise ValueError("功率数据列必须位于时段列之后")
    if len(rows) != 25:
        raise ValueError(f"{source_name} 应有 1 行表头和 24 行小时数据")

    periods: list[str] = []
    values: list[float] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) <= value_column:
            raise ValueError(
                f"{source_name} 第 {row_number} 行缺少第 {value_column + 1} 列数据"
            )
        period, raw_value = row[0], row[value_column]
        if period is None or raw_value is None:
            raise ValueError(f"{source_name} 第 {row_number} 行存在空时段或空数值")
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source_name} 第 {row_number} 行的功率值不是数值"
            ) from exc
        periods.append(str(period).strip())
        values.append(numeric_value)

    try:
        return HourlyProfile(periods=periods, values=values)
    except ValidationError as exc:
        raise ValueError(f"{source_name} 曲线数据校验失败：{exc}") from exc
