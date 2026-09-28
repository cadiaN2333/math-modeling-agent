"""定义能源园区时序数据的结构和输入校验。"""

from math import isfinite

from pydantic import BaseModel, Field, field_validator, model_validator


class HourlyProfile(BaseModel):
    """保存一天 24 个小时段的标幺功率曲线。"""

    periods: list[str] = Field(min_length=24, max_length=24)
    values: list[float] = Field(min_length=24, max_length=24)

    @field_validator("periods")
    @classmethod
    def periods_must_be_nonempty(cls, periods: list[str]) -> list[str]:
        if any(not period.strip() for period in periods):
            raise ValueError("时段标签不能为空")
        return periods

    @field_validator("values")
    @classmethod
    def values_must_be_valid_per_unit(cls, values: list[float]) -> list[float]:
        if any(not isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("标幺功率必须是 0 到 1 之间的有限数值")
        return values

    @model_validator(mode="after")
    def periods_must_be_unique(self) -> "HourlyProfile":
        if len(self.periods) != len(set(self.periods)):
            raise ValueError("时段标签不能重复")
        return self


class EnergyParkDataset(BaseModel):
    """保存题目附件中的典型曲线和风光场景曲线。"""

    ordinary_load: HourlyProfile
    typical_wind: HourlyProfile
    typical_pv: HourlyProfile
    wind_scenarios: list[HourlyProfile] = Field(min_length=6, max_length=6)
    pv_scenarios: list[HourlyProfile] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def all_curves_use_same_periods(self) -> "EnergyParkDataset":
        """确保附件中的曲线逐小时对齐后再进行功率计算。"""

        expected_periods = self.ordinary_load.periods
        profiles = [self.typical_wind, self.typical_pv]
        profiles.extend(self.wind_scenarios)
        profiles.extend(self.pv_scenarios)
        if any(profile.periods != expected_periods for profile in profiles):
            raise ValueError("所有曲线的小时标签必须完全一致")
        return self
