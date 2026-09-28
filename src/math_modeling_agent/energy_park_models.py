"""定义能源园区时序数据的结构和输入校验。"""

from math import isclose, isfinite

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


class EnergyParkTechnicalParameters(BaseModel):
    """保存 A 题正文给出的园区装机及额定产能参数。"""

    ordinary_load_peak_mw: float = Field(default=6, gt=0)
    wind_capacity_mw: float = Field(default=40, gt=0)
    pv_capacity_mw: float = Field(default=64, gt=0)
    alkaline_power_mw: float = Field(default=10, gt=0)
    pem_power_mw: float = Field(default=10, gt=0)
    ammonia_power_mw: float = Field(default=0.75, gt=0)
    alkaline_hydrogen_kg_per_hour: float = Field(default=140, gt=0)
    pem_hydrogen_kg_per_hour: float = Field(default=160, gt=0)
    ammonia_tons_per_hour: float = Field(default=1.5, gt=0)
    base_ammonia_tons_per_day: float = Field(default=36, gt=0)


class EnergyParkCostParameters(BaseModel):
    """保存附件 5—8 的成本、电价和储能效率数据。"""

    wind_lcoe_yuan_per_kwh: float = Field(gt=0, allow_inf_nan=False)
    pv_lcoe_yuan_per_kwh: float = Field(gt=0, allow_inf_nan=False)
    alkaline_om_yuan_per_kwh: float = Field(ge=0, allow_inf_nan=False)
    pem_om_yuan_per_kwh: float = Field(ge=0, allow_inf_nan=False)
    alkaline_lifetime_years: int = Field(gt=0)
    pem_lifetime_years: int = Field(gt=0)
    alkaline_efficiency_percent: float = Field(ge=0, le=100)
    pem_efficiency_percent: float = Field(ge=0, le=100)
    hydrogen_energy_kwh_per_kg: float = Field(gt=0, allow_inf_nan=False)
    storage_capex_yuan_per_kwh: float = Field(gt=0, allow_inf_nan=False)
    storage_om_yuan_per_kwh: float = Field(ge=0, allow_inf_nan=False)
    storage_lifetime_years: int = Field(gt=0)
    storage_charge_efficiency_percent: float = Field(gt=0, le=100)
    storage_discharge_efficiency_percent: float = Field(gt=0, le=100)
    storage_self_loss_percent: float = Field(ge=0, lt=100)
    ammonia_capex_yuan_per_kg_h2: float = Field(gt=0, allow_inf_nan=False)
    ammonia_om_yuan_per_kwh: float = Field(ge=0, allow_inf_nan=False)
    ammonia_lifetime_years: int = Field(gt=0)
    ammonia_energy_kwh_per_kg: float = Field(gt=0, allow_inf_nan=False)
    ammonia_hydrogen_kg_per_kg: float = Field(gt=0, allow_inf_nan=False)
    purchase_price_peak_yuan_per_kwh: float = Field(ge=0, allow_inf_nan=False)
    purchase_price_flat_yuan_per_kwh: float = Field(ge=0, allow_inf_nan=False)
    purchase_price_valley_yuan_per_kwh: float = Field(ge=0, allow_inf_nan=False)
    wind_export_price_yuan_per_kwh: float = Field(ge=0, allow_inf_nan=False)
    pv_export_price_yuan_per_kwh: float = Field(ge=0, allow_inf_nan=False)


class EnergyParkDataset(BaseModel):
    """保存题目附件中的典型曲线和风光场景曲线。"""

    ordinary_load: HourlyProfile
    typical_wind: HourlyProfile
    typical_pv: HourlyProfile
    wind_scenarios: list[HourlyProfile] = Field(min_length=6, max_length=6)
    pv_scenarios: list[HourlyProfile] = Field(min_length=4, max_length=4)
    technical: EnergyParkTechnicalParameters = Field(
        default_factory=EnergyParkTechnicalParameters
    )
    costs: EnergyParkCostParameters
    source_files: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def all_curves_use_same_periods(self) -> "EnergyParkDataset":
        """确保附件中的曲线逐小时对齐后再进行功率计算。"""

        expected_periods = self.ordinary_load.periods
        profiles = [self.typical_wind, self.typical_pv]
        profiles.extend(self.wind_scenarios)
        profiles.extend(self.pv_scenarios)
        if any(profile.periods != expected_periods for profile in profiles):
            raise ValueError("所有曲线的小时标签必须完全一致")

        alkaline_output = (
            self.technical.alkaline_power_mw
            * 1000
            * self.costs.alkaline_efficiency_percent
            / 100
            / self.costs.hydrogen_energy_kwh_per_kg
        )
        if not isclose(
            alkaline_output,
            self.technical.alkaline_hydrogen_kg_per_hour,
            rel_tol=0,
            abs_tol=1e-6,
        ):
            raise ValueError("碱性电解槽额定产氢量与功率效率不一致")

        pem_output = (
            self.technical.pem_power_mw
            * 1000
            * self.costs.pem_efficiency_percent
            / 100
            / self.costs.hydrogen_energy_kwh_per_kg
        )
        if not isclose(
            pem_output,
            self.technical.pem_hydrogen_kg_per_hour,
            rel_tol=0,
            abs_tol=1e-6,
        ):
            raise ValueError("质子交换膜电解槽额定产氢量与功率效率不一致")

        hydrogen_demand = (
            self.technical.ammonia_tons_per_hour
            * 1000
            * self.costs.ammonia_hydrogen_kg_per_kg
        )
        hydrogen_output = (
            self.technical.alkaline_hydrogen_kg_per_hour
            + self.technical.pem_hydrogen_kg_per_hour
        )
        if not isclose(hydrogen_output, hydrogen_demand, rel_tol=0, abs_tol=1e-6):
            raise ValueError("额定产氢量与合成氨单位耗氢量不一致")

        ammonia_power = (
            self.technical.ammonia_tons_per_hour
            * 1000
            * self.costs.ammonia_energy_kwh_per_kg
            / 1000
        )
        if not isclose(
            ammonia_power,
            self.technical.ammonia_power_mw,
            rel_tol=0,
            abs_tol=1e-6,
        ):
            raise ValueError("合成氨装置额定功率与单位产品耗电量不一致")
        return self
