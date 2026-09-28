"""计算电工杯 A 题问题一的典型日基准。"""

import re
from math import isfinite
from typing import Literal, Sequence

from pydantic import BaseModel, Field, field_validator

from .energy_park_models import EnergyParkDataset, HourlyProfile


class HourlyEnergyBalance(BaseModel):
    """保存一个小时的负荷、发电、购售电及平衡残差。"""

    period: str
    ordinary_load_mw: float
    process_load_mw: float
    total_load_mw: float
    wind_generation_mw: float
    pv_generation_mw: float
    renewable_generation_mw: float
    grid_purchase_mw: float
    grid_export_mw: float
    power_balance_residual_mw: float


class GreenMetrics(BaseModel):
    """同时保留赛题字面公式和物理自发自用口径。"""

    contest_formula_self_use_ratio: float | None
    physical_self_use_ratio: float | None
    green_share_of_load_ratio: float | None
    renewable_export_ratio: float | None
    contest_formula_self_use_passes: bool | None
    physical_self_use_passes: bool | None
    green_share_of_load_passes: bool | None
    renewable_export_passes: bool | None


class CostSensitivityItem(BaseModel):
    """描述一个成本参数变化对吨氨成本的局部影响。"""

    component_id: str
    component_name: str
    base_parameter_value: float
    parameter_unit: str
    amount_type: Literal["费用", "收入"]
    base_amount_yuan: float = Field(ge=0)
    variable_cost_delta_yuan_per_ton_for_one_percent_increase: float
    capex_included_cost_delta_yuan_per_ton_for_one_percent_increase: float


class TypicalDayCosts(BaseModel):
    """保存逐项成本，避免把假设不同的成本压成一个数字。"""

    wind_generation_cost_yuan: float
    pv_generation_cost_yuan: float
    renewable_generation_cost_yuan: float
    grid_purchase_cost_yuan: float
    wind_export_revenue_yuan: float
    pv_export_revenue_yuan: float
    export_revenue_yuan: float
    alkaline_om_yuan: float
    pem_om_yuan: float
    ammonia_om_yuan: float
    variable_operating_cost_yuan: float
    variable_cost_per_ton_yuan: float
    ammonia_capex_assumption: str
    ammonia_capex_daily_yuan: float
    cost_including_ammonia_capex_yuan: float
    cost_including_ammonia_capex_per_ton_yuan: float
    excluded_cost_items: list[str]
    sensitivity_assumption: str | None = None
    sensitivity_analysis: list[CostSensitivityItem] = Field(default_factory=list)


class TypicalDayResult(BaseModel):
    """逐时运行计划的功率平衡、绿电指标和成本分解结果。"""

    modeling_idea: str
    capacity_scale: float = Field(gt=0)
    process_fractions: list[float] = Field(min_length=24, max_length=24)
    ammonia_production_tons: float = Field(gt=0)
    hourly_balances: list[HourlyEnergyBalance] = Field(min_length=24, max_length=24)
    total_load_mwh: float
    wind_generation_mwh: float
    pv_generation_mwh: float
    renewable_generation_mwh: float
    grid_purchase_mwh: float
    grid_export_mwh: float
    green_metrics: GreenMetrics
    costs: TypicalDayCosts
    review_items: list[str]
    source_files: list[str]

    @field_validator("process_fractions")
    @classmethod
    def process_fractions_must_be_bounded(
        cls,
        fractions: list[float],
    ) -> list[float]:
        if any(not isfinite(value) or not 0 <= value <= 1 for value in fractions):
            raise ValueError("逐时运行比例必须是 0 到 1 之间的有限数值")
        return fractions


def _start_hour(period: str) -> int:
    """从题目小时标签提取时段起始小时。"""

    match = re.match(r"^\s*(\d{1,2}):\d{2}\s*-", period)
    if match is None:
        raise ValueError(f"无法识别小时标签：{period}")
    hour = int(match.group(1))
    if not 0 <= hour <= 23:
        raise ValueError(f"小时标签超出 0—23 点范围：{period}")
    return hour


def _purchase_price_yuan_per_kwh(data: EnergyParkDataset, hour: int) -> float:
    """按附件 7 的半开区间分配峰、平、谷电价。"""

    tier = _purchase_tier(hour)
    prices = {
        "peak": data.costs.purchase_price_peak_yuan_per_kwh,
        "flat": data.costs.purchase_price_flat_yuan_per_kwh,
        "valley": data.costs.purchase_price_valley_yuan_per_kwh,
    }
    return prices[tier]


def _purchase_tier(hour: int) -> Literal["peak", "flat", "valley"]:
    """返回某一时段所属的分时电价类别。"""

    if 10 <= hour < 15 or 18 <= hour < 21:
        return "peak"
    if 7 <= hour < 10 or 15 <= hour < 18 or 21 <= hour < 23:
        return "flat"
    return "valley"


def _ratio(numerator: float, denominator: float) -> float | None:
    """分母为零时保留不可计算状态。"""

    if denominator == 0:
        return None
    return numerator / denominator


def _cost_sensitivity_analysis(
    production_tons: float,
    components: Sequence[
        tuple[str, str, float, str, Literal["费用", "收入"], float, bool]
    ],
) -> list[CostSensitivityItem]:
    """计算各参数上调 1% 时固定调度下的精确吨成本变化。"""

    sensitivity: list[CostSensitivityItem] = []
    for (
        component_id,
        name,
        parameter_value,
        parameter_unit,
        amount_type,
        amount_yuan,
        is_capex,
    ) in components:
        direction = -1.0 if amount_type == "收入" else 1.0
        delta_per_ton = direction * amount_yuan * 0.01 / production_tons
        sensitivity.append(
            CostSensitivityItem(
                component_id=component_id,
                component_name=name,
                base_parameter_value=parameter_value,
                parameter_unit=parameter_unit,
                amount_type=amount_type,
                base_amount_yuan=amount_yuan,
                variable_cost_delta_yuan_per_ton_for_one_percent_increase=(
                    0.0 if is_capex else delta_per_ton
                ),
                capex_included_cost_delta_yuan_per_ton_for_one_percent_increase=(
                    delta_per_ton
                ),
            )
        )
    return sensitivity


def compute_energy_schedule(
    data: EnergyParkDataset,
    process_fractions: Sequence[float],
    *,
    capacity_scale: float = 1.0,
    wind_profile: HourlyProfile | None = None,
    pv_profile: HourlyProfile | None = None,
    include_cost_sensitivity: bool = True,
) -> TypicalDayResult:
    """按给定的逐时运行比例和产能倍数计算功率平衡与成本。"""

    parameters = data.technical
    if len(process_fractions) != 24:
        raise ValueError("逐时运行比例必须覆盖 24 个时段")
    fractions = [float(value) for value in process_fractions]
    if any(not isfinite(value) or not 0 <= value <= 1 for value in fractions):
        raise ValueError("逐时运行比例必须是 0 到 1 之间的有限数值")
    if not isfinite(capacity_scale) or capacity_scale <= 0:
        raise ValueError("装置产能倍数必须是正的有限数值")
    if sum(fractions) <= 0:
        raise ValueError("日产氨量为零，无法计算吨氨成本")

    wind_profile = wind_profile or data.typical_wind
    pv_profile = pv_profile or data.typical_pv
    if (
        wind_profile.periods != data.ordinary_load.periods
        or pv_profile.periods != data.ordinary_load.periods
    ):
        raise ValueError("负荷、风电和光伏曲线的小时标签必须完全一致")

    hourly: list[HourlyEnergyBalance] = []
    wind_export_mwh = 0.0
    pv_export_mwh = 0.0
    grid_purchase_cost_yuan = 0.0
    grid_purchase_kwh_by_tier = {"peak": 0.0, "flat": 0.0, "valley": 0.0}
    alkaline_om_yuan = 0.0
    pem_om_yuan = 0.0
    ammonia_om_yuan = 0.0
    for index, period in enumerate(data.ordinary_load.periods):
        fraction = fractions[index]
        ordinary_load_mw = (
            parameters.ordinary_load_peak_mw
            * data.ordinary_load.values[index]
        )
        wind_generation_mw = (
            parameters.wind_capacity_mw * wind_profile.values[index]
        )
        pv_generation_mw = parameters.pv_capacity_mw * pv_profile.values[index]
        renewable_generation_mw = wind_generation_mw + pv_generation_mw
        process_load_mw = (
            parameters.alkaline_power_mw
            + parameters.pem_power_mw
            + parameters.ammonia_power_mw
        ) * capacity_scale * fraction
        total_load_mw = ordinary_load_mw + process_load_mw
        grid_purchase_mw = max(total_load_mw - renewable_generation_mw, 0.0)
        grid_export_mw = max(renewable_generation_mw - total_load_mw, 0.0)
        residual_mw = (
            renewable_generation_mw
            + grid_purchase_mw
            - total_load_mw
            - grid_export_mw
        )

        if renewable_generation_mw > 0:
            wind_export_mwh += grid_export_mw * wind_generation_mw / renewable_generation_mw
            pv_export_mwh += grid_export_mw * pv_generation_mw / renewable_generation_mw

        hour = _start_hour(period)
        purchase_tier = _purchase_tier(hour)
        purchased_kwh = grid_purchase_mw * 1000
        grid_purchase_kwh_by_tier[purchase_tier] += purchased_kwh
        grid_purchase_cost_yuan += purchased_kwh * _purchase_price_yuan_per_kwh(
            data,
            hour,
        )
        alkaline_om_yuan += (
            parameters.alkaline_power_mw
            * capacity_scale
            * fraction
            * 1000
            * data.costs.alkaline_om_yuan_per_kwh
        )
        pem_om_yuan += (
            parameters.pem_power_mw
            * capacity_scale
            * fraction
            * 1000
            * data.costs.pem_om_yuan_per_kwh
        )
        ammonia_om_yuan += (
            parameters.ammonia_power_mw
            * capacity_scale
            * fraction
            * 1000
            * data.costs.ammonia_om_yuan_per_kwh
        )
        hourly.append(
            HourlyEnergyBalance(
                period=period,
                ordinary_load_mw=ordinary_load_mw,
                process_load_mw=process_load_mw,
                total_load_mw=total_load_mw,
                wind_generation_mw=wind_generation_mw,
                pv_generation_mw=pv_generation_mw,
                renewable_generation_mw=renewable_generation_mw,
                grid_purchase_mw=grid_purchase_mw,
                grid_export_mw=grid_export_mw,
                power_balance_residual_mw=residual_mw,
            )
        )

    # 分析间隔为一小时，因此 24 个功率值求和即为 MWh。
    total_load_mwh = sum(item.total_load_mw for item in hourly)
    wind_generation_mwh = sum(item.wind_generation_mw for item in hourly)
    pv_generation_mwh = sum(item.pv_generation_mw for item in hourly)
    renewable_generation_mwh = wind_generation_mwh + pv_generation_mwh
    grid_purchase_mwh = sum(item.grid_purchase_mw for item in hourly)
    grid_export_mwh = sum(item.grid_export_mw for item in hourly)

    contest_ratio = _ratio(
        total_load_mwh - grid_export_mwh - grid_purchase_mwh,
        renewable_generation_mwh,
    )
    physical_self_use_ratio = _ratio(
        renewable_generation_mwh - grid_export_mwh,
        renewable_generation_mwh,
    )
    green_share_ratio = _ratio(
        renewable_generation_mwh - grid_export_mwh,
        total_load_mwh,
    )
    export_ratio = _ratio(grid_export_mwh, renewable_generation_mwh)
    green_metrics = GreenMetrics(
        contest_formula_self_use_ratio=contest_ratio,
        physical_self_use_ratio=physical_self_use_ratio,
        green_share_of_load_ratio=green_share_ratio,
        renewable_export_ratio=export_ratio,
        contest_formula_self_use_passes=(
            contest_ratio > 0.60 if contest_ratio is not None else None
        ),
        physical_self_use_passes=(
            physical_self_use_ratio > 0.60
            if physical_self_use_ratio is not None
            else None
        ),
        green_share_of_load_passes=(
            green_share_ratio > 0.30 if green_share_ratio is not None else None
        ),
        renewable_export_passes=(
            export_ratio < 0.20 if export_ratio is not None else None
        ),
    )

    costs = data.costs
    wind_generation_cost_yuan = wind_generation_mwh * 1000 * costs.wind_lcoe_yuan_per_kwh
    pv_generation_cost_yuan = pv_generation_mwh * 1000 * costs.pv_lcoe_yuan_per_kwh
    renewable_generation_cost_yuan = wind_generation_cost_yuan + pv_generation_cost_yuan
    wind_export_revenue_yuan = wind_export_mwh * 1000 * costs.wind_export_price_yuan_per_kwh
    pv_export_revenue_yuan = pv_export_mwh * 1000 * costs.pv_export_price_yuan_per_kwh
    export_revenue_yuan = wind_export_revenue_yuan + pv_export_revenue_yuan
    variable_operating_cost_yuan = (
        renewable_generation_cost_yuan
        + grid_purchase_cost_yuan
        - export_revenue_yuan
        + alkaline_om_yuan
        + pem_om_yuan
        + ammonia_om_yuan
    )
    production_tons = (
        parameters.ammonia_tons_per_hour * capacity_scale * sum(fractions)
    )
    h2_processing_capacity_kg_per_hour = (
        parameters.alkaline_hydrogen_kg_per_hour
        + parameters.pem_hydrogen_kg_per_hour
    ) * capacity_scale
    ammonia_capex_daily_yuan = (
        costs.ammonia_capex_yuan_per_kg_h2
        * h2_processing_capacity_kg_per_hour
        / costs.ammonia_lifetime_years
        / 365
    )
    ammonia_capex_assumption = (
        f"将附件给出的 {costs.ammonia_capex_yuan_per_kg_h2:g} 元/kgH₂ "
        "解释为每 1 kgH₂/h 铭牌氢处理能力的投资，"
        "按合成氨装置寿命直线摊销至每日；此口径需与参考论文核对。"
    )
    costs_result = TypicalDayCosts(
        wind_generation_cost_yuan=wind_generation_cost_yuan,
        pv_generation_cost_yuan=pv_generation_cost_yuan,
        renewable_generation_cost_yuan=renewable_generation_cost_yuan,
        grid_purchase_cost_yuan=grid_purchase_cost_yuan,
        wind_export_revenue_yuan=wind_export_revenue_yuan,
        pv_export_revenue_yuan=pv_export_revenue_yuan,
        export_revenue_yuan=export_revenue_yuan,
        alkaline_om_yuan=alkaline_om_yuan,
        pem_om_yuan=pem_om_yuan,
        ammonia_om_yuan=ammonia_om_yuan,
        variable_operating_cost_yuan=variable_operating_cost_yuan,
        variable_cost_per_ton_yuan=variable_operating_cost_yuan / production_tons,
        ammonia_capex_assumption=ammonia_capex_assumption,
        ammonia_capex_daily_yuan=ammonia_capex_daily_yuan,
        cost_including_ammonia_capex_yuan=(
            variable_operating_cost_yuan + ammonia_capex_daily_yuan
        ),
        cost_including_ammonia_capex_per_ton_yuan=(
            variable_operating_cost_yuan + ammonia_capex_daily_yuan
        ) / production_tons,
        excluded_cost_items=[
            "附件未提供 ALKEL 与 PEM 电解槽资本投资额",
            "题目未提供输配电费、线损和税费参数",
        ],
        sensitivity_assumption=(
            (
                "分别将每个成本或收入参数相对上调 1%，保持当前运行计划、购售电量及产量不变；"
                "该敏感度是固定调度下的成本核算影响，不代表重新优化后的结果。"
            )
            if include_cost_sensitivity
            else None
        ),
        sensitivity_analysis=(
            _cost_sensitivity_analysis(
                production_tons,
                [
                    (
                        "wind_lcoe",
                        "风电度电成本",
                        costs.wind_lcoe_yuan_per_kwh,
                        "元/kWh",
                        "费用",
                        wind_generation_cost_yuan,
                        False,
                    ),
                    (
                        "pv_lcoe",
                        "光伏度电成本",
                        costs.pv_lcoe_yuan_per_kwh,
                        "元/kWh",
                        "费用",
                        pv_generation_cost_yuan,
                        False,
                    ),
                    (
                        "purchase_price_peak",
                        "峰时网购电",
                        costs.purchase_price_peak_yuan_per_kwh,
                        "元/kWh",
                        "费用",
                        grid_purchase_kwh_by_tier["peak"]
                        * costs.purchase_price_peak_yuan_per_kwh,
                        False,
                    ),
                    (
                        "purchase_price_flat",
                        "平时网购电",
                        costs.purchase_price_flat_yuan_per_kwh,
                        "元/kWh",
                        "费用",
                        grid_purchase_kwh_by_tier["flat"]
                        * costs.purchase_price_flat_yuan_per_kwh,
                        False,
                    ),
                    (
                        "purchase_price_valley",
                        "谷时网购电",
                        costs.purchase_price_valley_yuan_per_kwh,
                        "元/kWh",
                        "费用",
                        grid_purchase_kwh_by_tier["valley"]
                        * costs.purchase_price_valley_yuan_per_kwh,
                        False,
                    ),
                    (
                        "wind_export_price",
                        "风电上网收入",
                        costs.wind_export_price_yuan_per_kwh,
                        "元/kWh",
                        "收入",
                        wind_export_revenue_yuan,
                        False,
                    ),
                    (
                        "pv_export_price",
                        "光伏上网收入",
                        costs.pv_export_price_yuan_per_kwh,
                        "元/kWh",
                        "收入",
                        pv_export_revenue_yuan,
                        False,
                    ),
                    (
                        "alkaline_om",
                        "ALK 电解槽运维费",
                        costs.alkaline_om_yuan_per_kwh,
                        "元/kWh",
                        "费用",
                        alkaline_om_yuan,
                        False,
                    ),
                    (
                        "pem_om",
                        "PEM 电解槽运维费",
                        costs.pem_om_yuan_per_kwh,
                        "元/kWh",
                        "费用",
                        pem_om_yuan,
                        False,
                    ),
                    (
                        "ammonia_om",
                        "合成氨装置运维费",
                        costs.ammonia_om_yuan_per_kwh,
                        "元/kWh",
                        "费用",
                        ammonia_om_yuan,
                        False,
                    ),
                    (
                        "ammonia_capex",
                        "合成氨装置资本摊销",
                        costs.ammonia_capex_yuan_per_kg_h2,
                        "元/(kgH₂/h)",
                        "费用",
                        ammonia_capex_daily_yuan,
                        True,
                    ),
                ],
            )
            if include_cost_sensitivity
            else []
        ),
    )

    review_items = [
        "题面字面自用率和新能源发电量减上网电量的物理自用率必须分别审阅。",
        "吨氨成本同时报告不含固定资本摊销的运行成本和含合成氨装置摊销的估算成本。",
        "上网收入按每小时风光余电比例分摊；本题附件中风电和光伏上网电价相同。",
    ]
    return TypicalDayResult(
        modeling_idea=(
            "将 24 小时标幺曲线乘以对应峰值或装机容量，叠加逐时电氢氨设备功率；"
            "逐小时用功率平衡拆分网购和上网，再汇总产量、电量、绿电指标和成本。"
        ),
        capacity_scale=capacity_scale,
        process_fractions=fractions,
        ammonia_production_tons=production_tons,
        hourly_balances=hourly,
        total_load_mwh=total_load_mwh,
        wind_generation_mwh=wind_generation_mwh,
        pv_generation_mwh=pv_generation_mwh,
        renewable_generation_mwh=renewable_generation_mwh,
        grid_purchase_mwh=grid_purchase_mwh,
        grid_export_mwh=grid_export_mwh,
        green_metrics=green_metrics,
        costs=costs_result,
        review_items=review_items,
        source_files=data.source_files,
    )


def compute_typical_day(data: EnergyParkDataset) -> TypicalDayResult:
    """按题面满负荷假设计算典型日功率和成本。"""

    return compute_energy_schedule(data, [1.0] * 24)
