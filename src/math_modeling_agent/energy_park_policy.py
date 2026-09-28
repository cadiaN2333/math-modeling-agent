"""生成带官方依据的电工杯问题五影响分析草稿。"""

from pydantic import BaseModel, Field

from .energy_park_continuous import ContinuousQuestionThreeResult
from .energy_park_discrete import DiscreteQuestionTwoResult


class PolicySource(BaseModel):
    """记录政策或技术依据的可追溯来源。"""

    source_id: str
    title: str
    publisher: str
    published: str
    url: str
    scope_note: str


class PolicyClaim(BaseModel):
    """一项利弊或建议及其依据。"""

    title: str
    explanation: str
    quantitative_evidence: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(min_length=1)


class EnergyPolicyComparison(BaseModel):
    """同一日产量下连续调度相对离散调度的年度差异。"""

    target_tons_per_day: float
    annual_cost_per_ton_difference_yuan: float
    annual_grid_purchase_difference_mwh: float
    annual_grid_export_difference_mwh: float


class EnergyParkPolicyReport(BaseModel):
    """问题五的政策影响、建议、定量上下文和局限。"""

    benefits: list[PolicyClaim]
    risks: list[PolicyClaim]
    recommendations: list[PolicyClaim]
    quantitative_comparisons: list[EnergyPolicyComparison]
    sources: list[PolicySource]
    limitations: list[str]


POLICY_SOURCES = [
    PolicySource(
        source_id="ndrc_2025_single_user_direct",
        title="关于有序推动绿电直连发展有关事项的通知（发改能源〔2025〕650号）",
        publisher="国家发展改革委、国家能源局",
        published="2025-05-30",
        url="https://zfxxgk.ndrc.gov.cn/web/iteminfo.jsp?id=20511",
        scope_note="单用户绿电直连的规划、运行、安全、源荷匹配及计量要求。",
    ),
    PolicySource(
        source_id="ndrc_2025_direct_interpretation",
        title="国家能源局有关负责同志就绿电直连通知答记者问",
        publisher="国家发展改革委、国家能源局",
        published="2025-05-30",
        url="https://www.ndrc.gov.cn/xxgk/jd/jd/202505/t20250530_1398141_ext.html",
        scope_note="解释系统调节责任、市场交易、辅助服务和绿电直连经济性。",
    ),
    PolicySource(
        source_id="nea_2024_renewable_consumption",
        title="关于做好新能源消纳工作保障新能源高质量发展的通知（国能发电力〔2024〕44号）",
        publisher="国家能源局",
        published="2024-05-28",
        url="https://zfxxgk.nea.gov.cn/2024-05/28/c_1310777105.htm",
        scope_note="新能源消纳、系统调节资源、电网配置和用户侧灵活性的要求。",
    ),
    PolicySource(
        source_id="nea_2026_hosting_capacity",
        title="国家能源局发布分布式电源接入电力系统承载力评估导则等行业标准",
        publisher="国家能源局",
        published="2026-01-06",
        url="https://www.nea.gov.cn/20260106/16146b8b20d741f998ac264a69e5b1e7/c.html",
        scope_note="说明分布式电源接入需开展系统级和设备级承载力评估。",
    ),
    PolicySource(
        source_id="ndrc_2026_multi_user_direct",
        title="关于有序推动多用户绿电直连发展有关事项的通知（发改能源〔2026〕688号）",
        publisher="国家发展改革委、国家能源局",
        published="2026-05-20",
        url="https://zfxxgk.ndrc.gov.cn/wap/iteminfo.jsp?id=20631",
        scope_note="多用户项目的储能、负荷灵活性、调度、弃电和绿电溯源条款；用于行业背景参考，不能替代本题单用户项目口径。",
    ),
]


def build_energy_park_policy_report(
    discrete: DiscreteQuestionTwoResult,
    continuous: ContinuousQuestionThreeResult,
) -> EnergyParkPolicyReport:
    """根据问题二、三结果生成政策影响分析草稿。"""

    q2_by_target = {
        summary.target_tons_per_day: summary for summary in discrete.annual_summaries
    }
    q3_by_target = {
        summary.target_tons_per_day: summary for summary in continuous.annual_summaries
    }
    comparisons = [
        EnergyPolicyComparison(
            target_tons_per_day=item["target_tons_per_day"],
            annual_cost_per_ton_difference_yuan=item[
                "annual_cost_per_ton_difference_yuan"
            ],
            annual_grid_purchase_difference_mwh=item[
                "annual_grid_purchase_difference_mwh"
            ],
            annual_grid_export_difference_mwh=item[
                "annual_grid_export_difference_mwh"
            ],
        )
        for item in continuous.discrete_comparison_by_target
    ]
    if not comparisons:
        raise ValueError("问题五需要问题二、三的年度对比结果")

    best_target = continuous.best_annual_cost_per_ton_target_tons_per_day
    discrete_best = q2_by_target[best_target]
    continuous_best = q3_by_target[best_target]
    comparison = next(
        item for item in comparisons if item.target_tons_per_day == best_target
    )

    cost_change = comparison.annual_cost_per_ton_difference_yuan
    if cost_change < 0:
        flexibility_text = (
            f"在连续调度当前选出的 {best_target:g} 吨/日产量下，"
            f"年加权吨氨成本比离散调度低 {-cost_change:.2f} 元/吨。"
        )
    elif cost_change > 0:
        flexibility_text = (
            f"在连续调度当前选出的 {best_target:g} 吨/日产量下，"
            f"年加权吨氨成本比离散调度高 {cost_change:.2f} 元/吨。"
        )
    else:
        flexibility_text = (
            f"在 {best_target:g} 吨/日产量下，当前模型中连续调节与离散调度的"
            "年加权吨氨成本相同。"
        )

    benefits = [
        PolicyClaim(
            title="提高新能源就近消纳",
            explanation=(
                "绿电直连通过源荷匹配使风光电量优先供给园区负荷；"
                "本题问题二、三的购电和上网差异可用于量化运行策略的影响。"
            ),
            quantitative_evidence=[
                f"连续调节相对离散调节的年度购电变化："
                f"{comparison.annual_grid_purchase_difference_mwh:.2f} MWh。",
                f"连续调节相对离散调节的年度上网变化："
                f"{comparison.annual_grid_export_difference_mwh:.2f} MWh。",
            ],
            source_ids=["ndrc_2025_single_user_direct"],
        ),
        PolicyClaim(
            title="柔性电氢氨负荷提供调节空间",
            explanation=(
                "电解槽和合成氨负荷可改变运行时段或负荷率，帮助园区跟随风光出力；"
                "效果取决于生产计划、分时电价和设备运行边界。"
            ),
            quantitative_evidence=[flexibility_text],
            source_ids=[
                "nea_2024_renewable_consumption",
                "ndrc_2026_multi_user_direct",
            ],
        ),
        PolicyClaim(
            title="降低对公共电网的能量交换需求",
            explanation=(
                "提高本地自用并优化生产时段，可能降低并网容量需求；"
                "本题应以逐时购电、上网和场景分布支撑该判断。"
            ),
            quantitative_evidence=[
                f"以 {best_target:g} 吨/日为例，离散调度物理自用指标全满足天数："
                f"{discrete_best.green_status_days_physical_self_use['全满足']} / 360；"
                f"连续调节为 {continuous_best.green_status_days_physical_self_use['全满足']} / 360。"
            ],
            source_ids=["ndrc_2025_single_user_direct"],
        ),
    ]
    risks = [
        PolicyClaim(
            title="风光波动增加系统调节需求",
            explanation=(
                "当风光出力与园区负荷在时间上不匹配时，项目需要负荷调节、储能或电网互济；"
                "仅比较日总发电量不足以证明逐时可平衡。"
            ),
            quantitative_evidence=[
                f"在 {best_target:g} 吨/日下，按题面公式统计，离散方案非全满足天数为 "
                f"{360 - discrete_best.green_status_days_contest_formula['全满足']} 天。",
                f"连续方案非全满足天数为 "
                f"{360 - continuous_best.green_status_days_contest_formula['全满足']} 天。",
            ],
            source_ids=["nea_2024_renewable_consumption"],
        ),
        PolicyClaim(
            title="接入容量和反向潮流需要专项评估",
            explanation=(
                "若余电集中上网，可能受配电网可开放容量、网络结构和安全边界限制；"
                "本题没有线路阻抗、节点拓扑和电压数据，不能据能量平衡直接断言电网可接纳。"
            ),
            quantitative_evidence=[
                f"典型日上网电量为 {discrete.typical_runs[0].operation.grid_export_mwh:.2f} MWh；"
                "该值是园区侧能量结果，不是配电网承载力结论。"
            ],
            source_ids=[
                "nea_2026_hosting_capacity",
                "ndrc_2025_single_user_direct",
            ],
        ),
        PolicyClaim(
            title="系统服务成本与责任需要公平分担",
            explanation=(
                "绿电直连并网仍使用公共电网调节、备用和供电保障服务；"
                "项目需要把输配电、系统运行费用及交换功率责任纳入经济性评估。"
            ),
            quantitative_evidence=[],
            source_ids=["ndrc_2025_direct_interpretation"],
        ),
    ]
    recommendations = [
        PolicyClaim(
            title="按负荷和配电网承载力确定新能源规模",
            explanation=(
                "规划时联合评估逐时源荷匹配、并网交换功率、设备容量和可开放容量，"
                "对反向潮流集中时段开展系统级及设备级评估。"
            ),
            quantitative_evidence=[
                f"本题场景中离散调度物理自用指标全满足天数为 "
                f"{discrete_best.green_status_days_physical_self_use['全满足']} / 360。"
            ],
            source_ids=[
                "ndrc_2025_single_user_direct",
                "nea_2026_hosting_capacity",
            ],
        ),
        PolicyClaim(
            title="提高电解和合成负荷的可调能力",
            explanation=(
                "依据风光预测滚动安排制氢制氨时段，结合储能和需求响应；"
                "用多个场景验证购电、上网、成本与绿电比例，而非只优化典型日。"
            ),
            quantitative_evidence=[flexibility_text],
            source_ids=[
                "nea_2024_renewable_consumption",
                "ndrc_2026_multi_user_direct",
            ],
        ),
        PolicyClaim(
            title="建设可观可测可调可控的并网与计量系统",
            explanation=(
                "配置保护、通信、监测及双向计量，监控并网交换功率，"
                "将安全责任边界和接网容量写入运行协议。"
            ),
            quantitative_evidence=[],
            source_ids=["ndrc_2025_single_user_direct"],
        ),
        PolicyClaim(
            title="明确系统费用与市场参与规则",
            explanation=(
                "透明核算输配电费、系统运行费及辅助服务收益，"
                "让灵活调度价值与项目承担的系统责任同步进入投资评估。"
            ),
            quantitative_evidence=[],
            source_ids=["ndrc_2025_direct_interpretation"],
        ),
    ]
    return EnergyParkPolicyReport(
        benefits=benefits,
        risks=risks,
        recommendations=recommendations,
        quantitative_comparisons=comparisons,
        sources=POLICY_SOURCES,
        limitations=[
            "Q2/Q3 只计算园区逐时能量平衡，未建配电网线路拓扑、潮流、电压、短路容量或故障模型。",
            "未输入储能优化、备用需求、系统调节费用及辅助服务市场价格，相关结论只作定性讨论。",
            "多用户直连文件用于行业背景引用，其条款不能替代本题单用户项目的适用规则。",
        ],
    )
