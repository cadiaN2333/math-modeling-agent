"""独立复核最小费用网络流的路线容量、节点平衡和总费用。"""

from dataclasses import dataclass

from .models import MinCostFlowProblem


@dataclass
class MinCostFlowValidationReport:
    """保存网络流解的验证结果和独立重算总费用。"""

    is_valid: bool
    errors: list[str]
    recomputed_total_cost: int | None


def validate_min_cost_flow_solution(
    problem: MinCostFlowProblem,
    arc_flows: dict[str, int],
    reported_total_cost: int | None,
) -> MinCostFlowValidationReport:
    """独立检查所有边流量，并按节点守恒重算费用。"""

    errors: list[str] = []
    expected_arc_ids = {arc.arc_id for arc in problem.arcs}
    supplied_arc_ids = set(arc_flows)

    for arc_id in sorted(expected_arc_ids - supplied_arc_ids):
        errors.append(f"缺少路线流量：{arc_id}")
    for arc_id in sorted(supplied_arc_ids - expected_arc_ids):
        errors.append(f"包含未知路线：{arc_id}")

    valid_integer_flows: dict[str, int] = {}
    arc_by_id = {arc.arc_id: arc for arc in problem.arcs}
    for arc_id in sorted(expected_arc_ids & supplied_arc_ids):
        flow = arc_flows[arc_id]
        if not isinstance(flow, int) or isinstance(flow, bool) or flow < 0:
            errors.append(f"路线流量必须是非负整数：{arc_id}")
            continue

        valid_integer_flows[arc_id] = flow
        if flow > arc_by_id[arc_id].capacity:
            errors.append(
                f"路线容量超限：{arc_id} 的流量 {flow} 大于容量 "
                f"{arc_by_id[arc_id].capacity}"
            )

    all_flows_are_valid = (
        expected_arc_ids == supplied_arc_ids
        and len(valid_integer_flows) == len(expected_arc_ids)
    )
    recomputed_total_cost = None
    if all_flows_are_valid:
        outgoing_by_node = {node.node_id: 0 for node in problem.nodes}
        incoming_by_node = {node.node_id: 0 for node in problem.nodes}
        recomputed_total_cost = 0

        for arc in problem.arcs:
            flow = valid_integer_flows[arc.arc_id]
            outgoing_by_node[arc.from_node] += flow
            incoming_by_node[arc.to_node] += flow
            recomputed_total_cost += flow * arc.unit_cost

        for node in problem.nodes:
            net_outflow = (
                outgoing_by_node[node.node_id] - incoming_by_node[node.node_id]
            )
            if net_outflow != node.supply:
                errors.append(
                    f"节点供需守恒不满足：{node.node_id} 的净流出为 {net_outflow}，"
                    f"供给要求为 {node.supply}"
                )

    if (
        not isinstance(reported_total_cost, int)
        or isinstance(reported_total_cost, bool)
    ):
        errors.append("报告的总费用必须是整数")
    elif (
        recomputed_total_cost is not None
        and reported_total_cost != recomputed_total_cost
    ):
        errors.append(
            f"总费用不一致：重算为 {recomputed_total_cost}，报告为 {reported_total_cost}"
        )

    return MinCostFlowValidationReport(
        is_valid=not errors,
        errors=errors,
        recomputed_total_cost=recomputed_total_cost,
    )
