"""使用 OR-Tools SimpleMinCostFlow 求解整数单位的最小费用网络流。"""

from dataclasses import dataclass

from ortools.graph.python import min_cost_flow

from .models import MinCostFlowProblem


@dataclass
class MinCostFlowResult:
    """保存求解状态、每条路线的流量和整数总费用。"""

    status: str
    arc_flows: dict[str, int]
    total_cost: int | None


def _status_name(solver: min_cost_flow.SimpleMinCostFlow, status_code: object) -> str:
    """把 OR-Tools 状态枚举映射为稳定的字符串。"""

    known_statuses = {
        solver.OPTIMAL: "OPTIMAL",
        solver.INFEASIBLE: "INFEASIBLE",
        solver.BAD_COST_RANGE: "BAD_COST_RANGE",
        solver.BAD_CAPACITY_RANGE: "BAD_CAPACITY_RANGE",
    }
    if status_code in known_statuses:
        return known_statuses[status_code]

    status_name = getattr(status_code, "name", None)
    if status_name == "UNBALANCED":
        return "INFEASIBLE"
    if status_name:
        return f"UNKNOWN_{status_name}"
    return f"UNKNOWN_{status_code}"


def solve_min_cost_flow(problem: MinCostFlowProblem) -> MinCostFlowResult:
    """将供需节点和有向路线交给 OR-Tools 并返回最优路线流量。"""

    try:
        solver = min_cost_flow.SimpleMinCostFlow()
    except RuntimeError:
        return MinCostFlowResult(
            status="SOLVER_UNAVAILABLE",
            arc_flows={},
            total_cost=None,
        )

    node_indices = {
        node.node_id: index
        for index, node in enumerate(problem.nodes)
    }
    solver_arcs = [
        (
            arc.arc_id,
            solver.add_arc_with_capacity_and_unit_cost(
                node_indices[arc.from_node],
                node_indices[arc.to_node],
                arc.capacity,
                arc.unit_cost,
            ),
        )
        for arc in problem.arcs
    ]

    for node in problem.nodes:
        solver.set_node_supply(node_indices[node.node_id], node.supply)

    status = _status_name(solver, solver.solve())
    if status != "OPTIMAL":
        return MinCostFlowResult(
            status=status,
            arc_flows={},
            total_cost=None,
        )

    arc_ids = [solver_arc_index for _, solver_arc_index in solver_arcs]
    flows = solver.flows(arc_ids)
    return MinCostFlowResult(
        status=status,
        arc_flows={
            arc_id: int(flow)
            for (arc_id, _), flow in zip(solver_arcs, flows)
        },
        total_cost=int(solver.optimal_cost()),
    )
