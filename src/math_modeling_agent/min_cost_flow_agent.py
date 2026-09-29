"""编排最小费用网络流方法检索、求解和独立验证。"""

from dataclasses import dataclass

from .min_cost_flow_solver import MinCostFlowResult
from .min_cost_flow_validator import (
    MinCostFlowValidationReport,
    validate_min_cost_flow_solution,
)
from .models import MinCostFlowProblem
from .optimization_compilers import (
    compile_min_cost_flow_problem,
    decode_flow_solution,
)
from .optimization_registry import SolverRegistry
from .retriever import HMMLRetriever, MethodRecommendation


@dataclass
class MinCostFlowModelingRun:
    """保存最小费用流方法建议、求解结果和验证报告。"""

    solver_result: MinCostFlowResult
    validation_report: MinCostFlowValidationReport | None
    method_recommendations: list[MethodRecommendation]


def _describe_min_cost_flow_problem(problem: MinCostFlowProblem) -> str:
    """将节点、路线与单位整理成 HMML 可检索的描述。"""

    supply_nodes = sum(node.supply > 0 for node in problem.nodes)
    demand_nodes = sum(node.supply < 0 for node in problem.nodes)
    return (
        f"整数单商品最小费用网络流，有 {len(problem.nodes)} 个节点和 "
        f"{len(problem.arcs)} 条有向路线；包含 {supply_nodes} 个供给节点、"
        f"{demand_nodes} 个需求节点，流量单位为 {problem.flow_unit}，"
        f"路线单位费用为 {problem.cost_unit}。"
    )


def run_min_cost_flow_modeling(
    problem: MinCostFlowProblem,
) -> MinCostFlowModelingRun:
    """检索最小费用流方法，求解后仅验证最优流量。"""

    method_recommendations = HMMLRetriever().retrieve(
        problem_description=_describe_min_cost_flow_problem(problem),
        desired_outcome="满足全部节点供需，并最小化所有运输路线的总费用。",
    )
    # 网络语义直接保留在 IR，再由专用 SimpleMinCostFlow 后端求解。
    ir = compile_min_cost_flow_problem(problem, problem_id="flow-run")
    backend = SolverRegistry().select(ir)
    if backend is None:
        solver_result = MinCostFlowResult(
            status="UNSUPPORTED_MODEL",
            arc_flows={},
            total_cost=None,
        )
    else:
        ir_result = backend.solve(ir)
        if ir_result.status == "OPTIMAL" and isinstance(
            ir_result.objective_value,
            int,
        ):
            arc_flows = decode_flow_solution(problem, ir_result.variable_values)
            total_cost = ir_result.objective_value
        else:
            arc_flows = {}
            total_cost = None
        solver_result = MinCostFlowResult(
            status=ir_result.status,
            arc_flows=arc_flows,
            total_cost=total_cost,
        )

    if solver_result.status != "OPTIMAL":
        return MinCostFlowModelingRun(
            solver_result=solver_result,
            validation_report=None,
            method_recommendations=method_recommendations,
        )

    validation_report = validate_min_cost_flow_solution(
        problem,
        solver_result.arc_flows,
        solver_result.total_cost,
    )
    return MinCostFlowModelingRun(
        solver_result=solver_result,
        validation_report=validation_report,
        method_recommendations=method_recommendations,
    )
