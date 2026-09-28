EXPECTED_ARC_FLOWS = {
    "W1_S1": 20,
    "W1_S2": 0,
    "W2_S1": 5,
    "W2_S2": 25,
}


def make_transport_problem():
    from math_modeling_agent.models import FlowArc, FlowNode, MinCostFlowProblem

    return MinCostFlowProblem(
        nodes=[
            FlowNode(node_id="W1", name="仓库一", supply=20),
            FlowNode(node_id="W2", name="仓库二", supply=30),
            FlowNode(node_id="S1", name="门店一", supply=-25),
            FlowNode(node_id="S2", name="门店二", supply=-25),
        ],
        arcs=[
            FlowArc(
                arc_id="W1_S1",
                from_node="W1",
                to_node="S1",
                capacity=20,
                unit_cost=2,
            ),
            FlowArc(
                arc_id="W1_S2",
                from_node="W1",
                to_node="S2",
                capacity=20,
                unit_cost=4,
            ),
            FlowArc(
                arc_id="W2_S1",
                from_node="W2",
                to_node="S1",
                capacity=25,
                unit_cost=3,
            ),
            FlowArc(
                arc_id="W2_S2",
                from_node="W2",
                to_node="S2",
                capacity=30,
                unit_cost=1,
            ),
        ],
        flow_unit="箱",
        cost_unit="元/箱",
    )


def make_ready_analysis():
    from math_modeling_agent.analysis_agent import (
        AnalysisSubtask,
        LinearProgramDraft,
        MinCostFlowArcDraft,
        MinCostFlowDraft,
        MinCostFlowNodeDraft,
        ProblemAnalysis,
        SchedulingDraft,
    )

    return ProblemAnalysis(
        status="ready",
        problem_family="minimum_cost_flow",
        summary="以最低费用将两仓货物送至两家门店。",
        known_facts=["供给、需求、路线容量与单位费用均已给出。"],
        missing_information=[],
        clarifying_questions=[],
        unsupported_reasons=[],
        subtasks=[
            AnalysisSubtask(
                task_id="T1",
                description="建立仓库到门店的最小费用流模型。",
                objective="满足所有供需并最小化运输总费用。",
                data_needed=[],
                depends_on=[],
                hmml_problem_query="仓库供给、门店需求、路线容量与运输费用",
                hmml_goal_query="满足全部需求并最小化总运输费用",
            )
        ],
        scheduling_draft=SchedulingDraft(
            employees=[],
            shifts=[],
            coverage_requirements=[],
        ),
        linear_program_draft=LinearProgramDraft(
            variables=[],
            objective_direction="not_applicable",
            objective_terms=[],
            constraints=[],
        ),
        minimum_cost_flow_draft=MinCostFlowDraft(
            nodes=[
                MinCostFlowNodeDraft(node_id="W1", name="仓库一", supply=20),
                MinCostFlowNodeDraft(node_id="W2", name="仓库二", supply=30),
                MinCostFlowNodeDraft(node_id="S1", name="门店一", supply=-25),
                MinCostFlowNodeDraft(node_id="S2", name="门店二", supply=-25),
            ],
            arcs=[
                MinCostFlowArcDraft(
                    arc_id="W1_S1",
                    from_node="W1",
                    to_node="S1",
                    capacity=20,
                    unit_cost=2,
                ),
                MinCostFlowArcDraft(
                    arc_id="W1_S2",
                    from_node="W1",
                    to_node="S2",
                    capacity=20,
                    unit_cost=4,
                ),
                MinCostFlowArcDraft(
                    arc_id="W2_S1",
                    from_node="W2",
                    to_node="S1",
                    capacity=25,
                    unit_cost=3,
                ),
                MinCostFlowArcDraft(
                    arc_id="W2_S2",
                    from_node="W2",
                    to_node="S2",
                    capacity=30,
                    unit_cost=1,
                ),
            ],
            flow_unit="箱",
            cost_unit="元/箱",
        ),
    )


def make_transport_eval_payload():
    return {
        "analysis": make_ready_analysis().model_dump(mode="json"),
        "modeling_run": {
            "solver_result": {
                "status": "OPTIMAL",
                "arc_flows": EXPECTED_ARC_FLOWS,
                "total_cost": 80,
            },
            "validation_report": {
                "is_valid": True,
                "errors": [],
                "recomputed_total_cost": 80,
            },
            "method_recommendations": [
                {
                    "method_id": "minimum_cost_flow",
                    "implementation_status": "已实现",
                }
            ],
        },
    }
