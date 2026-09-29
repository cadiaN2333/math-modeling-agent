from types import SimpleNamespace
import json


def empty_min_cost_flow_draft():
    from math_modeling_agent.analysis_agent import MinCostFlowDraft

    return MinCostFlowDraft(
        nodes=[],
        arcs=[],
        flow_unit="not_applicable",
        cost_unit="not_applicable",
    )


def test_problem_analysis_schema_avoids_anyof_for_deepseek() -> None:
    from math_modeling_agent.analysis_agent import ProblemAnalysis

    schema = ProblemAnalysis.model_json_schema()

    assert "anyOf" not in json.dumps(schema)
    assert "problem_family" in schema["properties"]
    assert "linear_program_draft" in schema["properties"]
    assert "problem_family" in schema["required"]
    assert "linear_program_draft" in schema["required"]


def test_analysis_instructions_describe_all_supported_domains_and_boundaries() -> None:
    from math_modeling_agent.analysis_agent import ANALYSIS_INSTRUCTIONS

    assert "员工排班、连续/混合整数单目标线性规划" in ANALYSIS_INSTRUCTIONS
    assert "单商品最小费用网络流" in ANALYSIS_INSTRUCTIONS
    assert "多商品流" in ANALYSIS_INSTRUCTIONS
    assert "unsupported" in ANALYSIS_INSTRUCTIONS


def test_analysis_instructions_limit_lp_to_continuous_single_objective() -> None:
    from math_modeling_agent.analysis_agent import ANALYSIS_INSTRUCTIONS

    assert "连续、整数和二进制变量" in ANALYSIS_INSTRUCTIONS
    assert "单目标" in ANALYSIS_INSTRUCTIONS
    assert "整数变量" in ANALYSIS_INSTRUCTIONS
    assert "非线性" in ANALYSIS_INSTRUCTIONS


def test_linear_variable_domain_is_required_and_enum_limited_in_schema() -> None:
    from math_modeling_agent.analysis_agent import ProblemAnalysis

    schema = ProblemAnalysis.model_json_schema()
    variable_schema = schema["$defs"]["LinearVariableDraft"]

    assert "anyOf" not in json.dumps(schema)
    assert variable_schema["properties"]["domain"]["enum"] == [
        "continuous",
        "integer",
        "binary",
    ]
    assert "domain" in variable_schema["required"]


def test_linear_variable_draft_treats_legacy_missing_domain_as_continuous() -> None:
    from math_modeling_agent.analysis_agent import LinearVariableDraft

    variable = LinearVariableDraft.model_validate({"name": "x", "unit": "件"})

    assert variable.domain == "continuous"


def test_problem_analysis_schema_includes_fixed_min_cost_flow_draft() -> None:
    from math_modeling_agent.analysis_agent import ProblemAnalysis

    schema = ProblemAnalysis.model_json_schema()

    assert "anyOf" not in json.dumps(schema)
    assert "minimum_cost_flow_draft" in schema["properties"]
    assert "minimum_cost_flow_draft" in schema["required"]
    assert "minimum_cost_flow" in schema["properties"]["problem_family"]["enum"]


def test_ready_min_cost_flow_analysis_converts_to_internal_problem() -> None:
    from math_modeling_agent.analysis_agent import to_minimum_cost_flow_problem
    from min_cost_flow_fixtures import make_ready_analysis

    problem = to_minimum_cost_flow_problem(make_ready_analysis())

    assert len(problem.nodes) == 4
    assert [node.supply for node in problem.nodes] == [20, 30, -25, -25]
    assert len(problem.arcs) == 4
    assert problem.flow_unit == "箱"
    assert problem.cost_unit == "元/箱"


def test_nonready_min_cost_flow_analysis_requires_empty_domain_draft() -> None:
    import pytest
    from pydantic import ValidationError

    from math_modeling_agent.analysis_agent import (
        LinearProgramDraft,
        MinCostFlowArcDraft,
        MinCostFlowDraft,
        MinCostFlowNodeDraft,
        ProblemAnalysis,
        SchedulingDraft,
    )

    with pytest.raises(ValidationError, match="需要追问或不支持的问题必须使用空领域草稿"):
        ProblemAnalysis(
            status="unsupported",
            problem_family="minimum_cost_flow",
            summary="这是多商品流问题。",
            known_facts=[],
            missing_information=[],
            clarifying_questions=[],
            unsupported_reasons=["当前不支持多商品流。"],
            subtasks=[],
            scheduling_draft=SchedulingDraft(
                employees=[], shifts=[], coverage_requirements=[]
            ),
            linear_program_draft=LinearProgramDraft(
                variables=[],
                objective_direction="not_applicable",
                objective_terms=[],
                constraints=[],
            ),
            minimum_cost_flow_draft=MinCostFlowDraft(
                nodes=[
                    MinCostFlowNodeDraft(node_id="W1", name="仓库一", supply=10),
                    MinCostFlowNodeDraft(node_id="S1", name="门店一", supply=-10),
                ],
                arcs=[
                    MinCostFlowArcDraft(
                        arc_id="A1",
                        from_node="W1",
                        to_node="S1",
                        capacity=10,
                        unit_cost=1,
                    )
                ],
                flow_unit="箱",
                cost_unit="元/箱",
            ),
        )


def test_analysis_downgrades_vague_unsupported_to_clarification_when_data_is_missing() -> None:
    from types import SimpleNamespace

    from math_modeling_agent.analysis_agent import (
        LinearProgramDraft,
        ProblemAnalysis,
        SchedulingDraft,
        analyze_problem,
    )

    analysis = ProblemAnalysis(
        status="unsupported",
        problem_family="employee_scheduling",
        summary="排班所需员工信息不完整。",
        known_facts=["林晓要上第一班"],
        missing_information=["其他员工名单", "每名员工的最大工时"],
        clarifying_questions=["请补充员工名单和每人的最大工时。"],
        unsupported_reasons=["输入信息尚未完整，暂时无法直接生成方案。"],
        subtasks=[],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
        linear_program_draft=LinearProgramDraft(
            variables=[],
            objective_direction="not_applicable",
            objective_terms=[],
            constraints=[],
        ),
        minimum_cost_flow_draft=empty_min_cost_flow_draft(),
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=lambda **arguments: SimpleNamespace(output_parsed=analysis)
        )
    )

    result = analyze_problem("员工和工时信息未齐全", client=client)

    assert result.status == "needs_clarification"
    assert result.unsupported_reasons == []
    assert result.missing_information == analysis.missing_information


def test_analysis_keeps_explicit_unsupported_feature_even_if_data_is_missing() -> None:
    from types import SimpleNamespace

    from math_modeling_agent.analysis_agent import (
        LinearProgramDraft,
        ProblemAnalysis,
        SchedulingDraft,
        analyze_problem,
    )

    analysis = ProblemAnalysis(
        status="unsupported",
        problem_family="employee_scheduling",
        summary="用户要求排班并限制班次时间重叠。",
        known_facts=[],
        missing_information=["员工名单"],
        clarifying_questions=["请补充员工名单。"],
        unsupported_reasons=["当前版本尚不支持班次时间重叠约束。"],
        subtasks=[],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
        linear_program_draft=LinearProgramDraft(
            variables=[],
            objective_direction="not_applicable",
            objective_terms=[],
            constraints=[],
        ),
        minimum_cost_flow_draft=empty_min_cost_flow_draft(),
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=lambda **arguments: SimpleNamespace(output_parsed=analysis)
        )
    )

    result = analyze_problem("带时间重叠约束的排班", client=client)

    assert result.status == "unsupported"
    assert result.unsupported_reasons == analysis.unsupported_reasons


def test_analysis_does_not_treat_integer_schedule_counts_as_unsupported() -> None:
    from types import SimpleNamespace

    from math_modeling_agent.analysis_agent import (
        ProblemAnalysis,
        SchedulingDraft,
        analyze_problem,
    )

    analysis = ProblemAnalysis(
        status="unsupported",
        problem_family="employee_scheduling",
        summary="员工人数应为整数，但员工信息不全。",
        known_facts=[],
        missing_information=["员工名单与每人最大工时"],
        clarifying_questions=["请提供员工名单与每人最大工时。"],
        unsupported_reasons=["员工分配是整数决策，但目前信息不全。"],
        subtasks=[],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
        linear_program_draft=empty_linear_program_draft(),
        minimum_cost_flow_draft=empty_min_cost_flow_draft(),
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=lambda **arguments: SimpleNamespace(output_parsed=analysis)
        )
    )

    result = analyze_problem("员工信息不全，人数是整数", client=client)

    assert result.status == "needs_clarification"
    assert result.unsupported_reasons == []


class FakeResponses:
    def __init__(self, parsed_result) -> None:
        self.parsed_result = parsed_result
        self.arguments = None

    def parse(self, **arguments):
        self.arguments = arguments
        return SimpleNamespace(output_parsed=self.parsed_result)


def empty_linear_program_draft():
    from math_modeling_agent.analysis_agent import LinearProgramDraft

    return LinearProgramDraft(
        variables=[],
        objective_direction="not_applicable",
        objective_terms=[],
        constraints=[],
    )


def test_load_project_environment_uses_root_env_without_override(monkeypatch) -> None:
    import sys
    from pathlib import Path
    from types import ModuleType

    from math_modeling_agent.analysis_agent import load_project_environment

    calls = []
    fake_dotenv = ModuleType("dotenv")

    def fake_load_dotenv(*, dotenv_path, override):
        calls.append((Path(dotenv_path), override))

    fake_dotenv.load_dotenv = fake_load_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    load_project_environment()

    expected_env = Path(__file__).resolve().parents[1] / ".env"
    assert calls == [(expected_env, False)]


def test_analyzer_uses_deepseek_structured_responses() -> None:
    from math_modeling_agent.analysis_agent import (
        ProblemAnalysis,
        SchedulingDraft,
        analyze_problem,
    )

    expected = ProblemAnalysis(
        status="needs_clarification",
        problem_family="employee_scheduling",
        summary="用户希望安排员工班次。",
        known_facts=["需要排班"],
        missing_information=["员工人数"],
        clarifying_questions=["有多少名员工？"],
        unsupported_reasons=[],
        subtasks=[],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
        linear_program_draft=empty_linear_program_draft(),
        minimum_cost_flow_draft=empty_min_cost_flow_draft(),
    )
    responses = FakeResponses(expected)
    client = SimpleNamespace(responses=responses)

    result = analyze_problem("帮我安排一下员工班次。", client=client)

    assert result == expected
    assert responses.arguments["model"] == "deepseek-flash"
    assert responses.arguments["text_format"] is ProblemAnalysis
    assert "不能编造" in responses.arguments["instructions"]
    assert "scheduling_draft" in responses.arguments["instructions"]


def test_analyzer_loads_project_environment_before_default_client(monkeypatch) -> None:
    import sys
    from types import ModuleType

    from math_modeling_agent import analysis_agent
    from math_modeling_agent.analysis_agent import ProblemAnalysis, SchedulingDraft

    expected = ProblemAnalysis(
        status="needs_clarification",
        problem_family="employee_scheduling",
        summary="需要员工信息。",
        known_facts=[],
        missing_information=["员工人数"],
        clarifying_questions=["有多少名员工？"],
        unsupported_reasons=[],
        subtasks=[],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
        linear_program_draft=empty_linear_program_draft(),
        minimum_cost_flow_draft=empty_min_cost_flow_draft(),
    )
    calls = []
    monkeypatch.setattr(
        analysis_agent,
        "load_project_environment",
        lambda: calls.append("load_env"),
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")

    fake_openai = ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            calls.append(("client", api_key))
            self.responses = FakeResponses(expected)

    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    result = analysis_agent.analyze_problem("需要安排员工班次", client=None)

    assert result == expected
    assert calls == ["load_env", ("client", "test-only-key")]


def test_analyzer_rejects_empty_request_before_calling_model() -> None:
    import pytest

    from math_modeling_agent.analysis_agent import analyze_problem

    with pytest.raises(ValueError, match="不能为空"):
        analyze_problem("   ", client=object())


def test_ready_analysis_retrieves_hmml_methods_for_each_subtask() -> None:
    from math_modeling_agent.analysis_agent import (
        AnalysisSubtask,
        CoverageDraft,
        EmployeeDraft,
        ProblemAnalysis,
        SchedulingDraft,
        ShiftDraft,
        SkillRequirementDraft,
        retrieve_methods_for_subtasks,
    )

    analysis = ProblemAnalysis(
        status="ready",
        problem_family="employee_scheduling",
        summary="为员工安排符合技能与工时要求的班次。",
        known_facts=["每班至少一人"],
        missing_information=[],
        clarifying_questions=[],
        unsupported_reasons=[],
        subtasks=[
            AnalysisSubtask(
                task_id="T1",
                description="安排员工班次并满足技能覆盖。",
                objective="找到可行且总工时较少的排班。",
                data_needed=["员工技能", "最大工时", "班次时长"],
                depends_on=[],
                hmml_problem_query="员工排班、班次、技能约束、最低人数、工时上限",
                hmml_goal_query="满足班次覆盖并减少总排班时间",
            )
        ],
        scheduling_draft=SchedulingDraft(
            employees=[
                EmployeeDraft(
                    employee_id="E1",
                    name="急救员",
                    skills=["急救"],
                    max_hours=8,
                )
            ],
            shifts=[ShiftDraft(shift_id="S1", duration_hours=8)],
            coverage_requirements=[
                CoverageDraft(
                    shift_id="S1",
                    minimum_employees=1,
                    skill_requirements=[
                        SkillRequirementDraft(skill="急救", minimum_count=1)
                    ],
                )
            ],
        ),
        linear_program_draft=empty_linear_program_draft(),
        minimum_cost_flow_draft=empty_min_cost_flow_draft(),
    )

    recommendations = retrieve_methods_for_subtasks(analysis)

    assert recommendations["T1"]
    assert recommendations["T1"][0].method_id == "cp_sat_scheduling"


def test_incomplete_analysis_does_not_retrieve_methods() -> None:
    from math_modeling_agent.analysis_agent import (
        ProblemAnalysis,
        SchedulingDraft,
        retrieve_methods_for_subtasks,
    )

    analysis = ProblemAnalysis(
        status="needs_clarification",
        problem_family="employee_scheduling",
        summary="需要更多排班信息。",
        known_facts=[],
        missing_information=["员工人数"],
        clarifying_questions=["有多少名员工？"],
        unsupported_reasons=[],
        subtasks=[],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
        linear_program_draft=empty_linear_program_draft(),
        minimum_cost_flow_draft=empty_min_cost_flow_draft(),
    )

    assert retrieve_methods_for_subtasks(analysis) == {}


def test_ready_analysis_converts_to_solver_problem() -> None:
    from math_modeling_agent.analysis_agent import (
        AnalysisSubtask,
        CoverageDraft,
        EmployeeDraft,
        ProblemAnalysis,
        SchedulingDraft,
        ShiftDraft,
        SkillRequirementDraft,
        to_scheduling_problem,
    )

    analysis = ProblemAnalysis(
        status="ready",
        problem_family="employee_scheduling",
        summary="为急救员和普通员工安排两个班次。",
        known_facts=["两个班次各8小时"],
        missing_information=[],
        clarifying_questions=[],
        unsupported_reasons=[],
        subtasks=[
            AnalysisSubtask(
                task_id="T1",
                description="安排两个班次。",
                objective="满足人数、技能和工时要求。",
                data_needed=[],
                depends_on=[],
                hmml_problem_query="员工排班与技能覆盖",
                hmml_goal_query="满足班次需求",
            )
        ],
        scheduling_draft=SchedulingDraft(
            employees=[
                EmployeeDraft(
                    employee_id="E1",
                    name="林晓",
                    skills=["急救"],
                    max_hours=8,
                ),
                EmployeeDraft(
                    employee_id="E2",
                    name="陈立",
                    skills=[],
                    max_hours=8,
                ),
            ],
            shifts=[
                ShiftDraft(shift_id="S1", duration_hours=8),
                ShiftDraft(shift_id="S2", duration_hours=8),
            ],
            coverage_requirements=[
                CoverageDraft(
                    shift_id="S1",
                    minimum_employees=1,
                    skill_requirements=[
                        SkillRequirementDraft(skill="急救", minimum_count=1)
                    ],
                ),
                CoverageDraft(
                    shift_id="S2",
                    minimum_employees=1,
                    skill_requirements=[],
                ),
            ],
        ),
        linear_program_draft=empty_linear_program_draft(),
        minimum_cost_flow_draft=empty_min_cost_flow_draft(),
    )

    problem = to_scheduling_problem(analysis)

    assert problem.employees[0].skills == {"急救"}
    assert problem.coverage_requirements[0].required_skill_counts == {"急救": 1}
    assert problem.coverage_requirements[1].required_skill_counts == {}


def test_ready_linear_program_analysis_converts_to_internal_problem() -> None:
    from math_modeling_agent.analysis_agent import (
        AnalysisSubtask,
        LinearConstraintDraft,
        LinearProgramDraft,
        LinearTermDraft,
        LinearVariableDraft,
        ProblemAnalysis,
        SchedulingDraft,
        to_linear_program_problem,
    )

    analysis = ProblemAnalysis(
        status="ready",
        problem_family="linear_programming",
        summary="最大化两种产品的利润。",
        known_facts=["工时上限为100小时", "原料上限为80单位"],
        missing_information=[],
        clarifying_questions=[],
        unsupported_reasons=[],
        subtasks=[
            AnalysisSubtask(
                task_id="T1",
                description="建立产品产量线性规划。",
                objective="最大化总利润。",
                data_needed=[],
                depends_on=[],
                hmml_problem_query="连续变量、线性目标和资源约束的生产计划问题",
                hmml_goal_query="最大化利润并满足工时和原料限制",
            )
        ],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
        linear_program_draft=LinearProgramDraft(
            variables=[
                LinearVariableDraft(name="A", unit="件", domain="integer"),
                LinearVariableDraft(name="B", unit="件", domain="continuous"),
            ],
            objective_direction="maximize",
            objective_terms=[
                LinearTermDraft(variable="A", coefficient=40),
                LinearTermDraft(variable="B", coefficient=30),
            ],
            constraints=[
                LinearConstraintDraft(
                    constraint_id="labor",
                    terms=[
                        LinearTermDraft(variable="A", coefficient=2),
                        LinearTermDraft(variable="B", coefficient=1),
                    ],
                    relation="<=",
                    rhs=100,
                ),
                LinearConstraintDraft(
                    constraint_id="material",
                    terms=[
                        LinearTermDraft(variable="A", coefficient=1),
                        LinearTermDraft(variable="B", coefficient=1),
                    ],
                    relation="<=",
                    rhs=80,
                ),
                LinearConstraintDraft(
                    constraint_id="A_nonnegative",
                    terms=[LinearTermDraft(variable="A", coefficient=1)],
                    relation=">=",
                    rhs=0,
                ),
                LinearConstraintDraft(
                    constraint_id="B_nonnegative",
                    terms=[LinearTermDraft(variable="B", coefficient=1)],
                    relation=">=",
                    rhs=0,
                ),
            ],
        ),
        minimum_cost_flow_draft=empty_min_cost_flow_draft(),
    )

    problem = to_linear_program_problem(analysis)

    assert [variable.name for variable in problem.variables] == ["A", "B"]
    assert [variable.domain for variable in problem.variables] == [
        "integer",
        "continuous",
    ]
    assert problem.objective.direction == "maximize"
    assert [constraint.name for constraint in problem.constraints] == [
        "labor",
        "material",
        "A_nonnegative",
        "B_nonnegative",
    ]


def test_incomplete_integer_linear_program_is_clarification_not_unsupported() -> None:
    from types import SimpleNamespace

    from math_modeling_agent.analysis_agent import (
        ProblemAnalysis,
        SchedulingDraft,
        analyze_problem,
    )

    analysis = ProblemAnalysis(
        status="unsupported",
        problem_family="linear_programming",
        summary="整数生产计划缺少资源上限。",
        known_facts=["产品数量为整数"],
        missing_information=["可用工时"],
        clarifying_questions=["可用工时上限是多少？"],
        unsupported_reasons=["整数变量属于优化决策"],
        subtasks=[],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
        linear_program_draft=empty_linear_program_draft(),
        minimum_cost_flow_draft=empty_min_cost_flow_draft(),
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=lambda **arguments: SimpleNamespace(output_parsed=analysis)
        )
    )

    result = analyze_problem("整数生产计划，资源上限还未提供", client=client)

    assert result.status == "needs_clarification"
    assert result.unsupported_reasons == []
