from types import SimpleNamespace
import json


def test_problem_analysis_schema_avoids_anyof_for_deepseek() -> None:
    from math_modeling_agent.analysis_agent import ProblemAnalysis

    schema = ProblemAnalysis.model_json_schema()

    assert "anyOf" not in json.dumps(schema)


def test_analysis_instructions_limit_current_domain_to_scheduling() -> None:
    from math_modeling_agent.analysis_agent import ANALYSIS_INSTRUCTIONS

    assert "只支持员工排班" in ANALYSIS_INSTRUCTIONS
    assert "运输/网络流" in ANALYSIS_INSTRUCTIONS
    assert "unsupported" in ANALYSIS_INSTRUCTIONS


class FakeResponses:
    def __init__(self, parsed_result) -> None:
        self.parsed_result = parsed_result
        self.arguments = None

    def parse(self, **arguments):
        self.arguments = arguments
        return SimpleNamespace(output_parsed=self.parsed_result)


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
        summary="用户希望安排员工班次。",
        known_facts=["需要排班"],
        missing_information=["员工人数"],
        clarifying_questions=["有多少名员工？"],
        unsupported_reasons=[],
        subtasks=[],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
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
        summary="需要员工信息。",
        known_facts=[],
        missing_information=["员工人数"],
        clarifying_questions=["有多少名员工？"],
        unsupported_reasons=[],
        subtasks=[],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
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
        summary="需要更多排班信息。",
        known_facts=[],
        missing_information=["员工人数"],
        clarifying_questions=["有多少名员工？"],
        unsupported_reasons=[],
        subtasks=[],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
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
    )

    problem = to_scheduling_problem(analysis)

    assert problem.employees[0].skills == {"急救"}
    assert problem.coverage_requirements[0].required_skill_counts == {"急救": 1}
    assert problem.coverage_requirements[1].required_skill_counts == {}
