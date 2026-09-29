import json
import pytest


def test_cli_json_output_preserves_unicode_with_gbk_console(monkeypatch) -> None:
    from io import BytesIO, TextIOWrapper

    from math_modeling_agent import cli

    output_bytes = BytesIO()
    console = TextIOWrapper(output_bytes, encoding="gbk")
    monkeypatch.setattr(cli.sys, "stdout", console)

    cli._print_json({"化学式": "H₂", "说明": "绿电成本 🚀"})
    console.flush()

    serialized = output_bytes.getvalue().decode("gbk")
    assert json.loads(serialized) == {"化学式": "H₂", "说明": "绿电成本 🚀"}


def empty_linear_program_draft():
    from math_modeling_agent.analysis_agent import LinearProgramDraft

    return LinearProgramDraft(
        variables=[],
        objective_direction="not_applicable",
        objective_terms=[],
        constraints=[],
    )


def empty_min_cost_flow_draft():
    from math_modeling_agent.analysis_agent import MinCostFlowDraft

    return MinCostFlowDraft(
        nodes=[],
        arcs=[],
        flow_unit="not_applicable",
        cost_unit="not_applicable",
    )


def test_cli_sample_outputs_valid_schedule(capsys) -> None:
    # 延迟导入，让测试在命令行模块尚未实现时仍能被收集
    from math_modeling_agent.cli import main

    exit_code = main(["--sample"])
    output = capsys.readouterr().out
    result = json.loads(output)

    assert exit_code == 0
    assert result["solver_result"]["status"] in {"OPTIMAL", "FEASIBLE"}
    assert result["validation_report"]["is_valid"] is True


def test_cli_json_input_outputs_valid_schedule(capsys, tmp_path) -> None:
    # 创建一个最小的 JSON 排班问题文件
    from math_modeling_agent.cli import main

    problem_data = {
        "employees": [
            {
                "employee_id": "E1",
                "name": "急救员",
                "skills": ["急救"],
                "max_hours": 8,
            }
        ],
        "shifts": [{"shift_id": "S1", "duration_hours": 8}],
        "coverage_requirements": [
            {
                "shift_id": "S1",
                "minimum_employees": 1,
                "required_skill_counts": {"急救": 1},
            }
        ],
    }
    input_path = tmp_path / "排班问题.json"
    input_path.write_text(
        json.dumps(problem_data, ensure_ascii=False),
        encoding="utf-8",
    )

    exit_code = main(["--input", str(input_path)])
    output = capsys.readouterr().out
    result = json.loads(output)

    assert exit_code == 0
    assert result["solver_result"]["status"] in {"OPTIMAL", "FEASIBLE"}
    assert result["validation_report"]["is_valid"] is True


def test_cli_reports_malformed_json(capsys, tmp_path) -> None:
    from math_modeling_agent.cli import main

    input_path = tmp_path / "无效问题.json"
    input_path.write_text("{ 无效 JSON", encoding="utf-8")

    exit_code = main(["--input", str(input_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "JSON" in captured.err
    assert captured.out == ""


def test_cli_request_returns_clarifying_questions(capsys) -> None:
    from types import SimpleNamespace

    from math_modeling_agent.analysis_agent import ProblemAnalysis, SchedulingDraft
    from math_modeling_agent.cli import main

    analysis = ProblemAnalysis(
        status="needs_clarification",
        problem_family="employee_scheduling",
        summary="用户希望排班。",
        known_facts=["需要安排员工"],
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

    class FakeResponses:
        def parse(self, **arguments):
            return SimpleNamespace(output_parsed=analysis)

    client = SimpleNamespace(responses=FakeResponses())
    exit_code = main(
        ["--request", "帮我安排员工班次"],
        llm_client=client,
    )
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["analysis"]["status"] == "needs_clarification"
    assert result["analysis"]["clarifying_questions"] == ["有多少名员工？"]
    assert len(result["draft_hash"]) == 64
    assert result["method_recommendations"] == {}


def test_cli_ready_request_returns_draft_without_solving(capsys, monkeypatch) -> None:
    from types import SimpleNamespace

    from math_modeling_agent.analysis_agent import (
        AnalysisSubtask,
        CoverageDraft,
        EmployeeDraft,
        ProblemAnalysis,
        SchedulingDraft,
        ShiftDraft,
        SkillRequirementDraft,
    )
    from math_modeling_agent import cli

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

    class FakeResponses:
        def parse(self, **arguments):
            return SimpleNamespace(output_parsed=analysis)

    client = SimpleNamespace(responses=FakeResponses())
    def fail_if_solver_runs(*_arguments):
        pytest.fail("自然语言请求默认只返回草稿，不应自动求解")

    monkeypatch.setattr(cli.ModelingService, "solve_confirmed", fail_if_solver_runs)
    exit_code = cli.main(
        ["--request", "林晓上急救班，陈立上普通班。"],
        llm_client=client,
    )
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["analysis"]["status"] == "ready"
    assert len(result["draft_hash"]) == 64
    assert "modeling_run" not in result


def test_cli_solve_confirmed_draft_runs_local_solver(capsys, tmp_path) -> None:
    from math_modeling_agent.cli import main
    from math_modeling_agent.modeling_service import ModelingService
    from min_cost_flow_fixtures import make_ready_analysis

    analysis = make_ready_analysis()
    draft_hash = ModelingService().create_draft_from_analysis(analysis).draft_hash
    draft_path = tmp_path / "confirmed-draft.json"
    draft_path.write_text(
        json.dumps(
            {
                "analysis": analysis.model_dump(mode="json"),
                "method_recommendations": {},
                "draft_hash": draft_hash,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["--solve-draft", str(draft_path)])
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["modeling_run"]["solver_result"]["status"] == "OPTIMAL"
    assert result["modeling_run"]["validation_report"]["is_valid"] is True
    assert result["confirmation"]["state"] == "confirmed_by_cli"
    assert result["modeling_run"]["ir_schema_version"] == "1"
    assert result["modeling_run"]["backend_id"] == "ortools_simple_min_cost_flow"


def test_cli_solve_confirmed_integer_linear_draft_uses_scip(capsys, tmp_path) -> None:
    from math_modeling_agent.analysis_agent import (
        AnalysisSubtask,
        LinearConstraintDraft,
        LinearProgramDraft,
        LinearTermDraft,
        LinearVariableDraft,
        ProblemAnalysis,
        SchedulingDraft,
    )
    from math_modeling_agent import cli
    from math_modeling_agent.modeling_service import ModelingService

    analysis = ProblemAnalysis(
        status="ready",
        problem_family="linear_programming",
        summary="最大化整数产品x的产量。",
        known_facts=["x为非负整数", "产能约束为2x不超过5"],
        missing_information=[],
        clarifying_questions=[],
        unsupported_reasons=[],
        subtasks=[
            AnalysisSubtask(
                task_id="T1",
                description="建立整数线性生产模型。",
                objective="最大化产量。",
                data_needed=[],
                depends_on=[],
                hmml_problem_query="整数变量、线性目标和线性容量约束的生产计划",
                hmml_goal_query="最大化整数产量并满足容量限制",
            )
        ],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
        linear_program_draft=LinearProgramDraft(
            variables=[
                LinearVariableDraft(name="x", unit="件", domain="integer")
            ],
            objective_direction="maximize",
            objective_terms=[LinearTermDraft(variable="x", coefficient=1)],
            constraints=[
                LinearConstraintDraft(
                    constraint_id="capacity",
                    terms=[LinearTermDraft(variable="x", coefficient=2)],
                    relation="<=",
                    rhs=5,
                ),
                LinearConstraintDraft(
                    constraint_id="nonnegative",
                    terms=[LinearTermDraft(variable="x", coefficient=1)],
                    relation=">=",
                    rhs=0,
                ),
            ],
        ),
        minimum_cost_flow_draft=empty_min_cost_flow_draft(),
    )
    draft_path = tmp_path / "integer-draft.json"
    draft_hash = ModelingService().create_draft_from_analysis(analysis).draft_hash
    draft_path.write_text(
        json.dumps(
            {"analysis": analysis.model_dump(mode="json"), "draft_hash": draft_hash},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["--solve-draft", str(draft_path)])
    result = json.loads(capsys.readouterr().out)
    solver_result = result["modeling_run"]["solver_result"]

    assert exit_code == 0
    assert solver_result["status"] == "OPTIMAL"
    assert solver_result["solver_name"] == "SCIP"
    assert solver_result["variable_values"] == pytest.approx({"x": 2})
    assert result["modeling_run"]["validation_report"]["is_valid"] is True


def test_cli_scenario_file_reports_validated_objective_and_variable_deltas(
    capsys,
    tmp_path,
) -> None:
    from math_modeling_agent import cli
    from math_modeling_agent.models import (
        LinearConstraint,
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )
    from math_modeling_agent.scenario_models import (
        LinearScenario,
        ScenarioAnalysisRequest,
    )

    def make_problem(capacity: float, minimum: float = 0) -> LinearProgramProblem:
        return LinearProgramProblem(
            variables=[LinearVariable(name="x", unit="件")],
            objective=LinearObjective(
                direction="maximize",
                terms=[LinearTerm(variable="x", coefficient=3)],
            ),
            constraints=[
                LinearConstraint(
                    name="capacity",
                    terms=[LinearTerm(variable="x", coefficient=1)],
                    relation="<=",
                    rhs=capacity,
                ),
                LinearConstraint(
                    name="nonnegative",
                    terms=[LinearTerm(variable="x", coefficient=1)],
                    relation=">=",
                    rhs=minimum,
                ),
            ],
        )

    request = ScenarioAnalysisRequest(
        base_problem=make_problem(10),
        scenarios=[
            LinearScenario(
                scenario_id="capacity_8",
                description="将产能上限降低到8件",
                problem=make_problem(8),
            ),
            LinearScenario(
                scenario_id="infeasible",
                description="最低产量超过容量",
                problem=make_problem(4, minimum=10),
            ),
        ],
    )
    request_path = tmp_path / "scenarios.json"
    request_path.write_text(
        json.dumps(request.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )

    exit_code = cli.main(["--scenario-file", str(request_path)])
    result = json.loads(capsys.readouterr().out)
    scenario = result["scenario_runs"][0]

    assert exit_code == 0
    assert result["base_run"]["solver_result"]["objective_value"] == pytest.approx(30)
    assert scenario["modeling_run"]["solver_result"]["objective_value"] == pytest.approx(24)
    assert scenario["objective_delta_from_base"] == pytest.approx(-6)
    assert scenario["variable_deltas_from_base"] == pytest.approx({"x": -2})
    infeasible = result["scenario_runs"][1]
    assert infeasible["modeling_run"]["solver_result"]["status"] == "INFEASIBLE"
    assert infeasible["objective_delta_from_base"] is None


def test_cli_scenario_file_rejects_malformed_json(capsys, tmp_path) -> None:
    from math_modeling_agent import cli

    scenario_path = tmp_path / "invalid-scenarios.json"
    scenario_path.write_text("{ invalid", encoding="utf-8")

    exit_code = cli.main(["--scenario-file", str(scenario_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "JSON" in captured.err
    assert captured.out == ""


def test_cli_scenario_file_rejects_invalid_utf8(capsys, tmp_path) -> None:
    from math_modeling_agent import cli

    scenario_path = tmp_path / "invalid-encoding.json"
    scenario_path.write_bytes(b"{\xff}")

    exit_code = cli.main(["--scenario-file", str(scenario_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "UTF-8" in captured.err
    assert captured.out == ""


def test_cli_scenario_file_rejects_unknown_variable_fields(capsys, tmp_path) -> None:
    from math_modeling_agent import cli

    request = {
        "base_problem": {
            "variables": [{"name": "x", "unit": "件", "domian": "integer"}],
            "objective": {
                "direction": "maximize",
                "terms": [{"variable": "x", "coefficient": 1}],
            },
            "constraints": [],
        },
        "scenarios": [
            {
                "scenario_id": "same_typo",
                "description": "变量域拼写错误也不能静默忽略",
                "problem": {
                    "variables": [{"name": "x", "unit": "件", "domian": "integer"}],
                    "objective": {
                        "direction": "maximize",
                        "terms": [{"variable": "x", "coefficient": 1}],
                    },
                    "constraints": [],
                },
            }
        ],
    }
    scenario_path = tmp_path / "unknown-variable-field.json"
    scenario_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")

    exit_code = cli.main(["--scenario-file", str(scenario_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "domian" in captured.err
    assert captured.out == ""


def test_cli_scenario_file_rejects_incompatible_variable_domain(
    capsys,
    tmp_path,
) -> None:
    from math_modeling_agent import cli
    from math_modeling_agent.models import (
        LinearConstraint,
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )
    def make_problem(domain: str) -> LinearProgramProblem:
        return LinearProgramProblem(
            variables=[LinearVariable(name="x", unit="件", domain=domain)],
            objective=LinearObjective(
                direction="maximize",
                terms=[LinearTerm(variable="x", coefficient=1)],
            ),
            constraints=[
                LinearConstraint(
                    name="capacity",
                    terms=[LinearTerm(variable="x", coefficient=1)],
                    relation="<=",
                    rhs=5,
                )
            ],
        )

    raw_request = {
        "base_problem": make_problem("continuous").model_dump(mode="json"),
        "scenarios": [
            {
                "scenario_id": "integer_domain",
                "description": "将x改为整数",
                "problem": make_problem("integer").model_dump(mode="json"),
            }
        ],
    }
    scenario_path = tmp_path / "incompatible-scenarios.json"
    scenario_path.write_text(json.dumps(raw_request, ensure_ascii=False), encoding="utf-8")

    exit_code = cli.main(["--scenario-file", str(scenario_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "变量签名" in captured.err
    assert captured.out == ""


def test_cli_solve_confirmed_draft_rejects_malformed_json(capsys, tmp_path) -> None:
    from math_modeling_agent.cli import main

    draft_path = tmp_path / "invalid-draft.json"
    draft_path.write_text("{ invalid", encoding="utf-8")

    exit_code = main(["--solve-draft", str(draft_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "JSON" in captured.err
    assert captured.out == ""


def test_cli_rejects_draft_changed_after_user_review(capsys, tmp_path, monkeypatch) -> None:
    from math_modeling_agent import cli
    from math_modeling_agent.modeling_service import ModelingService
    from min_cost_flow_fixtures import make_ready_analysis

    analysis = make_ready_analysis()
    reviewed_hash = ModelingService().create_draft_from_analysis(analysis).draft_hash
    changed_analysis = analysis.model_dump(mode="json")
    changed_analysis["summary"] += "（已被修改）"
    draft_path = tmp_path / "changed-after-review.json"
    draft_path.write_text(
        json.dumps(
            {"analysis": changed_analysis, "draft_hash": reviewed_hash},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fail_if_solver_runs(*_arguments):
        pytest.fail("摘要不匹配的草稿不得调用求解器")

    monkeypatch.setattr(cli.ModelingService, "solve_confirmed", fail_if_solver_runs)
    exit_code = cli.main(["--solve-draft", str(draft_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "摘要" in captured.err
    assert captured.out == ""


def test_cli_langchain_framework_returns_reviewable_draft_and_rag_sources(
    capsys,
) -> None:
    from math_modeling_agent.knowledge_models import RetrievedEvidence
    from math_modeling_agent.modeling_service import ModelingService
    from math_modeling_agent.cli import main
    from min_cost_flow_fixtures import make_ready_analysis

    analysis = make_ready_analysis()
    session = ModelingService().create_draft_from_analysis(analysis)

    class FakeLangChainAdapter:
        def create_draft(self, request):
            assert request == "低成本运输"
            return session

        def get_evidence(self, session_id):
            assert session_id == session.session_id
            return [
                RetrievedEvidence(
                    chunk_id="chunk-1",
                    source_id="local-model-validation",
                    source_uri="repo://docs/knowledge/model-validation.md",
                    locator="草稿确认",
                    title="模型确认卡",
                    text="RAG 知识不能替代题目事实。",
                    retrieval_method="keyword",
                    rank=1,
                    problem_families=["minimum_cost_flow"],
                    review_status="approved",
                )
            ]

    exit_code = main(
        ["--request", "低成本运输", "--framework", "langchain"],
        langchain_agent=FakeLangChainAdapter(),
    )
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["framework"] == "langchain"
    assert len(result["draft_hash"]) == 64
    assert result["knowledge_evidence"][0]["source_id"] == "local-model-validation"
    assert "modeling_run" not in result


def test_cli_rejects_langchain_framework_without_natural_language_request(capsys) -> None:
    from math_modeling_agent.cli import main

    exit_code = main(["--sample", "--framework", "langchain"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "仅适用于 --request" in captured.err
    assert captured.out == ""


def test_cli_ready_linear_program_request_returns_draft_without_solving(capsys, monkeypatch) -> None:
    import pytest
    from types import SimpleNamespace

    from math_modeling_agent.analysis_agent import (
        AnalysisSubtask,
        LinearConstraintDraft,
        LinearProgramDraft,
        LinearTermDraft,
        LinearVariableDraft,
        ProblemAnalysis,
        SchedulingDraft,
    )
    from math_modeling_agent import cli

    analysis = ProblemAnalysis(
        status="ready",
        problem_family="linear_programming",
        summary="最大化两种产品的利润。",
        known_facts=["利润与资源限制已知"],
        missing_information=[],
        clarifying_questions=[],
        unsupported_reasons=[],
        subtasks=[
            AnalysisSubtask(
                task_id="T1",
                description="建立生产计划线性规划。",
                objective="最大化利润。",
                data_needed=[],
                depends_on=[],
                hmml_problem_query="连续变量的生产计划线性规划",
                hmml_goal_query="最大化利润并满足资源约束",
            )
        ],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
        linear_program_draft=LinearProgramDraft(
            variables=[
                LinearVariableDraft(name="A", unit="件", domain="continuous"),
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

    class FakeResponses:
        def parse(self, **arguments):
            return SimpleNamespace(output_parsed=analysis)

    def fail_if_solver_runs(*_arguments):
        pytest.fail("自然语言请求默认只返回 LP 草稿，不应自动求解")

    monkeypatch.setattr(cli.ModelingService, "solve_confirmed", fail_if_solver_runs)
    exit_code = cli.main(
        ["--request", "生产A、B两种产品，求利润最大化。"],
        llm_client=SimpleNamespace(responses=FakeResponses()),
    )
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["analysis"]["problem_family"] == "linear_programming"
    assert result["analysis"]["linear_program_draft"]["variables"]
    assert "modeling_run" not in result


def test_cli_integer_draft_recommends_scip_before_user_confirmation(
    capsys,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    import pytest

    from math_modeling_agent import cli
    from math_modeling_agent.analysis_agent import (
        AnalysisSubtask,
        LinearConstraintDraft,
        LinearProgramDraft,
        LinearTermDraft,
        LinearVariableDraft,
        ProblemAnalysis,
        SchedulingDraft,
    )

    analysis = ProblemAnalysis(
        status="ready",
        problem_family="linear_programming",
        summary="最大化整数产品x的产量。",
        known_facts=["x为非负整数", "资源约束为2x不超过5"],
        missing_information=[],
        clarifying_questions=[],
        unsupported_reasons=[],
        subtasks=[
            AnalysisSubtask(
                task_id="T1",
                description="建立整数线性生产模型。",
                objective="最大化产量。",
                data_needed=[],
                depends_on=[],
                hmml_problem_query="生产计划与资源约束",
                hmml_goal_query="最大化产量",
            )
        ],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
        linear_program_draft=LinearProgramDraft(
            variables=[
                LinearVariableDraft(name="x", unit="件", domain="integer")
            ],
            objective_direction="maximize",
            objective_terms=[LinearTermDraft(variable="x", coefficient=1)],
            constraints=[
                LinearConstraintDraft(
                    constraint_id="capacity",
                    terms=[LinearTermDraft(variable="x", coefficient=2)],
                    relation="<=",
                    rhs=5,
                ),
                LinearConstraintDraft(
                    constraint_id="nonnegative",
                    terms=[LinearTermDraft(variable="x", coefficient=1)],
                    relation=">=",
                    rhs=0,
                ),
            ],
        ),
        minimum_cost_flow_draft=empty_min_cost_flow_draft(),
    )

    class FakeResponses:
        def parse(self, **arguments):
            return SimpleNamespace(output_parsed=analysis)

    def fail_if_solver_runs(*_arguments):
        pytest.fail("--request 只生成草稿，不应启动求解器")

    monkeypatch.setattr(cli.ModelingService, "solve_confirmed", fail_if_solver_runs)
    exit_code = cli.main(
        ["--request", "整数产品x，最大化产量，约束2x不超过5。"],
        llm_client=SimpleNamespace(responses=FakeResponses()),
    )
    result = json.loads(capsys.readouterr().out)
    methods = result["method_recommendations"]["T1"]

    assert exit_code == 0
    assert methods[0]["method_id"] == "integer_programming"
    assert methods[0]["solver"] == "OR-Tools SCIP"
    assert all(item["method_id"] != "continuous_linear_programming" for item in methods)
    assert "modeling_run" not in result


def test_cli_ready_min_cost_flow_request_returns_draft_without_solving(capsys, monkeypatch) -> None:
    from types import SimpleNamespace

    from math_modeling_agent import cli
    from min_cost_flow_fixtures import make_ready_analysis

    analysis = make_ready_analysis()

    class FakeResponses:
        def parse(self, **arguments):
            return SimpleNamespace(output_parsed=analysis)

    def fail_if_solver_runs(*_arguments):
        pytest.fail("自然语言请求默认只返回网络流草稿，不应自动求解")

    monkeypatch.setattr(cli.ModelingService, "solve_confirmed", fail_if_solver_runs)
    exit_code = cli.main(
        ["--request", "以最低费用将两仓货物送至两家门店。"],
        llm_client=SimpleNamespace(responses=FakeResponses()),
    )
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["analysis"]["problem_family"] == "minimum_cost_flow"
    assert result["analysis"]["minimum_cost_flow_draft"]["nodes"]
    assert "modeling_run" not in result


def test_cli_does_not_solve_nonready_min_cost_flow_request(capsys, monkeypatch) -> None:
    from types import SimpleNamespace

    import pytest

    from math_modeling_agent.analysis_agent import ProblemAnalysis
    from math_modeling_agent import cli
    from min_cost_flow_fixtures import make_ready_analysis

    for status in ("needs_clarification", "unsupported"):
        analysis_data = make_ready_analysis().model_dump(mode="python")
        analysis_data["status"] = status
        analysis_data["subtasks"] = []
        analysis_data["minimum_cost_flow_draft"] = {
            "nodes": [],
            "arcs": [],
            "flow_unit": "not_applicable",
            "cost_unit": "not_applicable",
        }
        if status == "needs_clarification":
            analysis_data["missing_information"] = ["路线单位费用"]
            analysis_data["clarifying_questions"] = ["请提供各条路线的单位费用。"]
        else:
            analysis_data["unsupported_reasons"] = ["当前版本不支持多商品流。"]
        analysis = ProblemAnalysis.model_validate(analysis_data)

        def fail_if_called(*_arguments):
            pytest.fail("非 ready 网络流问题不得启动求解器")

        monkeypatch.setattr(cli.ModelingService, "solve_confirmed", fail_if_called)

        class FakeResponses:
            def parse(self, **arguments):
                return SimpleNamespace(output_parsed=analysis)

        exit_code = cli.main(
            ["--request", "帮我做网络流运输分析"],
            llm_client=SimpleNamespace(responses=FakeResponses()),
        )
        result = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert result["analysis"]["status"] == status
        assert "modeling_run" not in result


def test_cli_energy_park_q1_outputs_balanced_results(capsys, monkeypatch) -> None:
    from math_modeling_agent import cli
    from math_modeling_agent.energy_park_models import (
        EnergyParkCostParameters,
        EnergyParkDataset,
        HourlyProfile,
    )

    periods = [f"{hour}:00-{hour + 1}:00" for hour in range(24)]

    def profile(values):
        return HourlyProfile(periods=periods, values=values)

    dataset = EnergyParkDataset(
        ordinary_load=profile([1 / 6] * 24),
        typical_wind=profile([0.0] * 12 + [0.5] * 12),
        typical_pv=profile([0.0] * 12 + [0.25] * 12),
        wind_scenarios=[profile([0.5] * 24) for _ in range(6)],
        pv_scenarios=[profile([0.25] * 24) for _ in range(4)],
        costs=EnergyParkCostParameters(
            wind_lcoe_yuan_per_kwh=0.15,
            pv_lcoe_yuan_per_kwh=0.12,
            alkaline_om_yuan_per_kwh=0.1,
            pem_om_yuan_per_kwh=0.15,
            alkaline_lifetime_years=30,
            pem_lifetime_years=30,
            alkaline_efficiency_percent=70,
            pem_efficiency_percent=80,
            hydrogen_energy_kwh_per_kg=50,
            storage_capex_yuan_per_kwh=1000,
            storage_om_yuan_per_kwh=0.01,
            storage_lifetime_years=15,
            storage_charge_efficiency_percent=90,
            storage_discharge_efficiency_percent=90,
            storage_self_loss_percent=0.2,
            ammonia_capex_yuan_per_kg_h2=60000,
            ammonia_om_yuan_per_kwh=0.002,
            ammonia_lifetime_years=30,
            ammonia_energy_kwh_per_kg=0.5,
            ammonia_hydrogen_kg_per_kg=0.2,
            purchase_price_peak_yuan_per_kwh=0.8024,
            purchase_price_flat_yuan_per_kwh=0.6074,
            purchase_price_valley_yuan_per_kwh=0.3424,
            wind_export_price_yuan_per_kwh=0.3779,
            pv_export_price_yuan_per_kwh=0.3779,
        ),
        source_files=["附件1.xlsx", "附件2.xlsx"],
    )
    monkeypatch.setattr(cli, "load_energy_park_directory", lambda _path: dataset, raising=False)

    exit_code = cli.main(["--energy-park-q1", "D:\\例题\\电工杯A"])
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["total_load_mwh"] == pytest.approx(522)
    assert result["grid_purchase_mwh"] == pytest.approx(261)
    assert result["validation_report"]["is_valid"] is True
    assert result["source_files"] == ["附件1.xlsx", "附件2.xlsx"]
    assert len(result["costs"]["sensitivity_analysis"]) == 11


def test_cli_energy_park_q2_runs_all_discrete_targets(capsys, monkeypatch) -> None:
    from math_modeling_agent import cli
    from test_energy_park_discrete import _dataset_with_sunny_half_day

    monkeypatch.setattr(
        cli,
        "load_energy_park_directory",
        lambda _path: _dataset_with_sunny_half_day(),
        raising=False,
    )

    exit_code = cli.main(["--energy-park-q2", "D:\\例题\\电工杯A"])
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["target_levels_tons_per_day"] == [72.0, 63.0, 54.0, 45.0, 36.0]
    assert len(result["scenario_runs"]) == 120
    assert len(result["annual_summaries"]) == 5
    assert (
        result["typical_runs"][0]["operation"]["costs"]["sensitivity_analysis"]
        == []
    )


def test_cli_energy_park_q3_returns_continuous_and_discrete_comparison(
    capsys,
    monkeypatch,
) -> None:
    from math_modeling_agent import cli
    from test_energy_park_discrete import _dataset_with_sunny_half_day

    monkeypatch.setattr(
        cli,
        "load_energy_park_directory",
        lambda _path: _dataset_with_sunny_half_day(),
        raising=False,
    )

    exit_code = cli.main(["--energy-park-q3", "D:\\例题\\电工杯A"])
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(result["scenario_runs"]) == 120
    assert len(result["discrete_comparison_by_target"]) == 5
    assert result["modeling_assumptions"]
    assert (
        result["typical_runs"][0]["operation"]["costs"]["sensitivity_analysis"]
        == []
    )


def test_cli_energy_park_q5_returns_cited_policy_report(capsys, monkeypatch) -> None:
    from math_modeling_agent import cli
    from test_energy_park_discrete import _dataset_with_sunny_half_day

    monkeypatch.setattr(
        cli,
        "load_energy_park_directory",
        lambda _path: _dataset_with_sunny_half_day(),
        raising=False,
    )

    exit_code = cli.main(["--energy-park-q5", "D:\\例题\\电工杯A"])
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    report = result["policy_analysis"]
    assert len(report["benefits"]) >= 3
    assert len(report["risks"]) >= 3
    assert len(report["recommendations"]) >= 3
    assert report["sources"]
