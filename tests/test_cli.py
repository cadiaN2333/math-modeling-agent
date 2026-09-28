import json


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
        summary="用户希望排班。",
        known_facts=["需要安排员工"],
        missing_information=["员工人数"],
        clarifying_questions=["有多少名员工？"],
        unsupported_reasons=[],
        subtasks=[],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
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
    assert result["method_recommendations"] == {}


def test_cli_ready_request_runs_local_solver(capsys) -> None:
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
    from math_modeling_agent.cli import main

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

    class FakeResponses:
        def parse(self, **arguments):
            return SimpleNamespace(output_parsed=analysis)

    client = SimpleNamespace(responses=FakeResponses())
    exit_code = main(
        ["--request", "林晓上急救班，陈立上普通班。"],
        llm_client=client,
    )
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["analysis"]["status"] == "ready"
    assert result["modeling_run"]["solver_result"]["status"] in {
        "OPTIMAL",
        "FEASIBLE",
    }
    assert result["modeling_run"]["validation_report"]["is_valid"] is True
