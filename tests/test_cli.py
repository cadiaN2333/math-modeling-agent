import json
import pytest


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
    def fail_if_solver_runs(_problem):
        pytest.fail("自然语言请求默认只返回草稿，不应自动求解")

    monkeypatch.setattr(cli, "run_modeling", fail_if_solver_runs)
    exit_code = cli.main(
        ["--request", "林晓上急救班，陈立上普通班。"],
        llm_client=client,
    )
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["analysis"]["status"] == "ready"
    assert "modeling_run" not in result


def test_cli_solve_confirmed_draft_runs_local_solver(capsys, tmp_path) -> None:
    from math_modeling_agent.cli import main
    from min_cost_flow_fixtures import make_ready_analysis

    draft_path = tmp_path / "confirmed-draft.json"
    draft_path.write_text(
        json.dumps(
            {
                "analysis": make_ready_analysis().model_dump(mode="json"),
                "method_recommendations": {},
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


def test_cli_solve_confirmed_draft_rejects_malformed_json(capsys, tmp_path) -> None:
    from math_modeling_agent.cli import main

    draft_path = tmp_path / "invalid-draft.json"
    draft_path.write_text("{ invalid", encoding="utf-8")

    exit_code = main(["--solve-draft", str(draft_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "JSON" in captured.err
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
                LinearVariableDraft(name="A", unit="件"),
                LinearVariableDraft(name="B", unit="件"),
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

    def fail_if_solver_runs(_problem):
        pytest.fail("自然语言请求默认只返回 LP 草稿，不应自动求解")

    monkeypatch.setattr(cli, "run_linear_modeling", fail_if_solver_runs)
    exit_code = cli.main(
        ["--request", "生产A、B两种产品，求利润最大化。"],
        llm_client=SimpleNamespace(responses=FakeResponses()),
    )
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["analysis"]["problem_family"] == "linear_programming"
    assert result["analysis"]["linear_program_draft"]["variables"]
    assert "modeling_run" not in result


def test_cli_ready_min_cost_flow_request_returns_draft_without_solving(capsys, monkeypatch) -> None:
    from types import SimpleNamespace

    from math_modeling_agent import cli
    from min_cost_flow_fixtures import make_ready_analysis

    analysis = make_ready_analysis()

    class FakeResponses:
        def parse(self, **arguments):
            return SimpleNamespace(output_parsed=analysis)

    def fail_if_solver_runs(_problem):
        pytest.fail("自然语言请求默认只返回网络流草稿，不应自动求解")

    monkeypatch.setattr(cli, "run_min_cost_flow_modeling", fail_if_solver_runs)
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

        def fail_if_called(problem):
            pytest.fail("非 ready 网络流问题不得启动求解器")

        monkeypatch.setattr(cli, "run_min_cost_flow_modeling", fail_if_called)

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
