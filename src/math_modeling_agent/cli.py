"""通过命令行运行排班建模示例。"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from pydantic import ValidationError

from .analysis_agent import (
    analyze_problem,
    to_linear_program_problem,
    to_minimum_cost_flow_problem,
    retrieve_methods_for_subtasks,
    to_scheduling_problem,
    ProblemAnalysis,
)
from .agent import run_modeling
from .energy_park import compute_typical_day
from .energy_park_continuous import run_continuous_question_three
from .energy_park_discrete import run_discrete_question_two
from .energy_park_io import load_energy_park_directory
from .energy_park_policy import build_energy_park_policy_report
from .energy_park_validator import validate_typical_day
from .linear_agent import run_linear_modeling
from .min_cost_flow_agent import run_min_cost_flow_modeling
from .problem_io import load_problem_file
from .sample_data import make_sample_problem
from .scenario_analysis import run_scenario_analysis
from .scenario_models import ScenarioAnalysisRequest


def _escape_unencodable_json_characters(text: str, encoding: str) -> str:
    """将终端无法编码的 JSON 字符转成合法的 Unicode 转义序列。"""

    escaped_parts: list[str] = []
    for character in text:
        try:
            character.encode(encoding)
        except UnicodeEncodeError:
            escaped_character = json.dumps(character, ensure_ascii=True)
            escaped_parts.append(escaped_character[1:-1])
        else:
            escaped_parts.append(character)
    return "".join(escaped_parts)


def _print_json(payload: object) -> None:
    """按当前终端编码安全输出 JSON，必要时转义无法编码的字符。"""

    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    encoding = sys.stdout.encoding or "utf-8"
    try:
        serialized.encode(encoding)
    except UnicodeEncodeError:
        serialized = _escape_unencodable_json_characters(serialized, encoding)
    print(serialized)


def _run_confirmed_analysis(analysis: ProblemAnalysis):
    """按经校验的领域草稿选择本地求解器。"""

    if analysis.status != "ready":
        raise ValueError("只有 ready 状态的结构化草稿可以进入求解")
    if analysis.problem_family == "employee_scheduling":
        return run_modeling(to_scheduling_problem(analysis))
    if analysis.problem_family == "linear_programming":
        return run_linear_modeling(to_linear_program_problem(analysis))
    if analysis.problem_family == "minimum_cost_flow":
        return run_min_cost_flow_modeling(
            to_minimum_cost_flow_problem(analysis)
        )
    raise ValueError("当前问题领域没有已实现的求解适配器")


def _modeling_run_is_valid(modeling_run_data: dict) -> bool:
    """统一判断求解状态和独立 validator 结果。"""

    solver_result = modeling_run_data["solver_result"]
    validation_report = modeling_run_data["validation_report"]
    return (
        solver_result["status"] in {"OPTIMAL", "FEASIBLE"}
        and validation_report is not None
        and validation_report["is_valid"]
    )


def _scenario_modeling_run_is_reportable(modeling_run_data: dict) -> bool:
    """允许将无解/无界作为有效情景结论，但拒绝未完成或无效结果。"""

    status = modeling_run_data["solver_result"]["status"]
    if status in {"OPTIMAL", "FEASIBLE"}:
        validation_report = modeling_run_data["validation_report"]
        return validation_report is not None and validation_report["is_valid"]
    return status in {"INFEASIBLE", "UNBOUNDED"}


def main(argv: list[str] | None = None, *, llm_client=None) -> int:
    """运行样例、JSON 输入或 DeepSeek 自然语言分析。"""

    parser = argparse.ArgumentParser(description="可验证的运筹优化数学建模 Agent")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sample",
        action="store_true",
        help="运行内置的排班样例",
    )
    input_group.add_argument(
        "--input",
        type=Path,
        metavar="文件路径",
        help="从 JSON 文件读取排班问题",
    )
    input_group.add_argument(
        "--request",
        metavar="自然语言描述",
        help="用 DeepSeek 分析中文问题，并按子任务检索 HMML",
    )
    input_group.add_argument(
        "--energy-park-q1",
        type=Path,
        metavar="附件目录",
        help="读取电工杯 A 题附件，计算问题一典型日基准",
    )
    input_group.add_argument(
        "--energy-park-q2",
        type=Path,
        metavar="附件目录",
        help="读取电工杯 A 题附件，计算问题二离散开停调度",
    )
    input_group.add_argument(
        "--energy-park-q3",
        type=Path,
        metavar="附件目录",
        help="读取电工杯 A 题附件，计算问题三连续功率调度",
    )
    input_group.add_argument(
        "--energy-park-q5",
        type=Path,
        metavar="附件目录",
        help="基于问题二、三运行结果生成带来源的问题五分析",
    )
    input_group.add_argument(
        "--solve-draft",
        type=Path,
        metavar="JSON文件",
        help="求解经过用户审阅的结构化分析 JSON",
    )
    input_group.add_argument(
        "--scenario-file",
        type=Path,
        metavar="JSON文件",
        help="重算基准线性模型与多个命名情景并比较结果",
    )
    args = parser.parse_args(argv)

    if args.scenario_file is not None:
        try:
            document = json.loads(args.scenario_file.read_text(encoding="utf-8-sig"))
        except OSError as exc:
            print(f"无法读取情景文件：{exc}", file=sys.stderr)
            return 2
        except json.JSONDecodeError as exc:
            print(f"情景 JSON 格式错误：{exc}", file=sys.stderr)
            return 2

        try:
            request = ScenarioAnalysisRequest.model_validate(document)
            result = run_scenario_analysis(request)
        except ValidationError as exc:
            print(f"情景模型校验失败：{exc}", file=sys.stderr)
            return 2
        except (RuntimeError, ValueError) as exc:
            print(f"情景分析失败：{exc}", file=sys.stderr)
            return 1

        payload = asdict(result)
        _print_json(payload)
        modeling_runs = [
            payload["base_run"],
            *(scenario["modeling_run"] for scenario in payload["scenario_runs"]),
        ]
        return 0 if all(
            _scenario_modeling_run_is_reportable(run) for run in modeling_runs
        ) else 1

    if args.energy_park_q5 is not None:
        try:
            dataset = load_energy_park_directory(args.energy_park_q5)
            discrete_result = run_discrete_question_two(dataset)
            continuous_result = run_continuous_question_three(
                dataset,
                discrete_result=discrete_result,
            )
            policy_report = build_energy_park_policy_report(
                discrete_result,
                continuous_result,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"能源园区问题五分析失败：{exc}", file=sys.stderr)
            return 2

        payload = {
            "problem2_summary": {
                "best_typical_target_tons_per_day": (
                    discrete_result.best_typical_target_tons_per_day
                ),
                "best_annual_cost_per_ton_target_tons_per_day": (
                    discrete_result.best_annual_cost_per_ton_target_tons_per_day
                ),
                "annual_summaries": [
                    summary.model_dump(mode="json")
                    for summary in discrete_result.annual_summaries
                ],
            },
            "problem3_summary": {
                "best_typical_target_tons_per_day": (
                    continuous_result.best_typical_target_tons_per_day
                ),
                "best_annual_cost_per_ton_target_tons_per_day": (
                    continuous_result.best_annual_cost_per_ton_target_tons_per_day
                ),
                "annual_summaries": [
                    summary.model_dump(mode="json")
                    for summary in continuous_result.annual_summaries
                ],
                "discrete_comparison_by_target": (
                    continuous_result.discrete_comparison_by_target
                ),
            },
            "policy_analysis": policy_report.model_dump(mode="json"),
        }
        _print_json(payload)
        return 0

    if args.energy_park_q3 is not None:
        try:
            dataset = load_energy_park_directory(args.energy_park_q3)
            result = run_continuous_question_three(dataset)
        except (RuntimeError, ValueError) as exc:
            print(f"能源园区问题三计算失败：{exc}", file=sys.stderr)
            return 2

        payload = result.model_dump(mode="json")
        _print_json(payload)
        all_valid = all(
            run.solver_status == "OPTIMAL"
            and run.operation is not None
            and run.validation_report is not None
            and run.validation_report.is_valid
            for run in [*result.typical_runs, *result.scenario_runs]
        )
        return 0 if all_valid else 1

    if args.energy_park_q2 is not None:
        try:
            dataset = load_energy_park_directory(args.energy_park_q2)
            result = run_discrete_question_two(dataset)
        except (RuntimeError, ValueError) as exc:
            print(f"能源园区问题二计算失败：{exc}", file=sys.stderr)
            return 2

        payload = result.model_dump(mode="json")
        _print_json(payload)
        all_valid = all(
            run.solver_status == "OPTIMAL"
            and run.operation is not None
            and run.validation_report is not None
            and run.validation_report.is_valid
            for run in [*result.typical_runs, *result.scenario_runs]
        )
        return 0 if all_valid else 1

    if args.energy_park_q1 is not None:
        try:
            dataset = load_energy_park_directory(args.energy_park_q1)
            result = compute_typical_day(dataset)
        except (RuntimeError, ValueError) as exc:
            print(f"能源园区问题一计算失败：{exc}", file=sys.stderr)
            return 2

        validation_report = validate_typical_day(dataset, result)
        payload = result.model_dump(mode="json")
        payload["validation_report"] = validation_report.model_dump(mode="json")
        _print_json(payload)
        return 0 if validation_report.is_valid else 1

    if args.solve_draft is not None:
        try:
            document = json.loads(args.solve_draft.read_text(encoding="utf-8-sig"))
        except OSError as exc:
            print(f"无法读取已审阅草稿：{exc}", file=sys.stderr)
            return 2
        except json.JSONDecodeError as exc:
            print(f"草稿 JSON 格式错误：{exc}", file=sys.stderr)
            return 2

        try:
            if not isinstance(document, dict) or not isinstance(
                document.get("analysis"), dict
            ):
                raise ValueError("草稿文件必须包含 analysis JSON 对象")
            analysis = ProblemAnalysis.model_validate(document["analysis"])
            modeling_run = _run_confirmed_analysis(analysis)
        except (ValidationError, ValueError) as exc:
            print(f"已审阅草稿校验失败：{exc}", file=sys.stderr)
            return 2

        recommendations = retrieve_methods_for_subtasks(analysis)
        payload = {
            "analysis": analysis.model_dump(mode="json"),
            "method_recommendations": {
                task_id: [asdict(item) for item in items]
                for task_id, items in recommendations.items()
            },
            "confirmation": {"state": "confirmed_by_cli"},
            "modeling_run": asdict(modeling_run),
        }
        _print_json(payload)
        return 0 if _modeling_run_is_valid(payload["modeling_run"]) else 1

    if args.request is not None:
        try:
            analysis = analyze_problem(args.request, client=llm_client)
        except (RuntimeError, ValueError) as exc:
            print(f"自然语言分析失败：{exc}", file=sys.stderr)
            return 2

        recommendations = retrieve_methods_for_subtasks(analysis)
        payload = {
            "analysis": analysis.model_dump(mode="json"),
            "method_recommendations": {
                task_id: [asdict(item) for item in items]
                for task_id, items in recommendations.items()
            },
            "workflow_state": (
                "awaiting_user_review" if analysis.status == "ready" else analysis.status
            ),
        }
        _print_json(payload)
        return 0

    try:
        # 根据命令行参数选择内置样例或用户提供的 JSON 文件
        problem = (
            make_sample_problem()
            if args.sample
            else load_problem_file(args.input)
        )
    except ValueError as exc:
        print(f"输入错误：{exc}", file=sys.stderr)
        return 2

    # 执行求解和独立验证，并将 dataclass 转成可序列化字典
    result = run_modeling(problem)
    _print_json(asdict(result))

    # 只有找到可行排班并通过验证时，命令才以成功状态结束
    is_valid = (
        result.solver_result.status in {"OPTIMAL", "FEASIBLE"}
        and result.validation_report is not None
        and result.validation_report.is_valid
    )
    return 0 if is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
