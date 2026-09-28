"""排班 MVP 的案例加载、离线评分和可选在线 Evals 命令。"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .analysis_agent import analyze_problem, to_scheduling_problem
from .agent import ModelingRun, run_modeling


ANALYSIS_STATUSES = {"ready", "needs_clarification", "unsupported"}
SOLVER_STATUSES = {"OPTIMAL", "FEASIBLE", "INFEASIBLE", "MODEL_INVALID", "UNKNOWN"}
ANALYSIS_TEXT_FIELDS = {
    "summary",
    "known_facts",
    "missing_information",
    "clarifying_questions",
    "unsupported_reasons",
}


def load_eval_cases(path: Path | None = None) -> list[dict[str, Any]]:
    """读取并检查版本化 Evals 案例文件。"""

    cases_path = path or Path(__file__).resolve().parents[2] / "evals" / "cases.json"
    try:
        dataset = json.loads(cases_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 Evals 案例文件：{cases_path}") from exc

    if not isinstance(dataset, dict) or dataset.get("version") != 1:
        raise ValueError("Evals 案例文件必须是 version=1 的 JSON 对象")

    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evals 案例文件必须包含非空 cases 数组")

    case_ids: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("每个 Evals 案例都必须是 JSON 对象")

        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("每个 Evals 案例都必须有非空 id")
        case_ids.append(case_id)

        if not isinstance(case.get("family"), str) or not case["family"].strip():
            raise ValueError(f"案例 {case_id} 缺少 family")
        if not isinstance(case.get("request"), str) or not case["request"].strip():
            raise ValueError(f"案例 {case_id} 缺少 request")
        if case.get("expected_analysis_status") not in ANALYSIS_STATUSES:
            raise ValueError(f"案例 {case_id} 的 expected_analysis_status 无效")

        for group in case.get("required_text_groups", []):
            fields = group.get("fields")
            terms = group.get("any_of")
            if not isinstance(fields, list) or not fields or not set(fields) <= ANALYSIS_TEXT_FIELDS:
                raise ValueError(f"案例 {case_id} 的 required_text_groups.fields 无效")
            if not isinstance(terms, list) or not terms or not all(
                isinstance(term, str) and term for term in terms
            ):
                raise ValueError(f"案例 {case_id} 的 required_text_groups.any_of 无效")

        if case["expected_analysis_status"] == "ready":
            expected_solver_statuses = case.get("expected_solver_statuses")
            if (
                not isinstance(expected_solver_statuses, list)
                or not expected_solver_statuses
                or not set(expected_solver_statuses) <= SOLVER_STATUSES
            ):
                raise ValueError(f"ready 案例 {case_id} 必须声明有效求解状态")
            if not case.get("expected_method_id"):
                raise ValueError(f"ready 案例 {case_id} 必须声明预期方法")
            if case.get("expected_validation") not in {"valid", "not_run"}:
                raise ValueError(f"ready 案例 {case_id} 必须声明 validator 期望")
        elif "expected_solver_statuses" in case:
            raise ValueError(f"非 ready 案例 {case_id} 不能要求求解状态")

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evals 案例 id 不能重复")

    return cases


def _normalize_scheduling_draft(draft: Any) -> Any:
    """按稳定 ID 和技能名排序，避免列表顺序差异导致误报。"""

    if not isinstance(draft, dict):
        return draft

    normalized = dict(draft)
    employees = []
    for employee in draft.get("employees", []):
        item = dict(employee)
        item["skills"] = sorted(item.get("skills", []))
        employees.append(item)
    normalized["employees"] = sorted(
        employees,
        key=lambda item: item.get("employee_id", ""),
    )

    normalized["shifts"] = sorted(
        (dict(shift) for shift in draft.get("shifts", [])),
        key=lambda item: item.get("shift_id", ""),
    )

    coverage_requirements = []
    for requirement in draft.get("coverage_requirements", []):
        item = dict(requirement)
        item["skill_requirements"] = sorted(
            (dict(skill) for skill in item.get("skill_requirements", [])),
            key=lambda skill: (skill.get("skill", ""), skill.get("minimum_count", 0)),
        )
        coverage_requirements.append(item)
    normalized["coverage_requirements"] = sorted(
        coverage_requirements,
        key=lambda item: item.get("shift_id", ""),
    )

    return normalized


def evaluate_case(case: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """根据案例期望对分析结果及可选建模结果逐项评分。"""

    analysis = payload["analysis"]
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, expected: Any, actual: Any) -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "expected": expected, "actual": actual}
        )

    actual_status = analysis.get("status")
    expected_status = case["expected_analysis_status"]
    record("analysis_status", actual_status == expected_status, expected_status, actual_status)

    for index, group in enumerate(case.get("required_text_groups", [])):
        text_fields: list[Any] = []
        for field_name in group["fields"]:
            value = analysis.get(field_name, "")
            text_fields.extend(value if isinstance(value, list) else [value])
        searchable_text = " ".join(str(item) for item in text_fields).casefold()
        alternatives = group["any_of"]
        matched = any(term.casefold() in searchable_text for term in alternatives)
        record(f"required_text_{index + 1}", matched, alternatives, matched)

    expected_model = case.get("expected_model")
    if expected_model is not None:
        actual_model = analysis.get("scheduling_draft")
        normalized_expected = _normalize_scheduling_draft(expected_model)
        normalized_actual = _normalize_scheduling_draft(actual_model)
        record(
            "structured_model",
            normalized_actual == normalized_expected,
            normalized_expected,
            normalized_actual,
        )

    modeling_run = payload.get("modeling_run")
    expected_solver_statuses = case.get("expected_solver_statuses")
    if expected_solver_statuses is None:
        record(
            "solver_not_run",
            modeling_run is None,
            "not_run",
            "not_run" if modeling_run is None else "run",
        )
    else:
        solver_result = (modeling_run or {}).get("solver_result") or {}
        actual_solver_status = solver_result.get("status")
        record(
            "solver_status",
            actual_solver_status in expected_solver_statuses,
            expected_solver_statuses,
            actual_solver_status,
        )

        expected_method_id = case.get("expected_method_id")
        if expected_method_id:
            method_recommendations = (modeling_run or {}).get(
                "method_recommendations", []
            )
            implemented_method_ids = {
                item.get("method_id")
                for item in method_recommendations
                if item.get("implementation_status") == "已实现"
            }
            record(
                "implemented_method",
                expected_method_id in implemented_method_ids,
                expected_method_id,
                sorted(implemented_method_ids),
            )

        expected_validation = case.get("expected_validation")
        validation_report = (modeling_run or {}).get("validation_report")
        if expected_validation == "valid":
            is_valid = bool(validation_report and validation_report.get("is_valid"))
            record("validator", is_valid, True, validation_report)
        elif expected_validation == "not_run":
            record("validator", validation_report is None, "not_run", validation_report)

        required_pairs = {
            tuple(pair) for pair in case.get("required_assignment_pairs", [])
        }
        if required_pairs:
            actual_pairs = {
                (item.get("employee_id"), item.get("shift_id"))
                for item in solver_result.get("assignments", [])
            }
            missing_pairs = sorted(required_pairs - actual_pairs)
            record("required_assignments", not missing_pairs, sorted(required_pairs), missing_pairs)

    return {
        "case_id": case["id"],
        "family": case["family"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _error_result(case: dict[str, Any], error: Exception) -> dict[str, Any]:
    """记录错误类型，不输出可能含有凭据的异常原文。"""

    return {
        "case_id": case["id"],
        "family": case["family"],
        "passed": False,
        "error_type": type(error).__name__,
        "checks": [
            {
                "name": "evaluation_run",
                "passed": False,
                "expected": "completed",
                "actual": type(error).__name__,
            }
        ],
    }


def _check_rate(results: list[dict[str, Any]], check_name: str) -> float | None:
    """计算指定检查项的通过率；没有适用案例时返回 None。"""

    checks = [
        check
        for result in results
        for check in result.get("checks", [])
        if check["name"] == check_name
    ]
    if not checks:
        return None
    return round(sum(check["passed"] for check in checks) / len(checks), 3)


def _build_report(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """整理逐例结果和分层指标。"""

    count = len(results)
    passed_count = sum(result["passed"] for result in results)
    valid_case_ids = {
        case["id"]
        for case in cases
        if case.get("expected_validation") == "valid"
    }
    validator_results = [
        result
        for result in results
        if result["case_id"] in valid_case_ids
    ]
    validator_passed = sum(
        any(
            check["name"] == "validator" and check["passed"]
            for check in result.get("checks", [])
        )
        for result in validator_results
    )

    return {
        "mode": "live",
        "scope": "排班 MVP 与当前能力边界，不代表通用数学建模能力",
        "api_calls": count,
        "case_count": count,
        "passed_count": passed_count,
        "error_count": sum("error_type" in result for result in results),
        "pass_rate": round(passed_count / count, 3) if count else 0.0,
        "analysis_status_accuracy": _check_rate(results, "analysis_status"),
        "solver_status_accuracy": _check_rate(results, "solver_status"),
        "implemented_method_hit_rate": _check_rate(results, "implemented_method"),
        "validator_pass_rate": (
            round(validator_passed / len(validator_results), 3)
            if validator_results
            else None
        ),
        "results": results,
    }


def main(
    argv: list[str] | None = None,
    *,
    analyzer: Callable[[str], Any] | None = None,
) -> int:
    """默认只校验案例集；只有显式传入 --live 才调用 DeepSeek。"""

    parser = argparse.ArgumentParser(description="排班 MVP 分层 Evals")
    parser.add_argument(
        "--live",
        action="store_true",
        help="逐个调用 DeepSeek 运行在线评测；会消耗 API 配额",
    )
    parser.add_argument(
        "--case-id",
        help="只运行指定案例；必须与 --live 同时使用才会调用模型",
    )
    args = parser.parse_args(argv)

    try:
        all_cases = load_eval_cases()
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2

    cases = all_cases
    if args.case_id:
        cases = [case for case in all_cases if case["id"] == args.case_id]
        if not cases:
            print(f"未知 Evals 案例：{args.case_id}", file=sys.stderr)
            return 2

    if not args.live:
        print(
            json.dumps(
                {
                    "mode": "offline_validation",
                    "scope": "排班 MVP 与当前能力边界，不代表通用数学建模能力",
                    "case_count": len(cases),
                    "api_calls": 0,
                    "message": "案例文件有效；未调用 DeepSeek。传入 --live 才执行在线评测。",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    run_analysis = analyzer if analyzer is not None else analyze_problem
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            analysis = run_analysis(case["request"])
            payload: dict[str, Any] = {
                "analysis": analysis.model_dump(mode="json"),
                "modeling_run": None,
            }
            if analysis.status == "ready":
                problem = to_scheduling_problem(analysis)
                modeling_run: ModelingRun = run_modeling(problem)
                payload["modeling_run"] = asdict(modeling_run)
            results.append(evaluate_case(case, payload))
        except Exception as exc:
            results.append(_error_result(case, exc))

    report = _build_report(cases, results)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed_count"] == report["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
