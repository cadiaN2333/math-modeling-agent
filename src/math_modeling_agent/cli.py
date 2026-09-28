"""通过命令行运行排班建模示例。"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .analysis_agent import (
    analyze_problem,
    retrieve_methods_for_subtasks,
    to_scheduling_problem,
)
from .agent import run_modeling
from .problem_io import load_problem_file
from .sample_data import make_sample_problem


def main(argv: list[str] | None = None, *, llm_client=None) -> int:
    """运行样例、JSON 输入或 DeepSeek 自然语言分析。"""

    parser = argparse.ArgumentParser(description="可验证的排班建模示例")
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
    args = parser.parse_args(argv)

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
        }

        if analysis.status == "ready":
            try:
                problem = to_scheduling_problem(analysis)
            except ValueError as exc:
                print(f"排班草稿校验失败：{exc}", file=sys.stderr)
                return 2

            modeling_run = run_modeling(problem)
            payload["modeling_run"] = asdict(modeling_run)

        print(json.dumps(payload, ensure_ascii=False, indent=2))

        modeling_run_data = payload.get("modeling_run")
        if modeling_run_data is not None:
            solver_result = modeling_run_data["solver_result"]
            validation_report = modeling_run_data["validation_report"]
            if (
                solver_result["status"] not in {"OPTIMAL", "FEASIBLE"}
                or validation_report is None
                or not validation_report["is_valid"]
            ):
                return 1

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
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))

    # 只有找到可行排班并通过验证时，命令才以成功状态结束
    is_valid = (
        result.solver_result.status in {"OPTIMAL", "FEASIBLE"}
        and result.validation_report is not None
        and result.validation_report.is_valid
    )
    return 0 if is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
