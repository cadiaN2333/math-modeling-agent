"""负责从 JSON 文件读取并校验排班问题。"""

import json
from pathlib import Path

from pydantic import ValidationError

from .models import SchedulingProblem


def load_problem_file(path: str | Path) -> SchedulingProblem:
    """读取 UTF-8 JSON 文件，并转换成经过校验的排班问题。"""

    problem_path = Path(path)

    try:
        # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的 UTF-8 文件
        contents = problem_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"无法读取输入文件 {problem_path}：{exc}") from exc

    try:
        data = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON 格式错误（第 {exc.lineno} 行，第 {exc.colno} 列）：{exc.msg}"
        ) from exc

    try:
        return SchedulingProblem.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"排班问题数据校验失败：{exc}") from exc
