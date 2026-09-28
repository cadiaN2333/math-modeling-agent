"""加载并校验分层数学建模知识库。"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class HMMLMethod(BaseModel):
    """第三层：一个具体建模方法及其适用信息。"""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    problem_keywords: list[str] = Field(default_factory=list)
    goal_keywords: list[str] = Field(default_factory=list)
    core_idea: str = Field(min_length=1)
    use_when: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    implementation_status: Literal["已实现", "知识候选"]
    solver: str | None = None


class HMMLSubdomain(BaseModel):
    """第二层：一类相关建模问题或方法范围。"""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    problem_keywords: list[str] = Field(default_factory=list)
    goal_keywords: list[str] = Field(default_factory=list)
    methods: list[HMMLMethod] = Field(min_length=1)


class HMMLDomain(BaseModel):
    """第一层：较大的数学建模领域。"""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    problem_keywords: list[str] = Field(default_factory=list)
    goal_keywords: list[str] = Field(default_factory=list)
    subdomains: list[HMMLSubdomain] = Field(min_length=1)


class HMMLLibrary(BaseModel):
    """完整的三层 HMML 知识树。"""

    schema_version: str
    domains: list[HMMLDomain] = Field(min_length=1)

    @model_validator(mode="after")
    def method_ids_must_be_unique(self) -> "HMMLLibrary":
        method_ids = [
            method.id
            for domain in self.domains
            for subdomain in domain.subdomains
            for method in subdomain.methods
        ]
        if len(method_ids) != len(set(method_ids)):
            raise ValueError("HMML 方法编号必须唯一")
        return self


def load_hmml(path: str | Path | None = None) -> HMMLLibrary:
    """从 JSON 文件加载 HMML；未指定路径时使用项目内置知识库。"""

    library_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[2] / "data" / "hmml.json"
    )
    try:
        raw_data = json.loads(library_path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ValueError(f"无法读取 HMML 文件 {library_path}：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"HMML JSON 格式错误（第 {exc.lineno} 行，第 {exc.colno} 列）"
        ) from exc

    return HMMLLibrary.model_validate(raw_data)
