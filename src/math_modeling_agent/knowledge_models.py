"""知识清单、分块与可引用检索证据的数据契约。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ReviewStatus = Literal["approved", "draft"]


class _KnowledgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KnowledgeSource(_KnowledgeModel):
    """知识卡片的来源清单记录。"""

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    version: str = Field(min_length=1)
    problem_families: list[str] = Field(min_length=1)
    solver_ids: list[str] = Field(default_factory=list)
    review_status: ReviewStatus
    card_path: str = Field(min_length=1)

    @field_validator("problem_families")
    @classmethod
    def families_must_be_unique(cls, value: list[str]) -> list[str]:
        if any(not family.strip() for family in value):
            raise ValueError("问题族不能包含空名称")
        if len(value) != len(set(value)):
            raise ValueError("问题族不能重复")
        return value

    @field_validator("solver_ids")
    @classmethod
    def solver_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if any(not solver_id.strip() for solver_id in value):
            raise ValueError("求解器编号不能包含空名称")
        if len(value) != len(set(value)):
            raise ValueError("求解器编号不能重复")
        return value


class KnowledgeManifest(_KnowledgeModel):
    """知识来源清单及版本。"""

    version: Literal[1]
    sources: list[KnowledgeSource] = Field(min_length=1)

    @model_validator(mode="after")
    def source_ids_must_be_unique(self) -> "KnowledgeManifest":
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("知识来源 source_id 不能重复")
        return self


class KnowledgeChunk(_KnowledgeModel):
    """带完整来源定位和审核元数据的知识文本块。"""

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    title: str = Field(min_length=1)
    problem_families: list[str] = Field(min_length=1)
    solver_ids: list[str] = Field(default_factory=list)
    review_status: ReviewStatus

    @field_validator("problem_families", "solver_ids")
    @classmethod
    def metadata_lists_must_be_unique(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("元数据列表不能包含空名称")
        if len(value) != len(set(value)):
            raise ValueError("元数据列表不能包含重复值")
        return value


class RetrievedEvidence(_KnowledgeModel):
    """可呈现给用户并可写入报告的审核通过证据。"""

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    retrieval_method: Literal["keyword", "vector", "hybrid"]
    rank: int = Field(ge=1)
    problem_families: list[str] = Field(min_length=1)
    review_status: Literal["approved"]

    @model_validator(mode="after")
    def evidence_must_have_problem_family(self) -> "RetrievedEvidence":
        if not self.problem_families:
            raise ValueError("证据至少要标注一个适用问题族")
        return self
