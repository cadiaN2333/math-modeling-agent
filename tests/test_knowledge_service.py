import pytest
from pydantic import ValidationError


def _source(
    source_id: str = "src-linear",
    family: str = "linear_programming",
    source_uri: str = "repo://docs/knowledge/linear-modeling.md",
    card_path: str = "docs/knowledge/linear-modeling.md",
    solver_ids: list[str] | None = None,
):
    from math_modeling_agent.knowledge_models import KnowledgeSource

    return KnowledgeSource(
        source_id=source_id,
        title="线性规划方法卡",
        source_uri=source_uri,
        version="2026-09-29",
        problem_families=[family],
        solver_ids=solver_ids or ["glop", "scip"],
        review_status="approved",
        card_path=card_path,
    )


def _chunk(
    chunk_id: str,
    text: str,
    *,
    family="linear_programming",
    status="approved",
    source_id="src-linear",
    source_uri="repo://docs/knowledge/linear-modeling.md",
):
    from math_modeling_agent.knowledge_models import KnowledgeChunk

    return KnowledgeChunk(
        chunk_id=chunk_id,
        text=text,
        source_id=source_id,
        source_uri=source_uri,
        locator="线性模型 > 建模步骤",
        title="线性规划方法卡",
        problem_families=[family],
        solver_ids=["glop", "scip"],
        review_status=status,
    )


class FakeVectorSearch:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    def search(self, query, *, problem_family, solver_ids, top_k):
        self.calls.append((query, problem_family, solver_ids, top_k))
        return self._chunks[:top_k]


def test_knowledge_chunk_requires_complete_source_and_filter_metadata() -> None:
    from math_modeling_agent.knowledge_models import KnowledgeChunk

    with pytest.raises(ValidationError):
        KnowledgeChunk(
            chunk_id="chunk-1",
            text="线性规划知识",
            source_id="src-linear",
            source_uri="repo://linear.md",
            problem_families=["linear_programming"],
            solver_ids=[],
            review_status="approved",
        )


def test_knowledge_service_rejects_chunks_without_a_manifest_source() -> None:
    from math_modeling_agent.knowledge_service import KnowledgeService

    with pytest.raises(ValueError, match="source_id"):
        KnowledgeService(
            chunks=[_chunk("chunk-1", "线性规划目标函数")],
            sources=[],
        )


def test_knowledge_service_fuses_keyword_and_vector_ranks_deterministically() -> None:
    from math_modeling_agent.knowledge_service import KnowledgeService

    first = _chunk("a-linear", "线性规划 资源约束 目标函数")
    second = _chunk("b-linear", "线性规划 资源约束 目标函数")
    vector = FakeVectorSearch([second, first, first])
    service = KnowledgeService(
        chunks=[first, second],
        sources=[_source()],
        vector_search=vector,
    )

    evidence = service.retrieve(
        "线性规划 目标函数",
        problem_family="linear_programming",
        solver_ids=["glop"],
        top_k=5,
    )

    assert [item.chunk_id for item in evidence] == ["a-linear", "b-linear"]
    assert [item.rank for item in evidence] == [1, 2]
    assert all(item.retrieval_method == "hybrid" for item in evidence)
    assert evidence[0].source_id == "src-linear"
    assert evidence[0].source_uri == "repo://docs/knowledge/linear-modeling.md"
    assert evidence[0].locator == "线性模型 > 建模步骤"
    assert vector.calls == [
        ("线性规划 目标函数", "linear_programming", ["glop"], 25)
    ]


def test_knowledge_service_filters_family_solver_and_unapproved_chunks() -> None:
    from math_modeling_agent.knowledge_service import KnowledgeService

    linear = _chunk("linear", "线性规划 目标函数")
    storage = _chunk(
        "storage",
        "储能 SOC 充电功率",
        family="energy_park",
        source_id="src-energy",
        source_uri="repo://docs/knowledge/energy-storage.md",
    )
    draft = _chunk(
        "draft",
        "线性规划 草稿内容",
        status="draft",
    )
    vector = FakeVectorSearch([storage, draft, linear])
    service = KnowledgeService(
        chunks=[linear, storage, draft],
        sources=[
            _source(),
            _source(
                "src-energy",
                family="energy_park",
                source_uri="repo://docs/knowledge/energy-storage.md",
                card_path="docs/knowledge/energy-storage.md",
            ),
        ],
        vector_search=vector,
    )

    evidence = service.retrieve(
        "储能 线性规划",
        problem_family="linear_programming",
        solver_ids=["glop"],
    )

    assert [item.chunk_id for item in evidence] == ["linear"]
    assert evidence[0].review_status == "approved"


def test_keyword_only_results_are_labeled_and_query_inputs_are_validated() -> None:
    from math_modeling_agent.knowledge_service import KnowledgeService

    chunk = _chunk("linear", "线性规划 目标函数和约束")
    service = KnowledgeService(chunks=[chunk], sources=[_source()])

    evidence = service.retrieve(
        "线性规划 约束",
        problem_family="linear_programming",
    )

    assert len(evidence) == 1
    assert evidence[0].retrieval_method == "keyword"
    with pytest.raises(ValueError, match="query"):
        service.retrieve(" ", problem_family="linear_programming")
    with pytest.raises(ValueError, match="top_k"):
        service.retrieve("线性规划", problem_family="linear_programming", top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        service.retrieve("线性规划", problem_family="linear_programming", top_k=51)
