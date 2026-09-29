import json

import pytest


def _write_manifest(root, *, card_path="docs/knowledge/linear.md"):
    manifest_path = root / "data" / "knowledge" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": [
                    {
                        "source_id": "linear-card",
                        "title": "线性规划方法卡",
                        "source_uri": "repo://docs/knowledge/linear.md",
                        "version": "2026-09-29",
                        "problem_families": ["linear_programming"],
                        "solver_ids": ["glop", "scip"],
                        "review_status": "approved",
                        "card_path": card_path,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_load_knowledge_base_chunks_by_heading_and_generates_stable_ids(tmp_path) -> None:
    from math_modeling_agent.knowledge_index import load_knowledge_base

    card = tmp_path / "docs" / "knowledge" / "linear.md"
    card.parent.mkdir(parents=True)
    card.write_text(
        "# 线性规划\n\n## 建模步骤\n\n明确变量、目标函数和约束。\n\n"
        "## 单位检查\n\n确保变量和目标单位一致。\n",
        encoding="utf-8",
    )
    manifest_path = _write_manifest(tmp_path)

    sources_a, chunks_a = load_knowledge_base(manifest_path)
    sources_b, chunks_b = load_knowledge_base(manifest_path)

    assert len(sources_a) == 1
    assert [chunk.chunk_id for chunk in chunks_a] == [
        chunk.chunk_id for chunk in chunks_b
    ]
    assert chunks_a[0].locator == "线性规划 > 建模步骤"
    assert chunks_a[0].source_id == "linear-card"
    assert chunks_a[0].source_uri == "repo://docs/knowledge/linear.md"
    assert "明确变量" in chunks_a[0].text
    assert len(chunks_a) == 2


def test_load_knowledge_base_rejects_path_traversal_and_missing_cards(tmp_path) -> None:
    from math_modeling_agent.knowledge_index import load_knowledge_base

    outside_path = _write_manifest(tmp_path, card_path="../../outside.md")
    with pytest.raises(ValueError, match="仓库目录"):
        load_knowledge_base(outside_path)

    missing_path = _write_manifest(tmp_path / "missing", card_path="docs/missing.md")
    with pytest.raises(ValueError, match="读取知识卡片失败"):
        load_knowledge_base(missing_path)


def test_repository_manifest_loads_four_approved_cards_and_retrieves_storage_sources() -> None:
    from pathlib import Path

    from math_modeling_agent.knowledge_index import load_knowledge_base
    from math_modeling_agent.knowledge_service import KnowledgeService

    repo_root = Path(__file__).resolve().parents[1]
    sources, chunks = load_knowledge_base(
        repo_root / "data" / "knowledge" / "manifest.json"
    )
    service = KnowledgeService(chunks=chunks, sources=sources)

    evidence = service.retrieve(
        "SOC 10% 90% 储能额定时长 4小时 日损耗",
        problem_family="energy_park",
        solver_ids=["ortools_scip"],
    )

    assert len(sources) == 4
    assert chunks
    assert evidence
    assert evidence[0].source_id == "local-electric-cup-storage-assumptions"
    assert "并非由题目附件直接给出的物理事实" in evidence[0].text


def test_chroma_vector_adapter_requires_explicitly_built_local_index(tmp_path) -> None:
    from math_modeling_agent.chroma_store import ChromaVectorSearch

    with pytest.raises(RuntimeError, match="先显式运行"):
        ChromaVectorSearch.from_local(tmp_path / "not-built")


def test_chroma_collection_name_is_stable_but_changes_with_model() -> None:
    from pathlib import Path

    from math_modeling_agent.knowledge_index import (
        knowledge_collection_name,
        load_knowledge_base,
    )

    repo_root = Path(__file__).resolve().parents[1]
    sources, chunks = load_knowledge_base(
        repo_root / "data" / "knowledge" / "manifest.json"
    )

    first = knowledge_collection_name(sources, chunks, "BAAI/bge-m3")
    second = knowledge_collection_name(sources, chunks, "BAAI/bge-m3")
    other_model = knowledge_collection_name(sources, chunks, "other/model")

    assert first == second
    assert first != other_model
