from pathlib import Path

import pytest


def test_local_knowledge_eval_reports_recall_mrr_citation_and_abstention() -> None:
    from math_modeling_agent.knowledge_eval import (
        evaluate_knowledge_cases,
        load_knowledge_eval_cases,
    )
    from math_modeling_agent.knowledge_index import load_knowledge_base
    from math_modeling_agent.knowledge_service import KnowledgeService

    repo_root = Path(__file__).resolve().parents[1]
    sources, chunks = load_knowledge_base(
        repo_root / "data" / "knowledge" / "manifest.json"
    )
    cases = load_knowledge_eval_cases(repo_root / "evals" / "knowledge_cases.json")
    report = evaluate_knowledge_cases(
        cases,
        KnowledgeService(chunks=chunks, sources=sources),
        top_k=5,
    )

    assert report["case_count"] == 4
    assert report["recall_at_k"] == 1.0
    assert report["mrr"] >= 0.8
    assert 0.7 <= report["citation_source_accuracy"] <= 1.0
    assert report["no_evidence_refusal_rate"] == 1.0
    assert report["passed"] is True


def test_knowledge_eval_rejects_duplicate_case_ids_and_bad_source_filters(tmp_path) -> None:
    from math_modeling_agent.knowledge_eval import load_knowledge_eval_cases

    path = tmp_path / "cases.json"
    path.write_text(
        '{"version":1,"cases":['
        '{"id":"same","query":"x","problem_family":"linear_programming","expected_source_ids":[]},'
        '{"id":"same","query":"y","problem_family":"linear_programming","expected_source_ids":[]}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="id"):
        load_knowledge_eval_cases(path)
