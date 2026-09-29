"""离线评估 RAG 来源召回、排序、引用准确率与无证据拒答。"""

import json
from pathlib import Path
from typing import Any

from .knowledge_service import KnowledgeService


def load_knowledge_eval_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    """加载并校验带预期来源 ID 的检索评测案例。"""

    cases_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[2]
        / "evals"
        / "knowledge_cases.json"
    )
    try:
        payload = json.loads(cases_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取知识检索评测案例：{cases_path}") from error
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("知识检索评测文件必须是 version=1 的对象")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("知识检索评测文件必须含非空 cases 列表")

    case_ids: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("每个知识检索评测案例都必须是对象")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("每个知识检索评测案例都必须包含非空 id")
        case_ids.append(case_id)
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"评测案例 {case_id} 缺少 query")
        if (
            not isinstance(case.get("problem_family"), str)
            or not case["problem_family"].strip()
        ):
            raise ValueError(f"评测案例 {case_id} 缺少 problem_family")
        if "solver_ids" in case and (
            not isinstance(case["solver_ids"], list)
            or any(not isinstance(item, str) or not item.strip() for item in case["solver_ids"])
            or len(case["solver_ids"]) != len(set(case["solver_ids"]))
        ):
            raise ValueError(f"评测案例 {case_id} 的 solver_ids 无效")
        expected = case.get("expected_source_ids")
        if (
            not isinstance(expected, list)
            or any(not isinstance(item, str) or not item.strip() for item in expected)
            or len(expected) != len(set(expected))
        ):
            raise ValueError(f"评测案例 {case_id} 的 expected_source_ids 无效")

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("知识检索评测案例 id 不能重复")
    return cases


def evaluate_knowledge_cases(
    cases: list[dict[str, Any]],
    knowledge_service: KnowledgeService,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """计算 Recall@k、MRR、引用来源准确率和无证据拒答率。"""

    if top_k <= 0:
        raise ValueError("top_k 必须是正整数")

    per_case: list[dict[str, Any]] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    correct_citations = 0
    total_citations = 0
    no_evidence_cases = 0
    correct_abstentions = 0

    for case in cases:
        expected_sources = set(case["expected_source_ids"])
        evidence = knowledge_service.retrieve(
            case["query"],
            problem_family=case["problem_family"],
            solver_ids=case.get("solver_ids"),
            top_k=top_k,
        )
        retrieved_sources = [item.source_id for item in evidence]
        relevant_ranks = [
            item.rank
            for item in evidence
            if item.source_id in expected_sources
        ]

        if expected_sources:
            recalls.append(
                len(set(retrieved_sources) & expected_sources) / len(expected_sources)
            )
            reciprocal_ranks.append(
                1 / min(relevant_ranks) if relevant_ranks else 0.0
            )
        else:
            no_evidence_cases += 1
            if not evidence:
                correct_abstentions += 1

        correct_citations += sum(
            item.source_id in expected_sources
            for item in evidence
        )
        total_citations += len(evidence)
        passed = (
            bool(set(retrieved_sources) & expected_sources)
            if expected_sources
            else not evidence
        )
        per_case.append(
            {
                "case_id": case["id"],
                "passed": passed,
                "expected_source_ids": sorted(expected_sources),
                "retrieved_source_ids": retrieved_sources,
                "retrieved_chunk_ids": [item.chunk_id for item in evidence],
            }
        )

    recall_at_k = sum(recalls) / len(recalls) if recalls else None
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None
    citation_accuracy = (
        correct_citations / total_citations if total_citations else 1.0
    )
    refusal_rate = (
        correct_abstentions / no_evidence_cases if no_evidence_cases else None
    )
    return {
        "case_count": len(cases),
        "top_k": top_k,
        "recall_at_k": round(recall_at_k, 3) if recall_at_k is not None else None,
        "mrr": round(mrr, 3) if mrr is not None else None,
        "citation_source_accuracy": round(citation_accuracy, 3),
        "no_evidence_refusal_rate": (
            round(refusal_rate, 3) if refusal_rate is not None else None
        ),
        "passed": all(item["passed"] for item in per_case),
        "results": per_case,
    }
