"""离线评测已构建的本地知识索引；不调用 DeepSeek 或远程嵌入 API。"""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from math_modeling_agent.knowledge_eval import (
    evaluate_knowledge_cases,
    load_knowledge_eval_cases,
)
from math_modeling_agent.knowledge_index import load_knowledge_base
from math_modeling_agent.knowledge_service import KnowledgeService


def main() -> int:
    parser = argparse.ArgumentParser(description="离线评测本地优化知识检索")
    parser.add_argument(
        "--keyword-only",
        action="store_true",
        help="只评测 BM25/关键词路径，不加载向量模型",
    )
    args = parser.parse_args()
    if args.keyword_only:
        sources, chunks = load_knowledge_base(
            REPO_ROOT / "data" / "knowledge" / "manifest.json"
        )
        service = KnowledgeService(chunks=chunks, sources=sources)
    else:
        service = KnowledgeService.from_local_index(
            REPO_ROOT / "data" / "knowledge" / "manifest.json",
            REPO_ROOT / ".cache" / "optimization-rag",
            allow_model_download=False,
        )
    cases = load_knowledge_eval_cases(REPO_ROOT / "evals" / "knowledge_cases.json")
    report = evaluate_knowledge_cases(cases, service, top_k=5)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"知识检索评测失败：{error}", file=sys.stderr)
        raise SystemExit(2) from error
