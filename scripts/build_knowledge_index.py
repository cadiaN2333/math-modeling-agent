"""显式构建本地 RAG 索引；首次运行会加载或下载 BGE-M3。"""

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from math_modeling_agent.chroma_store import (
    DEFAULT_EMBEDDING_MODEL,
)
from math_modeling_agent.knowledge_index import (
    knowledge_collection_name,
    load_knowledge_base,
)


def build_index(
    manifest_path: Path,
    output_directory: Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, object]:
    """将审核通过的本地 Markdown 知识块写入持久化 Chroma 集合。"""

    sources, chunks = load_knowledge_base(manifest_path, repo_root=REPO_ROOT)
    approved_source_ids = {
        source.source_id
        for source in sources
        if source.review_status == "approved"
    }
    approved_chunks = [
        chunk
        for chunk in chunks
        if chunk.review_status == "approved"
        and chunk.source_id in approved_source_ids
    ]
    if not approved_chunks:
        raise ValueError("知识清单没有审核通过的本地知识块")
    approved_sources = [
        source for source in sources if source.review_status == "approved"
    ]
    collection_name = knowledge_collection_name(
        approved_sources,
        approved_chunks,
        model_name,
    )

    try:
        import chromadb
        from chromadb.config import Settings
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as error:
        raise RuntimeError(
            '缺少可选 RAG 依赖；请运行 python -m pip install -e ".[rag]"'
        ) from error

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    client = chromadb.PersistentClient(
        path=str(output_directory),
        settings=Settings(anonymized_telemetry=False),
    )
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    documents = [
        Document(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "source_uri": chunk.source_uri,
                "locator": chunk.locator,
                "title": chunk.title,
                "problem_families_json": json.dumps(
                    chunk.problem_families,
                    ensure_ascii=False,
                ),
                "solver_ids_json": json.dumps(
                    chunk.solver_ids,
                    ensure_ascii=False,
                ),
                "review_status": chunk.review_status,
            },
        )
        for chunk in approved_chunks
    ]
    vector_store.add_documents(
        documents=documents,
        ids=[chunk.chunk_id for chunk in approved_chunks],
    )
    return {
        "indexed_chunks": len(approved_chunks),
        "sources": sorted(approved_source_ids),
        "embedding_model": model_name,
        "persist_directory": str(output_directory),
        "collection_name": collection_name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="显式构建本地优化知识索引")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data" / "knowledge" / "manifest.json",
        help="版本化知识清单 JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".cache" / "optimization-rag",
        help="本地向量索引目录",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Hugging Face 嵌入模型；首次运行可能下载权重",
    )
    args = parser.parse_args(argv)
    try:
        result = build_index(args.manifest, args.output, model_name=args.model)
    except (RuntimeError, ValueError) as error:
        parser.exit(2, f"知识索引构建失败：{error}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
