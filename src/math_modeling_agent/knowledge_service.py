"""使用带来源的关键词/向量混合检索返回运筹知识证据。"""

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol

from .knowledge_models import KnowledgeChunk, KnowledgeSource, RetrievedEvidence
from .retriever import HMMLRetriever, MethodRecommendation


class VectorSearch(Protocol):
    """本地或托管向量库需要实现的最小检索接口。"""

    def search(
        self,
        query: str,
        *,
        problem_family: str,
        solver_ids: list[str] | None,
        top_k: int,
    ) -> list[KnowledgeChunk]: ...


def _tokenize(text: str) -> list[str]:
    """将中文文本拆成字符二元组，同时保留英文和数字词元。"""

    segments = re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9_]+", text.casefold())
    tokens: list[str] = []
    for segment in segments:
        if segment and "\u3400" <= segment[0] <= "\u9fff":
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(
                    segment[index : index + 2]
                    for index in range(len(segment) - 1)
                )
        else:
            tokens.append(segment)
    return tokens


def _bm25_scores(query: str, chunks: list[KnowledgeChunk]) -> dict[str, float]:
    """用 BM25 公式对候选知识块排序，不把分数解释为概率。"""

    if not chunks:
        return {}
    query_terms = set(_tokenize(query))
    if not query_terms:
        return {}

    tokenized_documents = [_tokenize(chunk.text) for chunk in chunks]
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        BM25Okapi = None

    if BM25Okapi is not None:
        bm25 = BM25Okapi(tokenized_documents)
        ranked_scores = bm25.get_scores(_tokenize(query))
        return {
            chunk.chunk_id: float(score)
            for chunk, score in zip(chunks, ranked_scores)
            if score > 0
        }

    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_documents:
        document_frequency.update(set(tokens))

    document_count = len(chunks)
    average_length = sum(map(len, tokenized_documents)) / document_count or 1.0
    k1 = 1.5
    b = 0.75
    scores: dict[str, float] = {}
    for chunk, tokens in zip(chunks, tokenized_documents):
        term_frequency = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = term_frequency[term]
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (document_count - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * len(tokens) / average_length
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1) / denominator
            )
        if score > 0:
            scores[chunk.chunk_id] = score
    return scores


class KnowledgeService:
    """只返回审核通过且能追溯到清单来源的证据片段。"""

    def __init__(
        self,
        *,
        chunks: list[KnowledgeChunk],
        sources: list[KnowledgeSource],
        vector_search: VectorSearch | None = None,
        hmml_retriever: HMMLRetriever | None = None,
    ) -> None:
        source_ids = [source.source_id for source in sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("知识来源清单中的 source_id 不能重复")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("知识块 chunk_id 不能重复")

        self._sources = {source.source_id: source for source in sources}
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        for chunk in chunks:
            source = self._sources.get(chunk.source_id)
            if source is None:
                raise ValueError(
                    f"知识块 {chunk.chunk_id} 引用了清单中不存在的 source_id："
                    f"{chunk.source_id}"
                )
            if source.source_uri != chunk.source_uri:
                raise ValueError(
                    f"知识块 {chunk.chunk_id} 的 source_uri 与清单不一致"
                )
            if not set(chunk.problem_families) <= set(source.problem_families):
                raise ValueError(
                    f"知识块 {chunk.chunk_id} 的问题族超出来源清单范围"
                )

        self._vector_search = vector_search
        self._hmml_retriever = hmml_retriever or HMMLRetriever()

    @classmethod
    def from_local_index(
        cls,
        manifest_path: str | Path,
        persist_directory: str | Path,
        *,
        model_name: str = "BAAI/bge-m3",
        allow_model_download: bool = False,
    ) -> "KnowledgeService":
        """显式连接本地清单和 Chroma 索引；默认禁止查询时下载模型。"""

        from .chroma_store import ChromaVectorSearch
        from .knowledge_index import (
            knowledge_collection_name,
            load_knowledge_base,
        )

        sources, chunks = load_knowledge_base(manifest_path)
        approved_sources = [
            source for source in sources if source.review_status == "approved"
        ]
        approved_source_ids = {source.source_id for source in approved_sources}
        approved_chunks = [
            chunk
            for chunk in chunks
            if chunk.review_status == "approved"
            and chunk.source_id in approved_source_ids
        ]
        vector_search = ChromaVectorSearch.from_local(
            persist_directory,
            model_name=model_name,
            allow_model_download=allow_model_download,
            collection_name=knowledge_collection_name(
                approved_sources,
                approved_chunks,
                model_name,
            ),
        )
        return cls(
            chunks=chunks,
            sources=sources,
            vector_search=vector_search,
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        vector_search: VectorSearch | None = None,
    ) -> "KnowledgeService":
        """仅加载本地清单供 BM25 使用，可选注入任意向量检索后端。"""

        from .knowledge_index import load_knowledge_base

        sources, chunks = load_knowledge_base(manifest_path)
        return cls(chunks=chunks, sources=sources, vector_search=vector_search)

    def retrieve(
        self,
        query: str,
        *,
        problem_family: str,
        solver_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:
        """按审核状态与问题族过滤后，以 RRF 合并 BM25 和向量排名。"""

        if not query.strip():
            raise ValueError("query 不能为空")
        if not problem_family.strip():
            raise ValueError("problem_family 不能为空")
        if isinstance(top_k, bool) or not 1 <= top_k <= 50:
            raise ValueError("top_k 必须在 1 到 50 之间")
        if solver_ids is not None and (
            any(not solver_id.strip() for solver_id in solver_ids)
            or len(solver_ids) != len(set(solver_ids))
        ):
            raise ValueError("solver_ids 不能包含空值或重复项")

        eligible_chunks = [
            chunk
            for chunk in self._chunks.values()
            if self._is_eligible(chunk, problem_family, solver_ids)
        ]
        eligible_by_id = {chunk.chunk_id: chunk for chunk in eligible_chunks}
        candidate_limit = max(top_k * 5, top_k)

        keyword_scores = _bm25_scores(query, eligible_chunks)
        keyword_ranked = sorted(
            keyword_scores,
            key=lambda chunk_id: (-keyword_scores[chunk_id], chunk_id),
        )[:candidate_limit]

        vector_ranked: list[str] = []
        if self._vector_search is not None:
            vector_chunks = self._vector_search.search(
                query,
                problem_family=problem_family,
                solver_ids=solver_ids,
                top_k=candidate_limit,
            )
            seen_vector_ids: set[str] = set()
            for candidate in vector_chunks:
                canonical = eligible_by_id.get(candidate.chunk_id)
                if canonical is None or canonical.chunk_id in seen_vector_ids:
                    continue
                seen_vector_ids.add(canonical.chunk_id)
                vector_ranked.append(canonical.chunk_id)

        rrf_scores: dict[str, float] = defaultdict(float)
        retrieval_methods: dict[str, set[str]] = defaultdict(set)
        for method, ranked_ids in (
            ("keyword", keyword_ranked),
            ("vector", vector_ranked),
        ):
            for rank, chunk_id in enumerate(ranked_ids, start=1):
                rrf_scores[chunk_id] += 1 / (60 + rank)
                retrieval_methods[chunk_id].add(method)

        ordered_ids = sorted(
            rrf_scores,
            key=lambda chunk_id: (-rrf_scores[chunk_id], chunk_id),
        )[:top_k]
        evidence: list[RetrievedEvidence] = []
        for rank, chunk_id in enumerate(ordered_ids, start=1):
            chunk = eligible_by_id[chunk_id]
            methods = retrieval_methods[chunk_id]
            method = (
                "hybrid"
                if len(methods) > 1
                else next(iter(methods))
            )
            evidence.append(
                RetrievedEvidence(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    source_uri=chunk.source_uri,
                    locator=chunk.locator,
                    title=chunk.title,
                    text=chunk.text,
                    retrieval_method=method,
                    rank=rank,
                    problem_families=chunk.problem_families,
                    review_status="approved",
                )
            )
        return evidence

    def retrieve_methods(
        self,
        problem_description: str,
        desired_outcome: str,
        *,
        required_method_id: str | None = None,
        top_k: int = 3,
    ) -> list[MethodRecommendation]:
        """调用 HMML 结构化方法检索；方法建议与证据片段分开返回。"""

        return self._hmml_retriever.retrieve(
            problem_description=problem_description,
            desired_outcome=desired_outcome,
            required_method_id=required_method_id,
            top_k=top_k,
        )

    def _is_eligible(
        self,
        chunk: KnowledgeChunk,
        problem_family: str,
        solver_ids: list[str] | None,
    ) -> bool:
        source = self._sources[chunk.source_id]
        if chunk.review_status != "approved" or source.review_status != "approved":
            return False
        if problem_family not in chunk.problem_families:
            return False
        if solver_ids is not None and not set(solver_ids).intersection(chunk.solver_ids):
            return False
        return True
