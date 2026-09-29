"""可选的本地 Chroma 向量检索适配器。"""

import json
from pathlib import Path
from typing import Any

from .knowledge_models import KnowledgeChunk


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_COLLECTION_NAME = "optimization-knowledge-v1"


class ChromaVectorSearch:
    """将本地持久化 Chroma 结果转换回经过校验的知识块。"""

    def __init__(self, vector_store: Any) -> None:
        self._vector_store = vector_store

    @classmethod
    def from_local(
        cls,
        persist_directory: str | Path,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        allow_model_download: bool = False,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> "ChromaVectorSearch":
        """连接本地索引；只有显式允许时才允许嵌入模型联网下载。"""

        persist_path = Path(persist_directory)
        if not persist_path.is_dir():
            raise RuntimeError(
                f"本地向量索引不存在：{persist_path}；请先显式运行 build_knowledge_index.py"
            )

        try:
            import chromadb
            from chromadb.config import Settings
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as error:
            raise RuntimeError(
                '缺少可选 RAG 依赖；请运行 python -m pip install -e ".[rag]"'
            ) from error

        model_kwargs = {} if allow_model_download else {"local_files_only": True}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        return cls(vector_store)

    def search(
        self,
        query: str,
        *,
        problem_family: str,
        solver_ids: list[str] | None,
        top_k: int,
    ) -> list[KnowledgeChunk]:
        """只向 Chroma 请求已审核知识；其他过滤在 KnowledgeService 再校验。"""

        documents = self._vector_store.similarity_search(
            query,
            k=top_k,
            filter={"review_status": "approved"},
        )
        chunks: list[KnowledgeChunk] = []
        for document in documents:
            metadata = document.metadata
            try:
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=metadata["chunk_id"],
                        text=document.page_content,
                        source_id=metadata["source_id"],
                        source_uri=metadata["source_uri"],
                        locator=metadata["locator"],
                        title=metadata["title"],
                        problem_families=json.loads(
                            metadata["problem_families_json"]
                        ),
                        solver_ids=json.loads(metadata["solver_ids_json"]),
                        review_status=metadata["review_status"],
                    )
                )
            except (KeyError, TypeError, json.JSONDecodeError, ValueError):
                # 元数据损坏或不完整的向量记录不会进入证据列表。
                continue
        return chunks
