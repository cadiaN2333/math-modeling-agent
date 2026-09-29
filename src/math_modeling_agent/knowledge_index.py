"""加载版本化知识清单并按 Markdown 标题生成稳定知识块。"""

import hashlib
import json
import re
from pathlib import Path

from pydantic import ValidationError

from .knowledge_models import KnowledgeChunk, KnowledgeManifest, KnowledgeSource


def _split_markdown(source: KnowledgeSource, markdown: str) -> list[KnowledgeChunk]:
    """按标题层级切块，并用来源、定位和归一化文本生成稳定 ID。"""

    sections: list[tuple[str, str]] = []
    headings: list[str] = []
    current_lines: list[str] = []
    saw_heading = False

    def flush() -> None:
        if not current_lines:
            return
        locator = " > ".join(headings) if headings else source.title
        content = "\n".join(current_lines).strip()
        body_lines = [
            line
            for line in current_lines
            if line.strip() and not re.match(r"^#{1,6}\s+", line)
        ]
        if content and body_lines:
            sections.append((locator, content))

    for line in markdown.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            flush()
            saw_heading = True
            level = len(match.group(1))
            headings = headings[: level - 1]
            headings.append(match.group(2).strip())
            current_lines = [line]
        else:
            current_lines.append(line)

    flush()
    if not saw_heading and not sections and markdown.strip():
        sections.append((source.title, markdown.strip()))

    chunks: list[KnowledgeChunk] = []
    for locator, text in sections:
        normalized_text = re.sub(r"\s+", " ", text).strip().casefold()
        stable_key = f"{source.source_id}\n{locator}\n{normalized_text}"
        chunk_id = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                text=text,
                source_id=source.source_id,
                source_uri=source.source_uri,
                locator=locator,
                title=source.title,
                problem_families=source.problem_families,
                solver_ids=source.solver_ids,
                review_status=source.review_status,
            )
        )
    if not chunks:
        raise ValueError(f"知识卡片没有可索引正文：{source.card_path}")
    return chunks


def load_knowledge_base(
    manifest_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> tuple[list[KnowledgeSource], list[KnowledgeChunk]]:
    """校验清单、限制卡片路径在仓库内，并生成审核元数据分块。"""

    manifest_file = Path(manifest_path).resolve()
    try:
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取知识清单：{manifest_file}") from error
    try:
        manifest = KnowledgeManifest.model_validate(manifest_data)
    except ValidationError as error:
        raise ValueError(f"知识清单格式无效：{error}") from error

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else manifest_file.parents[2]
    )
    all_chunks: list[KnowledgeChunk] = []
    for source in manifest.sources:
        relative_card = Path(source.card_path)
        if relative_card.is_absolute():
            raise ValueError(f"知识卡片路径必须是仓库相对路径：{source.card_path}")
        card_path = (root / relative_card).resolve()
        try:
            card_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"知识卡片路径不能越出仓库目录：{source.card_path}"
            ) from error

        try:
            markdown = card_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"读取知识卡片失败：{source.card_path}") from error
        all_chunks.extend(_split_markdown(source, markdown))

    return list(manifest.sources), all_chunks


def knowledge_collection_name(
    sources: list[KnowledgeSource],
    chunks: list[KnowledgeChunk],
    model_name: str,
) -> str:
    """根据知识内容、元数据和嵌入模型生成隔离的 Chroma 集合名。"""

    signature = {
        "model": model_name,
        "sources": sorted(
            (source.source_id, source.version, source.review_status)
            for source in sources
        ),
        "chunks": sorted(
            (
                chunk.chunk_id,
                chunk.source_uri,
                chunk.locator,
                tuple(chunk.problem_families),
                tuple(chunk.solver_ids),
                chunk.review_status,
            )
            for chunk in chunks
        ),
    }
    signature_text = json.dumps(signature, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()[:16]
    return f"optimization-knowledge-{digest}"
