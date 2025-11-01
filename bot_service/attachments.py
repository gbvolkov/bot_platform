from __future__ import annotations

import base64
import logging
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_classic.docstore.document import Document

from services.kb_manager.utils import load_single_document

from .schemas import AttachmentPayload, ContentType

logger = logging.getLogger(__name__)


_EXTENSION_CATEGORY_MAP: Dict[ContentType, Tuple[str, ...]] = {
    ContentType.IMAGES: (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"),
    ContentType.PDFS: (".pdf",),
    ContentType.TEXT_FILES: (".txt", ".text", ".log"),
    ContentType.MARKDOWN: (".md", ".markdown"),
    ContentType.DOCX_DOCUMENTS: (".doc", ".docx"),
    ContentType.CSVS: (".csv",),
    ContentType.EXCELS: (".xls", ".xlsx", ".xlsm"),
    ContentType.SOUNDS: (".wav", ".mp3", ".m4a", ".ogg", ".flac"),
    ContentType.VIDEOS: (".mp4", ".avi", ".mov", ".mkv", ".webm"),
}

_MIME_PREFIX_CATEGORIES: Tuple[Tuple[str, ContentType], ...] = (
    ("image/", ContentType.IMAGES),
    ("video/", ContentType.VIDEOS),
    ("audio/", ContentType.SOUNDS),
    ("text/plain", ContentType.TEXT_FILES),
    ("text/markdown", ContentType.MARKDOWN),
    ("application/pdf", ContentType.PDFS),
    ("application/msword", ContentType.DOCX_DOCUMENTS),
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ContentType.DOCX_DOCUMENTS),
    ("text/csv", ContentType.CSVS),
    ("application/vnd.ms-excel", ContentType.EXCELS),
    ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ContentType.EXCELS),
)


@dataclass(slots=True)
class ProcessedAttachment:
    attachment: AttachmentPayload
    category: Optional[ContentType]
    supported: bool
    text: Optional[str]
    error: Optional[str] = None

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "filename": self.attachment.filename,
            "content_type": self.attachment.content_type,
            "category": self.category.value if self.category else None,
            "supported_by_agent": self.supported,
            "text_available": bool(self.text),
            "converted_to_text": (not self.supported) and bool(self.text),
            "error": self.error,
        }


def attachment_to_text_segment(processed: ProcessedAttachment) -> Optional[str]:
    """Build a human-readable text segment for the processed attachment."""
    if not processed.text:
        return None

    qualifiers: List[str] = []
    if processed.category:
        qualifiers.append(processed.category.value)
    if not processed.supported:
        qualifiers.append("converted to text")
    qualifier = f" ({', '.join(qualifiers)})" if qualifiers else ""
    return f"[Attachment: {processed.attachment.filename}{qualifier}]\n{processed.text}"


def _safe_filename(filename: str) -> str:
    candidate = Path(filename).name
    if not candidate or candidate in {".", ".."}:
        candidate = f"attachment_{uuid.uuid4().hex}"
    return candidate


def _guess_category(filename: str, mime_type: Optional[str]) -> Optional[ContentType]:
    ext = Path(filename).suffix.lower()
    for category, extensions in _EXTENSION_CATEGORY_MAP.items():
        if ext in extensions:
            return category
    if mime_type:
        cleaned = mime_type.split(";")[0].strip().lower()
        for prefix, category in _MIME_PREFIX_CATEGORIES:
            if cleaned.startswith(prefix):
                return category
    return None


def _write_temp_file(base_dir: Path, attachment: AttachmentPayload) -> Path:
    if attachment.data is None:
        raise ValueError(f"Attachment '{attachment.filename}' does not provide binary data.")
    try:
        raw_bytes = base64.b64decode(attachment.data)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Attachment '{attachment.filename}' contains invalid base64 data.") from exc

    filename = _safe_filename(attachment.filename)
    target = base_dir / filename
    counter = 1
    while target.exists():
        target = base_dir / f"{Path(filename).stem}_{counter}{Path(filename).suffix}"
        counter += 1
    target.write_bytes(raw_bytes)
    return target


def _documents_to_text(documents: Sequence[Document]) -> str:
    parts: List[str] = []
    for doc in documents:
        content = getattr(doc, "page_content", None)
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n\n".join(parts)


def process_attachments(
    attachments: Sequence[AttachmentPayload],
    supported_categories: Iterable[ContentType],
    *,
    tmp_dir: Optional[Path] = None,
) -> List[ProcessedAttachment]:
    supported_set = set(supported_categories)
    if not attachments:
        return []

    cleanup: Optional[tempfile.TemporaryDirectory] = None
    if tmp_dir is None:
        cleanup = tempfile.TemporaryDirectory(prefix="bot_att_")
        base_dir = Path(cleanup.name)
    else:
        base_dir = tmp_dir
        base_dir.mkdir(parents=True, exist_ok=True)

    try:
        processed: List[ProcessedAttachment] = []
        for attachment in attachments:
            category = _guess_category(attachment.filename, attachment.content_type)
            supported = category in supported_set if category else False
            if attachment.text is not None:
                processed.append(
                    ProcessedAttachment(
                        attachment=attachment,
                        category=category,
                        supported=supported,
                        text=attachment.text,
                    )
                )
                continue

            temp_file: Optional[Path] = None
            try:
                temp_file = _write_temp_file(base_dir, attachment)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to stage attachment '%s': %s", attachment.filename, exc)
                processed.append(
                    ProcessedAttachment(
                        attachment=attachment,
                        category=category,
                        supported=supported,
                        text=None,
                        error=str(exc),
                    )
                )
                continue

            try:
                documents = load_single_document(str(temp_file))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse attachment '%s': %s", attachment.filename, exc)
                processed.append(
                    ProcessedAttachment(
                        attachment=attachment,
                        category=category,
                        supported=supported,
                        text=None,
                        error=str(exc),
                    )
                )
                continue
            finally:
                if temp_file is not None:
                    try:
                        temp_file.unlink(missing_ok=True)
                    except Exception:  # noqa: BLE001
                        logger.debug("Could not remove temp file %s", temp_file)

            text_content = _documents_to_text(documents)
            if not text_content and not supported:
                text_content = ""
            processed.append(
                ProcessedAttachment(
                    attachment=attachment,
                    category=category,
                    supported=supported,
                    text=text_content or None,
                )
            )
        return processed
    finally:
        if cleanup is not None:
            cleanup.cleanup()
