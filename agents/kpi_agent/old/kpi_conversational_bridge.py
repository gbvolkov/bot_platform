from __future__ import annotations

from langchain_core.documents import Document

from utils.kpi_retrieval import KPIHybridResult


def build_kpi_dialog_guidance_document(retrieval: KPIHybridResult) -> Document | None:
    """Преобразует scripted KPI-решение в служебный контекст для LLM."""
    direct_answer = str(retrieval.direct_answer or "").strip()
    if not direct_answer:
        return None

    diagnostics = retrieval.diagnostics or {}
    dialog_state = diagnostics.get("dialog_state") or {}
    scope_resolution = diagnostics.get("scope_resolution") or {}
    lines = [
        "Тип записи: Внутреннее решение KPI-диалога",
        f"Намерение: {dialog_state.get('intent') or diagnostics.get('intent') or ''}".strip(),
    ]

    if dialog_state.get("position"):
        lines.append(f"Зафиксированная должность: {dialog_state['position']}")
    if dialog_state.get("department"):
        lines.append(f"Зафиксированное подразделение: {dialog_state['department']}")
    if dialog_state.get("requested_slot"):
        lines.append(f"Нужное уточнение: {dialog_state['requested_slot']}")

    missing_slots = scope_resolution.get("missing_slots") or []
    if missing_slots:
        lines.append(f"Недостающие параметры: {', '.join(str(slot) for slot in missing_slots if slot)}")

    primary_slot = scope_resolution.get("primary_discriminating_slot") or ""
    if primary_slot:
        lines.append(f"Главное уточнение: {primary_slot}")

    lines.append(f"Смысл ответа: {direct_answer}")
    return Document(
        page_content="\n".join(line for line in lines if line.strip()),
        metadata={
            "source": "kpi_dialog_engine",
            "internal_guidance": True,
            "guidance_kind": "kpi_dialog",
        },
    )


def prepare_kpi_generation_inputs(
    retrieval: KPIHybridResult,
    merge_prompts,
    base_extra_system_prompt: str = "",
) -> tuple[list[Document], str]:
    """Добавляет structured guidance, но оставляет финальную формулировку за моделью."""
    docs = list(retrieval.documents)
    guidance_doc = build_kpi_dialog_guidance_document(retrieval)
    if guidance_doc is None:
        return docs, base_extra_system_prompt

    extra_prompt = merge_prompts(
        base_extra_system_prompt,
        (
            "Если среди документов есть запись «Внутреннее решение KPI-диалога», "
            "используй ее как точную опору для смысла ответа, но не копируй формулировки дословно. "
            "Сформулируй ответ естественно, как живой собеседник. "
            "Если нужно уточнение, задай один короткий вопрос без канцелярита. "
            "Не упоминай внутренние слова вроде intent, slot, diagnostics или scope."
        ),
    )
    return [guidance_doc, *docs], extra_prompt
