from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

from langchain_core.documents import Document

from utils.kpi_retrieval import (
    _dedupe_preserve_order,
    _extract_labeled_fields,
    _human_scope_hint,
)

if TYPE_CHECKING:
    from utils.kpi_dialog_engine import KPIConversationState, KPIScopeResolution


_SLOT_LABELS = {
    "department": "подразделение",
    "center": "центр ответственности",
    "worker_group": "группу работников",
    "position_group": "группу должностей",
    "position": "должность",
}

_SLOT_TITLE_LABELS = {
    "department": "подразделения",
    "center": "центра ответственности",
    "worker_group": "группы работников",
    "position_group": "группы должностей",
    "position": "должности",
}


def _display_department(value: str) -> str:
    segments = [part.strip() for part in value.split(">") if part.strip()]
    return segments[-1] if segments else value.strip()


def _display_slot_value(slot_name: str, value: str) -> str:
    if slot_name == "department":
        return _display_department(value)
    return str(value or "").strip()


class KPIAnswerFormatter:
    def _known_context_prefix(
        self,
        state: "KPIConversationState",
        exclude_slot: str | None = None,
    ) -> str:
        parts: list[str] = []
        if state.position and exclude_slot != "position":
            parts.append(f"должность — «{state.position}»")
        if state.department and exclude_slot != "department":
            parts.append(f"подразделение — «{_display_department(state.department)}»")
        if not parts:
            return ""
        return f"Зафиксировал: {'; '.join(parts)}. "

    def format_meta_help(self, state: "KPIConversationState") -> str:
        return "Чтобы определить ваши KPI, укажите должность и подразделение."

    def format_term_context_clarification(self) -> str:
        return (
            "Уточните, пожалуйста: нужен смысл термина в рамках KPI/конкретного показателя "
            "или общее определение?"
        )

    def format_profile_confirmation(self, state: "KPIConversationState") -> str:
        return "Чтобы назвать ваши KPI, уточните, пожалуйста, должность и подразделение."

    def format_missing_context(self, state: "KPIConversationState", resolution: "KPIScopeResolution") -> str:
        missing = resolution.missing_slots or []
        if not missing:
            return "Чтобы назвать точные KPI, укажите должность и подразделение."

        primary = missing[0]
        if primary == "department" and state.position:
            return self._known_context_prefix(state, exclude_slot="department") + "Уточните, пожалуйста, подразделение."
        if primary == "position" and state.department:
            return self._known_context_prefix(state, exclude_slot="position") + "Уточните, пожалуйста, должность."
        if len(missing) >= 2:
            return "Чтобы назвать точные KPI, укажите должность и подразделение."
        return f"Чтобы назвать точные KPI, укажите {_SLOT_LABELS.get(primary, primary)}."

    def format_clarification(self, state: "KPIConversationState", resolution: "KPIScopeResolution") -> str:
        options = resolution.clarification_options or []
        slot_name = resolution.primary_discriminating_slot or ""
        slot_label = _SLOT_LABELS.get(slot_name, "уточнение")
        title_label = _SLOT_TITLE_LABELS.get(slot_name, slot_label)

        if not options:
            return self.format_missing_context(state, resolution)

        if state.position and slot_name == "department":
            prefix = self._known_context_prefix(state, exclude_slot="department")
        elif state.department and slot_name == "position":
            prefix = self._known_context_prefix(state, exclude_slot="position")
        else:
            prefix = "Чтобы назвать точные KPI, нужно уточнение. "

        if len(options) > 5:
            return f"{prefix}Уточните, пожалуйста, {slot_label}."

        formatted_options = _dedupe_preserve_order(_display_slot_value(slot_name, value) for value in options)
        if len(formatted_options) == 1:
            return f"{prefix}Уточните, пожалуйста: {formatted_options[0]}?"

        joined = ", ".join(formatted_options[:-1]) + f" или {formatted_options[-1]}"
        return f"{prefix}Уточните, пожалуйста, {slot_label}: {joined}?"

    def format_kpi_list(
        self,
        state: "KPIConversationState",
        resolution: "KPIScopeResolution",
        documents: Sequence[Document],
    ) -> str:
        position = state.position or ""
        department = state.department or ""
        department_label = _display_department(department)

        if position and department_label:
            header = f"Для должности «{position}» в «{department_label}» применимы следующие KPI:"
        elif position:
            header = f"Для должности «{position}» применимы следующие KPI:"
        elif department_label:
            header = f"Для подразделения «{department_label}» применимы следующие KPI:"
        else:
            header = "Применимы следующие KPI:"

        lines = [header, ""]
        for index, doc in enumerate(documents, start=1):
            metadata = doc.metadata or {}
            fields = _extract_labeled_fields(doc.page_content)
            kpi_name = str(metadata.get("kpi_name") or fields.get("KPI") or "").strip()
            detail = str(fields.get("Детализация расчета", "")).strip()
            periodicity = str(fields.get("Периодичность расчета", "")).strip()
            specifics = str(fields.get("Специфика расчета", "")).strip()
            extra_parts = [value for value in [detail, periodicity, specifics] if value and value != "-"]
            line = f"{index}. {kpi_name}"
            if extra_parts:
                line += f" ({'; '.join(extra_parts[:3])})"
            lines.append(line)

        lines.append("")
        lines.append("Если нужно, могу отдельно показать методику расчета по любому KPI из списка.")
        return "\n".join(lines).strip()

    def format_kpi_explain_clarification(self, options: Sequence[str]) -> str:
        formatted_options = _dedupe_preserve_order(str(option or "").strip() for option in options)
        if not formatted_options:
            return "Уточните, по какому KPI показать методику расчета. Можно написать название KPI."

        if len(formatted_options) <= 3:
            if len(formatted_options) == 1:
                return (
                    f"Уточните, по KPI «{formatted_options[0]}» показать методику расчета?"
                )
            joined = ", ".join(formatted_options[:-1]) + f" или {formatted_options[-1]}"
            return (
                f"Уточните, по какому KPI показать методику расчета: {joined}? "
                "Можно назвать номер или название."
            )

        return (
            "Уточните, по какому KPI показать методику расчета. "
            "Можно написать номер из последнего списка или название KPI."
        )

    def format_no_match(
        self,
        state: "KPIConversationState",
        resolution: "KPIScopeResolution" | None = None,
    ) -> str:
        suggested = ""
        if resolution and resolution.clarification_options:
            suggested = _display_slot_value(
                resolution.primary_discriminating_slot or "department",
                resolution.clarification_options[0],
            )

        if resolution and resolution.primary_discriminating_slot == "department":
            prefix = self._known_context_prefix(state, exclude_slot="department")
            if suggested:
                return (
                    f"{prefix}Такого подразделения в доступных данных не найдено. "
                    f"Ближайший вариант — «{suggested}». "
                    "Уточните точное название отдела. KPI перечислю только при точном совпадении названия."
                )
            return (
                f"{prefix}Такого подразделения в доступных данных не найдено. "
                "Уточните точное название отдела. KPI перечислю только при точном совпадении названия."
            )

        if state.position and state.department:
            return (
                f"Не удалось найти KPI для должности «{state.position}» в «{_display_department(state.department)}». "
                "Проверьте формулировку должности или подразделения."
            )
        return "Не удалось найти точный KPI-контур. Уточните должность и подразделение."

    def format_slot_options(
        self,
        state: "KPIConversationState",
        slot_name: str,
        options: Sequence[str],
        total_count: int | None = None,
    ) -> str:
        slot_label = _SLOT_LABELS.get(slot_name, slot_name)
        title_label = _SLOT_TITLE_LABELS.get(slot_name, slot_label)
        formatted_options = _dedupe_preserve_order(_display_slot_value(slot_name, value) for value in options)

        if not formatted_options:
            if slot_name == "department" and state.position:
                return (
                    f"Не удалось подобрать {title_label} для должности «{state.position}». "
                    "Проверьте формулировку должности."
                )
            if slot_name == "position" and state.department:
                return (
                    f"Не удалось подобрать {title_label} для подразделения "
                    f"«{_display_department(state.department)}». Проверьте формулировку подразделения."
                )
            return f"Не удалось подобрать варианты для слота «{slot_label}»."

        prefix = self._known_context_prefix(state, exclude_slot=slot_name)
        if slot_name == "department":
            intro = f"{prefix}Для этой должности в KPI-матрице есть такие подразделения:"
        elif slot_name == "position":
            intro = f"{prefix}Для этого подразделения в KPI-матрице есть такие должности:"
        else:
            intro = f"{prefix}Доступные варианты {slot_label}:"

        lines = [intro, ""]
        for index, option in enumerate(formatted_options, start=1):
            lines.append(f"{index}. {option}")
        if total_count and total_count > len(formatted_options):
            lines.append("")
            lines.append(f"Показываю первые {len(formatted_options)} из {total_count}.")
        lines.append("")
        lines.append(f"Укажите нужное {slot_label}, и я назову точные KPI.")
        return "\n".join(lines).strip()

    def format_known_slot_value(self, state: "KPIConversationState", slot_name: str | None) -> str:
        if slot_name == "position" and state.position:
            return f"Ваша должность — «{state.position}»."
        if slot_name == "department" and state.department:
            return f"Ваше подразделение — «{_display_department(state.department)}»."
        if slot_name == "position":
            return "В текущем диалоге должность пока не зафиксирована."
        if slot_name == "department":
            return "В текущем диалоге подразделение пока не зафиксировано."
        return "В текущем диалоге нужный параметр пока не зафиксирован."

    def format_scope_hint(self, resolution: "KPIScopeResolution") -> list[str]:
        return _dedupe_preserve_order(_human_scope_hint(scope) for scope in resolution.scopes)
