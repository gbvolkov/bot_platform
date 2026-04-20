from __future__ import annotations

import re
from datetime import datetime
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from src.config.settings import settings
from utils.kpi_answer_formatter import KPIAnswerFormatter
from utils.kpi_query_resolver import extract_last_listed_kpis
from utils.kpi_retrieval import (
    DialogMessage,
    KPICatalog,
    KPIHybridResult,
    KPIHybridRetriever,
    KPIScopeEntry,
    KPISlotValueEntry,
    _dedupe_preserve_order,
    _drop_current_user_message,
    _get_last_message,
    normalize_kpi_text,
)


_SLOT_ORDER = ("department", "position", "center", "worker_group", "position_group")
_FILTER_ORDER = ("position", "department", "center", "worker_group", "position_group")
_KEY_SLOTS = ("position", "department")
_LIGHT_STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "по",
    "для",
    "у",
    "к",
    "ко",
    "с",
    "со",
    "из",
    "от",
    "до",
    "или",
    "ли",
    "не",
    "что",
    "это",
    "как",
    "какие",
    "какой",
    "какая",
    "каковы",
    "я",
    "мы",
    "мой",
    "моя",
    "мое",
    "мои",
    "наш",
    "наша",
    "наше",
    "наши",
    "вопрос",
    "нужны",
    "нужно",
    "покажи",
    "перечисли",
    "назови",
    "список",
    "перечень",
    "kpi",
    "кпи",
    "кпэ",
}
_SOFT_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "иями",
    "ого",
    "ему",
    "ому",
    "его",
    "ыми",
    "ими",
    "иях",
    "ах",
    "ях",
    "ов",
    "ев",
    "ей",
    "ой",
    "ий",
    "ый",
    "ая",
    "ое",
    "ые",
    "ам",
    "ям",
    "ом",
    "ем",
    "ую",
    "юю",
    "ия",
    "ья",
    "ие",
    "ье",
    "ию",
    "ью",
    "иям",
    "ием",
    "ии",
    "а",
    "я",
    "у",
    "ю",
    "е",
    "ы",
    "и",
)
_META_HELP_RE = re.compile(
    r"\b(?:как\s+узнат|как\s+посмотр|как\s+определ|как\s+понят|"
    r"где\s+посмотр|что\s+нужн[оа]|что\s+указат)\b",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(
    r"^\s*(?:привет|здравствуй(?:те)?|добрый\s+(?:день|вечер|утро)|hello|hi)\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_PERSONAL_KPI_RE = re.compile(
    r"\b(?:мой|мои|моя|моё|мое|у\s+меня|для\s+меня)\b.*\b(?:kpi|кпи|кпэ|показател)",
    re.IGNORECASE,
)
_AFFIRM_RE = re.compile(r"^\s*(?:да|ага|угу|именно|верно|точно)\s*[!.?]*\s*$", re.IGNORECASE)
_POSITION_SLOT_RE = re.compile(
    r"\b(?:должност(?:ь|и|ью|ей)?|роль|роли|позици(?:я|и|ю|ей))\b|^\s*кто\s+я\b",
    re.IGNORECASE,
)
_DEPARTMENT_SLOT_RE = re.compile(
    r"\b(?:подразделен(?:ие|ия|ию|ием|ии)|отдел(?:а|е|ом)?|направлен(?:ие|ия|ию|ием|ии)|"
    r"департамент(?:а|е|ом)?|сектор(?:а|е|ом)?|офис(?:а|е|ом)?)\b",
    re.IGNORECASE,
)
_DEPARTMENT_VALUE_RE = re.compile(
    r"\b(?:подразделен(?:ие|ия|ию|ием|ии)|отдел(?:а|е|ом)?|направлен(?:ие|ия|ию|ием|ии)|"
    r"департамент(?:а|е|ом)?|сектор(?:а|е|ом)?|офис(?:а|е|ом)?)\b(?:\s+[0-9A-Za-zА-Яа-яЁё-]+){1,6}",
    re.IGNORECASE,
)
_SLOT_RECALL_RE = re.compile(
    r"\b(?:кака(?:я|ой|ое)|кто\s+я|напомни|повтори|скажи|подскажи)\b",
    re.IGNORECASE,
)
_SLOT_OPTIONS_RE = re.compile(
    r"\b(?:какие\s+есть|какие\s+бывают|что\s+есть|список|перечисли|вариант(?:ы|ов)?|не\s+помню)\b",
    re.IGNORECASE,
)
_TERM_QUERY_RE = re.compile(
    r"\b(?:что\s+такое|что\s+означает(?:\s+слово|\s+термин)?|что\s+значит|"
    r"значение\s+(?:слова|термина)|дай\s+определени\w*|объясни\s+термин|поясни\s+термин)\b",
    re.IGNORECASE,
)
_GENERAL_TERM_DEFINITION_RE = re.compile(
    r"\b(?:общее\s+определени\w*|общее\s+поняти\w*|обычн\w*\s+определени\w*|"
    r"в\s+общем|вообще|без\s+привязки\s+к\s+(?:kpi|кпи|кпэ)|"
    r"без\s+ссыл(?:ки|ок)\s+на\s+документ\w*|вне\s+(?:kpi|кпи|кпэ)|"
    r"не\s+в\s+рамках\s+(?:kpi|кпи|кпэ))\b",
    re.IGNORECASE,
)
_CALCULATION_HINT_RE = re.compile(
    r"\b(?:формул\w*|методик\w*|расч[её]т\w*|рассчит\w*|детализац\w*|нюанс\w*)\b",
    re.IGNORECASE,
)
_GENERIC_DEPARTMENT_TOKENS = {
    "отдел",
    "подразделен",
    "департамент",
    "сектор",
    "офис",
    "направлен",
    "управлен",
    "служб",
    "групп",
    "начальник",
    "директор",
    "заместител",
    "руководител",
    "специалист",
    "менеджер",
}


@dataclass
class KPIConversationState:
    intent: str | None
    raw_query: str = ""
    position: str | None = None
    department: str | None = None
    center: str | None = None
    worker_group: str | None = None
    position_group: str | None = None
    kpi_name: str | None = None
    raw_mentions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    slot_sources: Dict[str, str] = field(default_factory=dict)
    slot_timestamps: Dict[str, str] = field(default_factory=dict)
    current_slot_names: List[str] = field(default_factory=list)
    recent_slot_names: List[str] = field(default_factory=list)
    session_is_stale: bool = False
    profile_confirmed: bool = False
    stale_profile_detected: bool = False
    current_query_is_meta_help: bool = False
    current_query_is_personal: bool = False
    current_query_mentions_department: bool = False
    requested_slot: str | None = None
    latest_profile_update_at: str | None = None
    slot_exact_matches: Dict[str, bool] = field(default_factory=dict)
    profile: KPIProfileState = field(default_factory=lambda: KPIProfileState())
    stale_profile: KPIProfileState | None = None


@dataclass
class KPIProfileState:
    position: str | None = None
    department: str | None = None
    center: str | None = None
    worker_group: str | None = None
    position_group: str | None = None
    slot_sources: Dict[str, str] = field(default_factory=dict)
    slot_timestamps: Dict[str, str] = field(default_factory=dict)
    confirmed: bool = False
    updated_at: str | None = None
    stale: bool = False


@dataclass
class KPISlotMatch:
    slot_name: str
    value: str
    normalized_value: str
    score: float
    source: str
    scope_keys: Tuple[str, ...]
    timestamp: str | None = None
    is_exact: bool = False
    matched_alias: str | None = None


@dataclass
class KPIScopeResolution:
    status: str
    scopes: List[KPIScopeEntry] = field(default_factory=list)
    missing_slots: List[str] = field(default_factory=list)
    discriminating_slots: List[str] = field(default_factory=list)
    primary_discriminating_slot: str = ""
    clarification_options: List[str] = field(default_factory=list)


def _token_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _soft_stem(token: str) -> str:
    token = str(token or "").casefold().replace("ё", "е")
    for suffix in _SOFT_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _slot_tokens(value: str) -> set[str]:
    tokens = [token.casefold().replace("ё", "е") for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", str(value or ""))]
    return {
        _soft_stem(token)
        for token in tokens
        if token not in _LIGHT_STOPWORDS and len(token) > 1
    }


def _display_department(value: str) -> str:
    segments = [part.strip() for part in str(value or "").split(">") if part.strip()]
    return segments[-1] if segments else str(value or "").strip()


def _parse_timestamp(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _split_current_message(
    dialog_messages: Sequence[DialogMessage],
    current_query: str,
) -> tuple[str | None, List[DialogMessage]]:
    remaining: List[DialogMessage] = []
    skipped = False
    current_timestamp: str | None = None
    current_query = (current_query or "").strip()

    for message_text, role, timestamp in dialog_messages:
        if not skipped and role == "user" and (message_text or "").strip() == current_query:
            skipped = True
            current_timestamp = timestamp
            continue
        remaining.append((message_text, role, timestamp))

    return current_timestamp, remaining


class KPISlotExtractor:
    def __init__(self, catalog: KPICatalog):
        self.catalog = catalog

    def extract(
        self,
        text: str,
        source: str,
        min_score: float = 2.4,
        max_per_slot: int = 3,
    ) -> List[KPISlotMatch]:
        query_norm = normalize_kpi_text(text)
        if not query_norm:
            return []

        query_tokens = _slot_tokens(text)
        matches: List[KPISlotMatch] = []

        for slot_name, entries in self.catalog.slot_entries.items():
            slot_matches: List[KPISlotMatch] = []
            for entry in entries:
                score = self._score_entry(query_norm, query_tokens, entry)
                if score < min_score:
                    continue
                exact_alias = self._detect_exact_alias(query_norm, entry)
                slot_matches.append(
                    KPISlotMatch(
                        slot_name=slot_name,
                        value=entry.canonical_value,
                        normalized_value=entry.normalized_value,
                        score=score,
                        source=source,
                        scope_keys=entry.scope_keys,
                        is_exact=bool(exact_alias),
                        matched_alias=exact_alias,
                    )
                )
            slot_matches.sort(key=lambda item: item.score, reverse=True)
            matches.extend(slot_matches[:max_per_slot])

        return matches

    @staticmethod
    def _detect_exact_alias(query_norm: str, entry: KPISlotValueEntry) -> str | None:
        padded_query = f" {query_norm} "
        for alias, alias_norm in zip(entry.aliases, entry.normalized_aliases):
            if not alias_norm:
                continue
            if query_norm == alias_norm or f" {alias_norm} " in padded_query:
                return alias
        return None

    def _score_entry(
        self,
        query_norm: str,
        query_tokens: set[str],
        entry: KPISlotValueEntry,
    ) -> float:
        best_score = 0.0
        query_len = len(query_norm)

        for alias_norm in entry.normalized_aliases:
            if not alias_norm:
                continue

            alias_tokens = _slot_tokens(alias_norm)
            score = 0.0
            matched_alias_tokens: set[str] = set()

            if alias_norm in query_norm:
                score += 11.0
            elif query_len >= 8 and query_norm in alias_norm:
                score += 7.0

            overlap = query_tokens & alias_tokens
            if overlap:
                matched_alias_tokens.update(overlap)
                score += sum(self.catalog.token_idf.get(token, 1.0) for token in overlap) * 1.65
                if len(overlap) >= 2:
                    score += 2.0

            for query_token in query_tokens:
                best_ratio = 0.0
                best_alias_token = ""
                for alias_token in alias_tokens:
                    ratio = _token_similarity(query_token, alias_token)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_alias_token = alias_token
                if best_ratio >= 0.84 and best_alias_token and best_alias_token not in matched_alias_tokens:
                    matched_alias_tokens.add(best_alias_token)
                    score += self.catalog.token_idf.get(best_alias_token, 1.0) * best_ratio * 1.15

            unmatched_alias_tokens = max(0, len(alias_tokens - matched_alias_tokens))
            if unmatched_alias_tokens:
                score -= unmatched_alias_tokens * 1.35

            if alias_tokens and matched_alias_tokens:
                coverage = len(matched_alias_tokens) / len(alias_tokens)
                score *= 0.8 + coverage * 0.4

            whole_ratio = _token_similarity(query_norm, alias_norm)
            if min(len(alias_norm), query_len) >= 10 and whole_ratio >= 0.9:
                score += whole_ratio * 4.0

            best_score = max(best_score, score)

        return best_score


class KPIDialogResolver:
    def __init__(self, catalog: KPICatalog, slot_extractor: KPISlotExtractor, retriever: KPIHybridRetriever):
        self.catalog = catalog
        self.slot_extractor = slot_extractor
        self.retriever = retriever
        self.session_ttl_minutes = max(5, int(settings.KPI_DIALOG_SESSION_TTL_MINUTES))

    def resolve_state(
        self,
        query: str,
        dialog_messages: Sequence[DialogMessage],
    ) -> KPIConversationState:
        analysis = self.retriever.analyze_query(query, dialog_messages)
        raw_query = (query or "").strip()
        current_timestamp, history = _split_current_message(dialog_messages, raw_query)
        current_dt = _parse_timestamp(current_timestamp)
        recent_history, recent_user_messages, stale_user_messages, session_is_stale = self._partition_user_history(
            history,
            current_dt,
        )

        current_by_slot = self._extract_best_matches(
            raw_query,
            source="current_message",
            timestamp=current_timestamp,
        )
        recent_by_slot = self._recent_history_matches(recent_user_messages, source="recent_history")
        best_matches: Dict[str, KPISlotMatch] = dict(current_by_slot)

        for slot_name, match in recent_by_slot.items():
            best_matches.setdefault(slot_name, match)

        current_slot_names = sorted(current_by_slot)
        recent_slot_names = sorted(
            slot_name for slot_name, match in best_matches.items() if match.source == "recent_history"
        )
        current_query_is_meta_help = self._is_meta_help_query(raw_query)
        current_query_is_personal = self._is_personal_query(raw_query)
        profile = self._build_profile(
            best_matches,
            recent_history,
            current_slot_names,
            session_is_stale,
        )
        stale_profile = self._build_stale_profile(stale_user_messages)

        state = KPIConversationState(
            intent=None,
            raw_query=raw_query,
            position=self._slot_value(best_matches, "position"),
            department=self._slot_value(best_matches, "department"),
            center=self._slot_value(best_matches, "center"),
            worker_group=self._slot_value(best_matches, "worker_group"),
            position_group=self._slot_value(best_matches, "position_group"),
            kpi_name=analysis.explicit_kpi_name,
            raw_mentions=[match.value for match in best_matches.values()],
            confidence=sum(match.score for match in best_matches.values()),
            slot_sources={slot_name: match.source for slot_name, match in best_matches.items()},
            slot_timestamps={
                slot_name: str(getattr(match, "timestamp", "") or "")
                for slot_name, match in best_matches.items()
                if getattr(match, "timestamp", None)
            },
            current_slot_names=current_slot_names,
            recent_slot_names=recent_slot_names,
            session_is_stale=session_is_stale,
            profile_confirmed=profile.confirmed,
            stale_profile_detected=bool(
                stale_profile and any(getattr(stale_profile, slot_name) for slot_name in _KEY_SLOTS)
            ),
            current_query_is_meta_help=current_query_is_meta_help,
            current_query_is_personal=current_query_is_personal,
            current_query_mentions_department=self._query_attempts_department_value(raw_query),
            requested_slot=None,
            latest_profile_update_at=profile.updated_at,
            slot_exact_matches={slot_name: match.is_exact for slot_name, match in best_matches.items()},
            profile=profile,
            stale_profile=stale_profile,
        )
        state.requested_slot = self._infer_requested_slot(raw_query, state)
        current_query_is_general_term_definition = self._is_general_term_definition_query(raw_query)
        current_query_is_term_query = self._is_term_query(raw_query)

        if current_query_is_general_term_definition:
            state.intent = "general_term_definition"
        elif current_query_is_term_query and not analysis.explicit_kpi_name:
            state.intent = "clarify_term_context"
        elif analysis.intent == "explain_kpi" or analysis.explicit_kpi_name:
            state.intent = "explain_kpi"
        elif self._is_slot_recall_query(raw_query, state.requested_slot):
            state.intent = "recall_known_slot"
        elif self._is_slot_options_query(raw_query, state.requested_slot):
            state.intent = "list_slot_options"
        elif current_query_is_meta_help:
            state.intent = "meta_help"
        elif analysis.intent == "list_kpi" or best_matches:
            state.intent = "list_kpi"
        elif current_query_is_personal:
            state.intent = "meta_help"
        else:
            state.intent = analysis.intent

        return state

    def _extract_best_matches(
        self,
        text: str,
        source: str,
        timestamp: str | None = None,
    ) -> Dict[str, KPISlotMatch]:
        if not (text or "").strip():
            return {}
        matches = self._filter_matches(
            self.slot_extractor.extract(text, source=source),
            guard_query=text,
        )
        best = self._best_matches_by_slot(matches)
        extracted: Dict[str, KPISlotMatch] = {}
        for slot_name, match in best.items():
            extracted[slot_name] = KPISlotMatch(
                slot_name=match.slot_name,
                value=match.value,
                normalized_value=match.normalized_value,
                score=match.score,
                source=source,
                scope_keys=match.scope_keys,
                timestamp=timestamp,
                is_exact=match.is_exact,
                matched_alias=match.matched_alias,
            )
        return extracted

    def _recent_history_matches(
        self,
        messages: Sequence[DialogMessage],
        source: str,
    ) -> Dict[str, KPISlotMatch]:
        best: Dict[str, KPISlotMatch] = {}
        for message_text, role, timestamp in messages:
            if role != "user" or not (message_text or "").strip():
                continue
            for slot_name, match in self._extract_best_matches(message_text, source=source, timestamp=timestamp).items():
                best.setdefault(slot_name, match)
        return best

    @staticmethod
    def _best_matches_by_slot(matches: Sequence[KPISlotMatch]) -> Dict[str, KPISlotMatch]:
        best: Dict[str, KPISlotMatch] = {}
        for match in matches:
            existing = best.get(match.slot_name)
            if (
                existing is None
                or match.score > existing.score
                or (match.is_exact and not existing.is_exact and match.score >= existing.score - 1.0)
            ):
                best[match.slot_name] = match
        return best

    def _filter_matches(
        self,
        matches: Sequence[KPISlotMatch],
        guard_query: str,
    ) -> List[KPISlotMatch]:
        filtered: List[KPISlotMatch] = []
        for match in matches:
            if match.slot_name == "position" and self._match_is_negated(match, guard_query):
                continue
            filtered.append(match)
        return filtered

    @staticmethod
    def _detect_explicit_slot(query: str) -> str | None:
        matched_slots: List[str] = []
        raw_query = str(query or "")
        if _POSITION_SLOT_RE.search(raw_query):
            matched_slots.append("position")
        if _DEPARTMENT_SLOT_RE.search(raw_query):
            matched_slots.append("department")
        if len(matched_slots) == 1:
            return matched_slots[0]
        return None

    def _infer_requested_slot(self, query: str, state: KPIConversationState) -> str | None:
        explicit_slot = self._detect_explicit_slot(query)
        if explicit_slot:
            return explicit_slot

        raw_query = str(query or "")
        if _SLOT_RECALL_RE.search(raw_query) or _SLOT_OPTIONS_RE.search(raw_query):
            missing_slots = [slot_name for slot_name in _KEY_SLOTS if not getattr(state, slot_name)]
            if len(missing_slots) == 1:
                return missing_slots[0]
        return None

    @staticmethod
    def _is_slot_recall_query(query: str, requested_slot: str | None) -> bool:
        if requested_slot is None:
            return False
        return bool(_SLOT_RECALL_RE.search(str(query or "")))

    @staticmethod
    def _is_slot_options_query(query: str, requested_slot: str | None) -> bool:
        if requested_slot is None:
            return False
        return bool(_SLOT_OPTIONS_RE.search(str(query or "")))

    @staticmethod
    def _query_mentions_role(query: str) -> bool:
        return bool(
            re.search(
                r"\b(?:должност|сотрудник|исполнител|руководител|начальник|менеджер|специалист|агент|директор|куратор)\b",
                str(query or ""),
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _negated_tokens(query: str) -> set[str]:
        tokens: set[str] = set()
        for negated in re.findall(
            r"\bне\s+([0-9A-Za-zА-Яа-яЁё-]+(?:\s+[0-9A-Za-zА-Яа-яЁё-]+){0,3})",
            str(query or ""),
            flags=re.IGNORECASE,
        ):
            tokens |= _slot_tokens(negated)
        return tokens

    def _match_is_negated(self, match: KPISlotMatch, query: str) -> bool:
        negated_tokens = self._negated_tokens(query)
        if not negated_tokens:
            return False
        return bool(_slot_tokens(match.value) & negated_tokens)

    @staticmethod
    def _merge_slot_matches(
        best_matches: Dict[str, KPISlotMatch],
        matches: Sequence[KPISlotMatch],
        source_bonus: float,
        source_name: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        for match in matches:
            adjusted = KPISlotMatch(
                slot_name=match.slot_name,
                value=match.value,
                normalized_value=match.normalized_value,
                score=match.score + source_bonus,
                source=source_name or match.source,
                scope_keys=match.scope_keys,
                timestamp=timestamp or match.timestamp,
                is_exact=match.is_exact,
                matched_alias=match.matched_alias,
            )
            existing = best_matches.get(match.slot_name)
            if existing is None or adjusted.score > existing.score:
                best_matches[match.slot_name] = adjusted

    @staticmethod
    def _slot_value(matches: Dict[str, KPISlotMatch], slot_name: str) -> str | None:
        match = matches.get(slot_name)
        return match.value if match else None

    def _partition_user_history(
        self,
        history: Sequence[DialogMessage],
        current_dt: datetime | None,
    ) -> tuple[List[DialogMessage], List[DialogMessage], List[DialogMessage], bool]:
        recent_history: List[DialogMessage] = []
        recent: List[DialogMessage] = []
        stale: List[DialogMessage] = []
        session_is_stale = False
        boundary_hit = False

        for index, (message_text, role, timestamp) in enumerate(history):
            raw_text = (message_text or "").strip()
            message_dt = _parse_timestamp(timestamp)
            age_minutes = (current_dt - message_dt).total_seconds() / 60.0 if current_dt and message_dt else 0.0
            is_stale = bool(current_dt and message_dt and age_minutes > self.session_ttl_minutes)
            if is_stale:
                session_is_stale = True
                stale.extend(
                    [
                        item
                        for item in history[index:]
                        if item[1] == "user" and (item[0] or "").strip()
                    ]
                )
                boundary_hit = True
                break

            recent_history.append((message_text, role, timestamp))
            if role == "user" and raw_text:
                recent.append((message_text, role, timestamp))
                if _GREETING_RE.match(raw_text):
                    older_user_messages = [
                        item
                        for item in history[index + 1 :]
                        if item[1] == "user" and (item[0] or "").strip()
                    ]
                    if older_user_messages:
                        stale.extend(older_user_messages)
                        session_is_stale = True
                    boundary_hit = True
                    break

        if not boundary_hit:
            stale = []

        return recent_history, recent, stale, session_is_stale

    def _build_profile(
        self,
        matches: Dict[str, KPISlotMatch],
        history: Sequence[DialogMessage],
        current_slot_names: Sequence[str],
        session_is_stale: bool,
    ) -> KPIProfileState:
        profile = KPIProfileState(
            position=self._slot_value(matches, "position"),
            department=self._slot_value(matches, "department"),
            center=self._slot_value(matches, "center"),
            worker_group=self._slot_value(matches, "worker_group"),
            position_group=self._slot_value(matches, "position_group"),
            slot_sources={slot_name: match.source for slot_name, match in matches.items()},
            slot_timestamps={
                slot_name: str(getattr(match, "timestamp", "") or "")
                for slot_name, match in matches.items()
                if getattr(match, "timestamp", None)
            },
            stale=session_is_stale,
        )
        key_sources = {slot_name: profile.slot_sources.get(slot_name, "") for slot_name in _KEY_SLOTS}
        has_all_key_slots = all(getattr(profile, slot_name) for slot_name in _KEY_SLOTS)
        has_current_key_slot = any(slot_name in current_slot_names for slot_name in _KEY_SLOTS)
        assistant_confirms = self._assistant_confirms_profile(history, profile)
        profile.confirmed = bool(
            has_all_key_slots
            and all(source in {"current_message", "recent_history"} for source in key_sources.values())
            and (has_current_key_slot or assistant_confirms)
        )
        if assistant_confirms and has_all_key_slots and not has_current_key_slot:
            for slot_name in _KEY_SLOTS:
                if key_sources.get(slot_name) == "recent_history":
                    profile.slot_sources[slot_name] = "confirmed_profile"
        timestamps = [value for value in profile.slot_timestamps.values() if value]
        if timestamps:
            profile.updated_at = max(timestamps)
        return profile

    def _build_stale_profile(
        self,
        stale_user_messages: Sequence[DialogMessage],
    ) -> KPIProfileState | None:
        stale_matches = self._recent_history_matches(stale_user_messages, source="stale_history")
        if not stale_matches:
            return None
        return self._build_profile(
            stale_matches,
            stale_user_messages,
            current_slot_names=(),
            session_is_stale=True,
        )

    @staticmethod
    def _assistant_confirms_profile(history: Sequence[DialogMessage], profile: KPIProfileState) -> bool:
        if not profile.position or not profile.department:
            return False
        position_norm = normalize_kpi_text(profile.position)
        department_norm = normalize_kpi_text(profile.department)
        for message_text, role, _ in history:
            if role != "assistant" or not (message_text or "").strip():
                continue
            message_norm = normalize_kpi_text(message_text)
            if position_norm and position_norm in message_norm and department_norm and department_norm in message_norm:
                return True
        return False

    @staticmethod
    def _is_meta_help_query(query: str) -> bool:
        raw_query = (query or "").strip()
        if not raw_query:
            return False
        return bool(_META_HELP_RE.search(raw_query) and re.search(r"\b(?:kpi|кпи|кпэ|показател)", raw_query, flags=re.IGNORECASE))

    @staticmethod
    def _is_personal_query(query: str) -> bool:
        raw_query = (query or "").strip()
        if not raw_query:
            return False
        return bool(_PERSONAL_KPI_RE.search(raw_query))

    @staticmethod
    def _query_attempts_department_value(query: str) -> bool:
        raw_query = (query or "").strip()
        if not raw_query:
            return False
        return bool(_DEPARTMENT_VALUE_RE.search(raw_query))

    @staticmethod
    def _is_term_query(query: str) -> bool:
        raw_query = (query or "").strip()
        if not raw_query:
            return False
        if _CALCULATION_HINT_RE.search(raw_query):
            return False
        return bool(_TERM_QUERY_RE.search(raw_query))

    @staticmethod
    def _is_general_term_definition_query(query: str) -> bool:
        raw_query = (query or "").strip()
        if not raw_query:
            return False
        return bool(_GENERAL_TERM_DEFINITION_RE.search(raw_query))


class KPIScopeResolver:
    def __init__(self, catalog: KPICatalog):
        self.catalog = catalog

    def resolve(self, state: KPIConversationState) -> KPIScopeResolution:
        if not state.position and not state.department:
            return KPIScopeResolution(status="need_more_context", missing_slots=["position", "department"])
        if not state.position:
            return KPIScopeResolution(status="need_more_context", missing_slots=["position"])
        if state.current_query_mentions_department and "department" not in state.current_slot_names:
            suggestion = self.closest_slot_option(state, "department", state.raw_query)
            return KPIScopeResolution(
                status="not_found",
                primary_discriminating_slot="department",
                clarification_options=[suggestion] if suggestion else [],
            )
        if not state.department:
            return KPIScopeResolution(status="need_more_context", missing_slots=["department"])
        if not state.slot_exact_matches.get("department"):
            suggestion = self.closest_slot_option(
                state,
                "department",
                state.raw_query,
                fallback=state.department,
            )
            return KPIScopeResolution(
                status="not_found",
                primary_discriminating_slot="department",
                clarification_options=[suggestion] if suggestion else [],
            )

        candidate_keys = set(self.catalog.scopes)

        for slot_name in _KEY_SLOTS:
            value = getattr(state, slot_name)
            if not value:
                continue
            normalized_value = normalize_kpi_text(value)
            scope_keys = set(self.catalog.slot_value_index.get(slot_name, {}).get(normalized_value, ()))
            if not scope_keys:
                return KPIScopeResolution(status="not_found")
            candidate_keys &= scope_keys
            if not candidate_keys:
                return KPIScopeResolution(status="not_found")

        scopes = [self.catalog.scopes[scope_key] for scope_key in sorted(candidate_keys)]
        if len(scopes) == 1:
            return KPIScopeResolution(status="resolved", scopes=scopes)

        fingerprints = {
            self.catalog.scope_fingerprints.get(scope.scope_key, ())
            for scope in scopes
        }
        if len(fingerprints) == 1:
            return KPIScopeResolution(status="resolved", scopes=scopes)

        discriminating_slots = self._discriminating_slots(scopes)
        missing_slots = [
            slot_name
            for slot_name in _SLOT_ORDER
            if slot_name in discriminating_slots and not getattr(state, slot_name)
        ]
        primary_slot = missing_slots[0] if missing_slots else (discriminating_slots[0] if discriminating_slots else "")
        options = self._clarification_options(scopes, primary_slot)
        return KPIScopeResolution(
            status="ambiguous",
            scopes=scopes,
            missing_slots=missing_slots,
            discriminating_slots=discriminating_slots,
            primary_discriminating_slot=primary_slot,
            clarification_options=options,
        )

    def slot_options(self, state: KPIConversationState, slot_name: str) -> List[str]:
        scopes = self._candidate_scopes(state, exclude_slots={slot_name})
        return self._clarification_options(scopes, slot_name)

    def closest_slot_option(
        self,
        state: KPIConversationState,
        slot_name: str,
        query: str,
        fallback: str | None = None,
    ) -> str | None:
        options = self.slot_options(state, slot_name)
        if fallback:
            options = [fallback, *options]

        best_option = ""
        best_score = 0.0
        for option in _dedupe_preserve_order(options):
            score = self._score_slot_option(query, option, slot_name)
            if score > best_score:
                best_option = option
                best_score = score

        if best_option and best_score >= 5.0:
            return best_option

        if fallback and normalize_kpi_text(fallback):
            return fallback
        return None

    def _candidate_scopes(
        self,
        state: KPIConversationState,
        exclude_slots: set[str] | None = None,
    ) -> List[KPIScopeEntry]:
        excluded = exclude_slots or set()
        candidate_keys = set(self.catalog.scopes)
        constrained = False

        for slot_name in _FILTER_ORDER:
            if slot_name in excluded:
                continue
            value = getattr(state, slot_name)
            if not value:
                continue
            normalized_value = normalize_kpi_text(value)
            scope_keys = set(self.catalog.slot_value_index.get(slot_name, {}).get(normalized_value, ()))
            if not scope_keys:
                return []
            candidate_keys &= scope_keys
            constrained = True
            if not candidate_keys:
                return []

        if not constrained:
            return [self.catalog.scopes[scope_key] for scope_key in sorted(self.catalog.scopes)]
        return [self.catalog.scopes[scope_key] for scope_key in sorted(candidate_keys)]

    def documents_for_resolution(self, resolution: KPIScopeResolution) -> List[Document]:
        merged: Dict[str, Document] = {}
        for scope in resolution.scopes:
            for entry in self.catalog.scope_documents.get(scope.scope_key, []):
                merged.setdefault(entry.kpi_name_norm, entry.document)
        return sorted(
            merged.values(),
            key=lambda doc: (
                int(doc.metadata.get("sheet_row_min") or 0),
                normalize_kpi_text(doc.metadata.get("kpi_name") or ""),
            ),
        )

    @staticmethod
    def _scope_value(scope: KPIScopeEntry, slot_name: str) -> str:
        if slot_name == "department":
            return scope.department_path
        if slot_name == "center":
            return scope.center
        if slot_name == "worker_group":
            return scope.worker_group
        if slot_name == "position_group":
            return scope.position_group
        if slot_name == "position":
            return scope.position
        return ""

    def _discriminating_slots(self, scopes: Sequence[KPIScopeEntry]) -> List[str]:
        discriminating: List[str] = []
        for slot_name in _SLOT_ORDER:
            values = {
                normalize_kpi_text(self._scope_value(scope, slot_name))
                for scope in scopes
                if self._scope_value(scope, slot_name)
            }
            if len(values) > 1:
                discriminating.append(slot_name)
        return discriminating

    def _clarification_options(self, scopes: Sequence[KPIScopeEntry], slot_name: str) -> List[str]:
        options: List[str] = []
        for scope in scopes:
            value = self._scope_value(scope, slot_name)
            if value:
                options.append(value)
        deduped: List[str] = []
        seen: set[str] = set()
        for option in options:
            key = normalize_kpi_text(option)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(option)
        return deduped

    @staticmethod
    def _meaningful_slot_tokens(value: str, slot_name: str) -> set[str]:
        tokens = _slot_tokens(value)
        if slot_name == "department":
            return {token for token in tokens if token not in _GENERIC_DEPARTMENT_TOKENS}
        return tokens

    def _score_slot_option(self, query: str, option: str, slot_name: str) -> float:
        query_norm = normalize_kpi_text(query)
        option_label = _display_department(option) if slot_name == "department" else str(option or "").strip()
        option_norm = normalize_kpi_text(option_label)
        if not query_norm or not option_norm:
            return 0.0

        score = max(
            _token_similarity(query_norm, normalize_kpi_text(candidate))
            for candidate in (option, option_label)
            if normalize_kpi_text(candidate)
        ) * 4.0

        query_tokens = self._meaningful_slot_tokens(query, slot_name)
        option_tokens = self._meaningful_slot_tokens(option_label, slot_name)
        overlap = query_tokens & option_tokens
        if overlap:
            score += 4.0 + len(overlap) * 1.75

        for query_token in query_tokens:
            best_ratio = max((_token_similarity(query_token, option_token) for option_token in option_tokens), default=0.0)
            if best_ratio >= 0.72:
                score += best_ratio * 1.35

        if query_tokens and option_tokens and overlap:
            score += len(overlap) / max(len(option_tokens), 1)

        return score


class KPIDialogEngine:
    def __init__(
        self,
        catalog: KPICatalog,
        retriever: KPIHybridRetriever,
        formatter: KPIAnswerFormatter | None = None,
    ):
        self.catalog = catalog
        self.retriever = retriever
        self.formatter = formatter or KPIAnswerFormatter()
        self.slot_extractor = KPISlotExtractor(catalog)
        self.dialog_resolver = KPIDialogResolver(catalog, self.slot_extractor, retriever)
        self.scope_resolver = KPIScopeResolver(catalog)

    def handle(
        self,
        query: str,
        dialog_messages: Sequence[DialogMessage] = (),
        n_results: int = 6,
    ) -> KPIHybridResult:
        state = self.dialog_resolver.resolve_state(query, dialog_messages)
        diagnostics: Dict[str, object] = {
            "dialog_state": {
                "intent": state.intent,
                "raw_query": state.raw_query,
                "position": state.position or "",
                "department": state.department or "",
                "center": state.center or "",
                "worker_group": state.worker_group or "",
                "position_group": state.position_group or "",
                "kpi_name": state.kpi_name or "",
                "confidence": round(state.confidence, 2),
                "slot_sources": dict(state.slot_sources),
                "session_is_stale": state.session_is_stale,
                "profile_confirmed": state.profile_confirmed,
                "stale_profile_detected": state.stale_profile_detected,
                "current_slot_names": list(state.current_slot_names),
                "recent_slot_names": list(state.recent_slot_names),
                "slot_exact_matches": dict(state.slot_exact_matches),
                "current_query_mentions_department": state.current_query_mentions_department,
                "requested_slot": state.requested_slot or "",
                "latest_profile_update_at": state.latest_profile_update_at or "",
            }
        }

        if state.intent == "meta_help":
            return KPIHybridResult(
                documents=[],
                direct_answer=self.formatter.format_meta_help(state),
                diagnostics=diagnostics,
            )

        if state.intent == "clarify_term_context":
            return KPIHybridResult(
                documents=[],
                direct_answer=self.formatter.format_term_context_clarification(),
                diagnostics=diagnostics,
            )

        if state.intent == "recall_known_slot":
            return KPIHybridResult(
                documents=[],
                direct_answer=self.formatter.format_known_slot_value(state, state.requested_slot),
                diagnostics=diagnostics,
            )

        if state.intent == "list_slot_options":
            requested_slot = state.requested_slot or ""
            options = self.scope_resolver.slot_options(state, requested_slot) if requested_slot else []
            shown_options = options[:10]
            diagnostics["slot_options"] = {
                "slot_name": requested_slot,
                "count": len(options),
            }
            return KPIHybridResult(
                documents=[],
                direct_answer=self.formatter.format_slot_options(
                    state,
                    requested_slot,
                    shown_options,
                    total_count=len(options),
                ),
                diagnostics=diagnostics,
            )

        if state.intent == "explain_kpi" and not state.kpi_name:
            recent_kpi_options = extract_last_listed_kpis(dialog_messages)
            diagnostics["explain_kpi_clarification"] = {
                "options_count": len(recent_kpi_options),
            }
            return KPIHybridResult(
                documents=[],
                direct_answer=self.formatter.format_kpi_explain_clarification(recent_kpi_options),
                diagnostics=diagnostics,
            )

        if state.intent == "list_kpi":
            resolution = self.scope_resolver.resolve(state)
            diagnostics["scope_resolution"] = {
                "status": resolution.status,
                "scopes": [scope.scope_key for scope in resolution.scopes],
                "missing_slots": list(resolution.missing_slots),
                "discriminating_slots": list(resolution.discriminating_slots),
                "primary_discriminating_slot": resolution.primary_discriminating_slot,
            }

            if resolution.status == "resolved":
                documents = self.scope_resolver.documents_for_resolution(resolution)
                return KPIHybridResult(
                    documents=documents,
                    direct_answer=self.formatter.format_kpi_list(state, resolution, documents),
                    diagnostics=diagnostics,
                )

            if resolution.status == "ambiguous":
                return KPIHybridResult(
                    documents=[],
                    direct_answer=self.formatter.format_clarification(state, resolution),
                    diagnostics=diagnostics,
                )

            if resolution.status == "need_more_context":
                return KPIHybridResult(
                    documents=[],
                    direct_answer=self.formatter.format_missing_context(state, resolution),
                    diagnostics=diagnostics,
                )

            if resolution.status == "not_found" and any(
                [state.position, state.department, state.center, state.worker_group, state.position_group]
            ):
                return KPIHybridResult(
                    documents=[],
                    direct_answer=self.formatter.format_no_match(state, resolution),
                    diagnostics=diagnostics,
                )

        retrieval = self.retriever.retrieve(query=query, dialog_messages=dialog_messages, n_results=n_results)
        retrieval.diagnostics = {**diagnostics, **retrieval.diagnostics}
        return retrieval
