from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages

from langchain_core.messages import AIMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig

from ..llm_utils import get_llm
from ..utils import ModelType
from platform_utils.llm_logger import JSONFileTracer
from services.kb_manager.notifications import KBReloadContext, register_reload_listener

import config as cfg

from ..state.state import ConfigSchema
from .utilities.hh_search import search_vacancies, resolve_area_id

#USE_SEMANTIC_SCORE = True
from .scorers.semantic_job_scorer import semantic_score_job, SemanticScoreWeights

from utils.translate import translate_deepl

from .utilities.feature_extractor import extract_features_from_resume

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
#logger = logging.getLogger(__name__)


class JobAgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    resume_text: str
    extracted_features: Dict[str, Any]
    job_candidates: List[Dict[str, Any]]
    ranked_jobs: List[Dict[str, Any]]


DEFAULT_QUERY_PER_PAGE = 20
MAX_POSITION_QUERIES = 7
MAX_LOCATION_QUERIES = 4
MAX_TOTAL_RESULTS = 120
HH_QUERY_TEMPLATE = "({terms})"


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"[A-Za-zÀ-ÿ\u0400-\u04FF0-9]+", text.lower())


def _convert_salary(salary_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(salary_info, dict):
        return {}
    currency = salary_info.get("currency") or "RUB"
    return {
        "min": salary_info.get("from"),
        "max": salary_info.get("to"),
        "currency": currency,
    }


def _normalise_vacancy(item: Dict[str, Any]) -> Dict[str, Any]:
    employer = item.get("employer") or {}
    area = item.get("area") or {}
    address = item.get("address") or {}

    location = area.get("name") or "" if isinstance(area, dict) else area or ""
    if not location and isinstance(address, dict):
        location = address.get("raw") or address.get("city") or ""

    raw_skills = item.get("key_skills") or []
    skills: List[str] = []
    for skill in raw_skills:
        name = skill.get("name") if isinstance(skill, dict) else skill
        if isinstance(name, str) and name.strip():
            skills.append(name.strip())

    snippet = item.get("snippet") or {}
    description = " ".join(
        part.strip()
        for part in (snippet.get("requirement"), snippet.get("responsibility"))
        if isinstance(part, str) and part.strip()
    )

    experience_field = item.get("experience") or {}
    if isinstance(experience_field, dict):
        experience = experience_field.get("name") or ""
    elif isinstance(experience_field, str):
        experience = experience_field
    else:
        experience = ""

    match_score = item.get("score") or item.get("sort_point_distance") or item.get("match_score") or 0.0
    try:
        match_score = float(match_score)
    except (TypeError, ValueError):
        match_score = 0.0

    employer_name = employer.get("name") if isinstance(employer, dict) else employer

    return {
        "id": item.get("id"),
        "title": item.get("name"),
        "company": employer_name,
        "location": location,
        "salary": _convert_salary(item.get("salary")),
        "url": item.get("alternate_url") or item.get("url"),
        "skills": skills,
        "match_score": match_score,
        "source": "hh.ru",
        "description": description,
        "experience": experience,
        "published_at": item.get("published_at"),
    }


def _resolve_area_ids(locations: List[str]) -> Dict[str, Optional[str]]:
    resolved: Dict[str, Optional[str]] = {}
    for location in locations:
        if not location or location in resolved:
            continue
        try:
            resolved[location] = resolve_area_id(location)
        except Exception as exc:  # pragma: no cover - network or API errors
            logging.warning("job_lookup: failed to resolve hh.ru area '%s': %s", location, exc)
            resolved[location] = None
    return resolved


USE_RERANK_EMB = True
from .rerankers import VacancyEmbeddingReranker

def _rerank_vacancies(vacancies: List[Dict[str, Any]], features: Dict[str, Any], resume_text: str) -> List[Dict[str, Any]]: 
    if USE_RERANK_EMB:
        reranker = VacancyEmbeddingReranker()
        return reranker.rerank(
            vacancies,
            resume_text=resume_text,
            features=features,
            use_cross_encoder=True,
            top_k = 5
        )
    else:
        resume_tokens = set(_tokenize(resume_text)) 
        position_terms = [set(_tokenize(pos)) for pos in features.get("positions", []) if isinstance(pos, str) and pos] 
        def score(job: Dict[str, Any]) -> float: 
            base_score = job.get("match_score") or 0.0 
            try: 
                score_value = float(base_score) 
            except (TypeError, ValueError): 
                score_value = 0.0 
            title_tokens = set(_tokenize(job.get("title") or "")) 
            for tokens in position_terms: 
                if not tokens: 
                    continue 
                if tokens.issubset(title_tokens): 
                    score_value += 3.0 
                else: 
                    overlap = len(tokens & title_tokens) 
                    if overlap: 
                        score_value += 1.5 * (overlap / len(tokens)) 
            skills = [s.lower() for s in job.get("skills", []) if isinstance(s, str)] 
            if skills and resume_tokens: 
                skill_hits = sum(s in resume_tokens
                            for s in skills) 
                score_value += skill_hits * 1.2 
            description_tokens = set(_tokenize(job.get("description") or "")) 
            if description_tokens and resume_tokens: 
                overlap_desc = len(description_tokens & resume_tokens) 
                score_value += min(overlap_desc, 30) * 0.08 
            experience_tokens = set(_tokenize(job.get("experience") or "")) 
            if experience_tokens and resume_tokens: 
                exp_overlap = len(experience_tokens & resume_tokens) 
                score_value += exp_overlap * 0.2 
            return round(score_value, 4) 
        for job in vacancies: 
            job["rank_score"] = score(job) 
        vacancies.sort(key=lambda job: job.get("rank_score", 0.0), reverse=True) 
        return vacancies


def hh_search_vacancies(
    features: Dict[str, Any],
    resume_text: str,
    per_page: int = DEFAULT_QUERY_PER_PAGE,
    max_results: int = MAX_TOTAL_RESULTS,
) -> List[Dict[str, Any]]:
    positions = [p.strip() for p in features.get("positions", []) if isinstance(p, str) and p.strip()]
    locations = [l.strip() for l in features.get("locations", []) if isinstance(l, str) and l.strip()]
    salary_range = features.get("salary_range") or {}
    salary_min = salary_range.get("min")
    salary_min = int(salary_min) if isinstance(salary_min, (float, int)) else None
    per_page = max(1, min(per_page, 100))
    area_lookup = _resolve_area_ids(locations[:MAX_LOCATION_QUERIES])
    seen_ids: set[str] = set()
    vacancies: List[Dict[str, Any]] = []

    query_positions = positions[:MAX_POSITION_QUERIES]
    if not query_positions:
        fallback_terms = list(dict.fromkeys(_tokenize(resume_text)))[:3]
        if fallback_terms:
            query_positions = [" ".join(fallback_terms)]
        else:
            query_positions = ["Специалист"]

    query_terms = []
    for pos in query_positions:
        if cleaned := pos.replace('"', '').strip():
            query_terms.append(f'({cleaned})')
    query_text = HH_QUERY_TEMPLATE.format(terms=' OR '.join(query_terms)) if query_terms else "Специалист"

    query_locations: List[Optional[str]] = locations[:MAX_LOCATION_QUERIES] or [None]

    for location in query_locations:
        area_id = area_lookup.get(location) if location else None
        try:
            raw_vacancies = search_vacancies(
                text=query_text,
                area_id=area_id,
                salary_from=salary_min,
                per_page=per_page,
            )
        except Exception as exc:  # pragma: no cover - network or API errors
            logging.warning(
                "job_lookup: hh.ru search failed for '%s'/'%s': %s", query_text, location or "-", exc
            )
            continue

        for vacancy in raw_vacancies:
            vacancy_id = vacancy.get("id")
            if vacancy_id is not None:
                vacancy_id = str(vacancy_id)
            if vacancy_id:
                if vacancy_id in seen_ids:
                    continue
                seen_ids.add(vacancy_id)
            normalised = _normalise_vacancy(vacancy)
            normalised["search_position"] = query_text
            normalised["search_location"] = location
            vacancies.append(normalised)

    if not vacancies:
        return []

    reranked = _rerank_vacancies(vacancies, features, resume_text)
    return reranked[:max_results] if max_results else reranked



def _read_resume_text(state: JobAgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if message.type != "human":
            continue
        text_parts = []
        for part in getattr(message, "content", []) or []:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        if candidate_text := "\n".join(
            p.strip() for p in text_parts if p.strip()
        ):
            return candidate_text
    return ""



def reset_or_run(state: JobAgentState, config: RunnableConfig) -> str:
    if state["messages"][-1].content[0].get("type") == "reset":
        return "reset_memory"
    else:
        return "capture_resume"

def reset_memory(state: JobAgentState) -> JobAgentState:
    """
    Delete every message currently stored in the thread’s state.
    """
    all_msg_ids = [m.id for m in state["messages"]]
    # Returning RemoveMessage instances instructs the reducer to delete them
    return {
        "messages": [RemoveMessage(id=mid) for mid in all_msg_ids]
    }

def user_info(state: JobAgentState, config: RunnableConfig):
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None)
    user_role = configuration.get("user_role", "default")
    return {"user_info": {"user_id": user_id, "user_role": user_role}}


def capture_resume(state: JobAgentState, config: Optional[RunnableConfig] = None) -> JobAgentState:
    resume_text = _read_resume_text(state)
    return {"resume_text": resume_text}


def build_feature_node(extractor_llm):
    def extract_features(state: JobAgentState, config: Optional[RunnableConfig] = None) -> JobAgentState:
        resume_text = state.get("resume_text", "")
        if not resume_text:
            return {"extracted_features": {"positions": [], "locations": [], "salary_range": {}}}
        features = extract_features_from_resume(resume_text, extractor_llm)
        return {"extracted_features": features}

    return extract_features


def build_job_lookup_node(fetch_jobs_fn):
    def job_lookup(state: JobAgentState, config: Optional[RunnableConfig] = None) -> JobAgentState:
        logging.info("Looking for jobs...")
        features = state.get("extracted_features") or {}
        logging.info(f"Looking for jobs... {features}")
        resume_text = state.get("resume_text", "")
        if not features:
            return {"job_candidates": []}
        try:
            job_list = fetch_jobs_fn(features, resume_text)
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("job_lookup: hh search failed: %s", exc)
            job_list = []
        logging.info(f"...jobs retrieved {len(job_list)}")
        return {"job_candidates": job_list}

    return job_lookup



def score_job(job: Dict[str, Any], resume_tokens: List[str], features: Dict[str, Any]) -> float:
    heuristic_score = job.get("rank_score", job.get("match_score", 0.0))
    try:
        score = float(heuristic_score)
    except (TypeError, ValueError):
        score = 0.0

    resume_token_set = set(resume_tokens)

    title_tokens = set(_tokenize(job.get("title") or ""))
    candidate_positions = [
        set(_tokenize(p)) for p in features.get("positions", []) if isinstance(p, str)
    ]
    for tokens in candidate_positions:
        if not tokens:
            continue
        if tokens.issubset(title_tokens):
            score += 2.5
        else:
            overlap = len(tokens & title_tokens)
            if overlap:
                score += 1.0 * (overlap / len(tokens))

    candidate_locations = [l.lower() for l in features.get("locations", []) if isinstance(l, str)]
    location_value = (job.get("location") or "").lower()
    if candidate_locations and location_value in candidate_locations:
        score += 1.2

    salary = features.get("salary_range", {}) or {}
    desired_min = salary.get("min")
    desired_max = salary.get("max")
    job_salary = job.get("salary", {}) or {}
    if desired_min is not None and job_salary.get("max") and job_salary["max"] >= desired_min:
        score += 0.8
    if desired_max is not None and job_salary.get("min") and job_salary["min"] <= desired_max:
        score += 0.4

    job_skills = [skill.lower() for skill in job.get("skills", []) if isinstance(skill, str)]
    overlap = sum(skill in resume_token_set for skill in job_skills)
    score += overlap * 0.6

    resume_text_value = " ".join(resume_tokens).strip()
    if resume_text_value:
        # Push more weight to description/experience in the semantic scorer.
        semantic_score = semantic_score_job(
            job=job,
            resume_text=resume_text_value,
            features=features,
            weights=SemanticScoreWeights(
                title_position=0.2,
                skills_resume=0.25,
                description_resume=0.55,
                experience_resume=0.0,
            ),
        )
        score = 0.4 * score + 0.6 * float(semantic_score)

    return round(score, 3)



def rank_jobs(state: JobAgentState, config: Optional[RunnableConfig] = None) -> JobAgentState:
    logging.info("Ranking jobs...")

    jobs = state.get("job_candidates") or []
    resume_text = state.get("resume_text", "")
    features = state.get("extracted_features") or {}
    resume_tokens = list(_tokenize(resume_text))

    ranked: List[Dict[str, Any]] = []
    for job in jobs:
        job_score = score_job(job, resume_tokens, features)
        ranked.append({**job, "rank_score": job_score})

    ranked.sort(key=lambda item: item["rank_score"], reverse=True)
    logging.info("...jobs ranked")
    return {"ranked_jobs": ranked[:5]}



def format_jobs_markdown(payload: Dict[str, Any]) -> str:
    summary_text = payload.get("summary") or ""
    vacancies = payload.get("vacancies") or []

    lines: List[str] = []

    summary_items = [
        line.strip()
        for line in summary_text.splitlines()
        if isinstance(line, str) and line.strip()
    ]
    if summary_items:
        lines.append("## Краткое описание")
        for item in summary_items:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Вакансии")
    if not vacancies:
        lines.append("- Подходящих вакансий не найдено.")
        return "\n".join(lines)

    def format_salary(salary_info: Any) -> str:
        if not isinstance(salary_info, dict):
            return ""
        min_amount = salary_info.get("min")
        max_amount = salary_info.get("max")
        currency = salary_info.get("currency")

        parts: List[str] = []
        if min_amount is not None:
            parts.append(f"от {min_amount}")
        if max_amount is not None:
            parts.append(f"до {max_amount}")
        if currency:
            parts.append(str(currency))
        return " ".join(parts)

    highlight_pattern = re.compile(r"</?highlighttext>", re.IGNORECASE)

    def apply_highlight_formatting(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        formatted = highlight_pattern.sub("_", text)
        return formatted.strip()

    for index, vacancy in enumerate(vacancies, start=1):
        title = vacancy.get("title") or "Без названия"
        lines.append(f"{index}. **{title}**")

        def add_field(label: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, list):
                rendered = ", ".join(str(item).strip() for item in value if str(item).strip())
                if not rendered:
                    return
            elif isinstance(value, float):
                rendered = f"{value:.3f}"
            else:
                rendered = str(value).strip()
                if not rendered:
                    return
            lines.append(f"   - **{label}:** {rendered}")

        #add_field("Идентификатор", vacancy.get("id"))
        add_field("Компания", vacancy.get("company"))
        add_field("Локация", vacancy.get("location"))

        salary_text = format_salary(vacancy.get("salary"))
        if salary_text:
            add_field("Зарплата", salary_text)

        add_field("Навыки", vacancy.get("skills"))

        #rank_score = vacancy.get("rank_score")
        #if isinstance(rank_score, (int, float)):
        #    add_field("Оценка ранжирования", float(rank_score))
        #elif rank_score:
        #    add_field("Оценка ранжирования", rank_score)

        #match_score = vacancy.get("match_score")
        #if isinstance(match_score, (int, float)):
        #    add_field("Оценка совпадения", float(match_score))
        #elif match_score:
        #    add_field("Оценка совпадения", match_score)

        add_field("Требуемый опыт", vacancy.get("experience"))
        add_field("Источник", vacancy.get("source"))
        add_field("Дата публикации", vacancy.get("published_at"))
        #add_field("Позиция запроса", vacancy.get("search_position"))
        #add_field("Локация запроса", vacancy.get("search_location"))

        description = vacancy.get("description")
        if isinstance(description, str):
            formatted_description = apply_highlight_formatting(description)
            if formatted_description:
                lines.append(f"   - **Описание:** {formatted_description}")
        elif description:
            add_field("Описание", description)

        url = vacancy.get("url")
        if isinstance(url, str) and url.strip():
            link = url.strip()
            lines.append(f"   - **Ссылка:** [Открыть вакансию]({link})")

        lines.append("")

    return "\n".join(lines).rstrip()


def respond_with_jobs(state: JobAgentState, config: Optional[RunnableConfig] = None) -> JobAgentState:
    jobs = state.get("ranked_jobs") or []
    features = state.get("extracted_features") or {}

    summary_lines: List[str] = []
    positions = [p for p in features.get("positions", []) if isinstance(p, str) and p]
    locations = [l for l in features.get("locations", []) if isinstance(l, str) and l]
    salary_range = features.get("salary_range") or {}

    if positions:
        summary_lines.append("Желаемые позиции: " + ", ".join(positions))
    if locations:
        summary_lines.append("Регионы поиска: " + ", ".join(locations))
    if salary_range.get("min") or salary_range.get("max"):
        min_amount = salary_range.get("min")
        max_amount = salary_range.get("max")
        currency = salary_range.get("currency") or "RUB"
        parts = []
        if min_amount is not None:
            parts.append(f"от {min_amount}")
        if max_amount is not None:
            parts.append(f"до {max_amount}")
        if currency:
            parts.append(currency)
        summary_lines.append("Желаемая зарплата: " + " ".join(parts))

    if jobs:
        summary_lines.append(f"Найдено {len(jobs)} подходящих вакансий.")
    else:
        summary_lines.append("Подходящих вакансий не найдено.")

    payload = {
        "summary": "\n".join(summary_lines),
        "vacancies": [
            {
                "id": job.get("id"),
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "url": job.get("url"),
                "salary": job.get("salary"),
                "skills": job.get("skills"),
                "rank_score": job.get("rank_score"),
                "match_score": job.get("match_score"),
                "experience": job.get("experience"),
                "source": job.get("source"),
                "published_at": job.get("published_at"),
                "search_position": job.get("search_position"),
                "search_location": job.get("search_location"),
                "description": job.get("description"),
            }
            for job in jobs
            if isinstance(job, dict)
        ],
    }

    formatted_jobs = format_jobs_markdown(payload)
    if cfg.JOB_FIND_TRANSLATE:
        try:
            message_content = translate_deepl(text=formatted_jobs, target_lang="EN", use_free_api=True, preserve_formatting=True)
        except Exception as exc:
            logging.warning("Translation failed: %s", exc)
            message_content = formatted_jobs
    else:
        message_content = formatted_jobs

    return {"messages": [AIMessage(content=message_content)]}


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    role: str = "default",
    use_platform_store: bool = False,
):
    log_name = f"job_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    callback_handlers = [json_handler]

    if cfg.LANGFUSE_URL:
        try:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler as LangfuseHandler

            langfuse = Langfuse(
                public_key=cfg.LANGFUSE_PUBLIC,
                secret_key=cfg.LANGFUSE_SECRET,
                host=cfg.LANGFUSE_URL,
            )
            callback_handlers.append(LangfuseHandler())
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.warning("Langfuse initialisation failed: %s", exc)

    provider_name = provider.value if isinstance(provider, ModelType) else str(provider)
    extractor_llm = get_llm(model="nano", provider=provider_name, temperature=0)
    memory = None if use_platform_store else MemorySaver()

    builder = StateGraph(JobAgentState, config_schema=ConfigSchema)
    builder.add_node("fetch_user_info", user_info)
    builder.add_node("reset_memory", reset_memory)
    builder.add_node("capture_resume", capture_resume)
    builder.add_node("extract_features", build_feature_node(extractor_llm))
    builder.add_node("job_lookup", build_job_lookup_node(hh_search_vacancies))
    builder.add_node("rank_jobs", rank_jobs)
    builder.add_node("respond", respond_with_jobs)

    #builder.add_edge(START, "capture_resume")
    builder.add_edge(START, "fetch_user_info")
    builder.add_conditional_edges(
        "fetch_user_info",
        reset_or_run,
        {
            "reset_memory": "reset_memory",
            "capture_resume": "capture_resume",
        }
    )
    builder.add_edge("capture_resume", "extract_features")
    builder.add_edge("extract_features", "job_lookup")
    builder.add_edge("job_lookup", "rank_jobs")
    builder.add_edge("rank_jobs", "respond")
    builder.add_edge("respond", END)

    graph = builder.compile(name="find_job_agent", checkpointer=memory)

    agent_key = "find_job"
    def _handle_kb_reload(context: KBReloadContext) -> None:
        logging.info(
            "KB reload requested for %s (reason=%s); no KB-backed retrievers configured yet.",
            agent_key,
            context.reason,
        )
    register_reload_listener(agent_key, _handle_kb_reload)

    return graph.with_config({"callbacks": callback_handlers})





