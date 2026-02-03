from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any, Dict, List, NotRequired, Optional, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.agents.structured_output import (
    AutoStrategy,
    ProviderStrategy,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

import config

from utils.utils import is_valid_json_string
from agents.tools.think import ThinkTool
from agents.tools.yandex_search import YandexSearchTool as SearchTool

from agents.utils import ModelType, get_llm, _extract_text
from agents.structured_prompt_utils import build_json_prompt, provider_then_tool
from agents.prettifier import prettify
from platform_utils.llm_logger import JSONFileTracer

from .models import ArticleRecord, IdeatorReport, set_locale as set_models_locale
from .prompts import (
    #FACT_REF_HINT,
    #IDEAS_INSTRUCTION,
    #IDEATOR_SYSTEM_PROMPT,
    #SENSE_LINE_INSTRUCTION,
    #TOOL_POLICY_PROMPT,
    get_locale,
)
from .report_loader import load_report, process_report
from .sense_lines_pipeline import build_sense_lines_from_report
from .state import IdeatorAgentContext, IdeatorAgentState

LOG = logging.getLogger(__name__)

MIN_REPORT_LENGTH = 2000
PRECOMPUTED_SENSE_LINES_NOTE = (
    "\nNOTE: The sense lines above are precomputed by clustering. "
    "Return them unchanged unless the user explicitly asks to regenerate or edit them. "
    "If the user requests regeneration, set decision.regen_lines = true and keep the current sense_lines.\n"
)


def _safe_stream_writer():
    """Return a writer suitable for `stream_mode="custom"`; otherwise no-op."""
    try:
        return get_stream_writer()
    except Exception:
        return lambda *_args, **_kwargs: None


class StreamWriterCallbackHandler(BaseCallbackHandler):
    """Forward LangChain callbacks (tool/chain lifecycle) into LangGraph custom stream."""

    def on_tool_start(self, serialized, input_str=None, **kwargs):
        writer = _safe_stream_writer()
        name = (serialized or {}).get("name") or (serialized or {}).get("id") or "tool"
        writer({"type": "tool_start", "name": name})

    def on_tool_end(self, output=None, **kwargs):
        writer = _safe_stream_writer()
        writer({"type": "tool_end"})

    def on_tool_error(self, error, **kwargs):
        writer = _safe_stream_writer()
        writer({"type": "tool_error", "error": str(error)})

    def on_chain_start(self, serialized, inputs, **kwargs):
        writer = _safe_stream_writer()
        name = (serialized or {}).get("name") or "chain"
        writer({"type": "chain_start", "name": name})

    def on_chain_end(self, outputs, **kwargs):
        writer = _safe_stream_writer()
        writer({"type": "chain_end"})

class ArticleRef(TypedDict):
    id: Annotated[int, "Id from the provided list"]
    title: Annotated[str, "Title of the article"]
    summary: Annotated[str, "Summary of the article"]

class Decision(TypedDict, total=False):
    selected_line_index: Annotated[Optional[int], "1-based line index if chosen"]
    custom_line_text: Annotated[Optional[str], "User provided custom line"]
    consent_generate: Annotated[Optional[bool], "User ready to generate ideas"]
    regen_lines: Annotated[Optional[bool], "User asked to regenerate sense lines"]
    more_ideas: Annotated[Optional[bool], "User wants more ideas for current line or user proposed to combine a few provided sence lines or user proposed to change in some line."]
    selected_idea_index: Annotated[Optional[int], "1-based idea index if chosen"]
    custom_idea_text: Annotated[Optional[str], "User provided custom idea"]
    finish: Annotated[Optional[bool], "User wants to finish"]


class SenseLineItem(TypedDict):
    id: Annotated[str, "Stable id like L1/L2"]
    short_title: Annotated[str, "Short name of the sense line"]
    description: Annotated[str, "1-2 sentences grounded in the provided articles"]
    #article_ids: Annotated[List[int], "Ids from the provided list only"]
    articles: Annotated[List[ArticleRef], "List of articles from the provided list only"]
    region_note: Annotated[str, "Region applicability note"]


class SenseLineResponse(TypedDict):
    assistant_message: Annotated[str, "Natural language reply for the user, including a brief summary of generated sense_lines (if any) and summary of decision."]
    sense_lines: List[SenseLineItem]
    decision: NotRequired[Decision]


class IdeaItem(TypedDict):
    title: Annotated[str, "Idea headline (1 short sentence)"]
    summary: Annotated[str, "1-2 sentences grounded strictly in provided articles"]
    #article_ids: Annotated[List[int], "Ids from the provided list only"]
    articles: Annotated[List[ArticleRef], "List of articles from the provided list only"]
    region_note: Annotated[str, "Region applicability note"]
    importance_hint: Annotated[NotRequired[str], "High/Medium/Low or empty"]


class IdeaListResponse(TypedDict):
    assistant_message: Annotated[str, "Natural language reply for the user, including a brief summary of generated ideas (if any) and summary of decision."]
    ideas: Annotated[List[IdeaItem], "List of generated ideas."]
    decision: NotRequired[Decision]


_think_tool = ThinkTool()
_yandex_tool = SearchTool(
    api_key=config.YA_API_KEY,
    folder_id=config.YA_FOLDER_ID,
    max_results=3,
    summarize=True
)


_LOCALE: Dict[str, Any] = {}
_AGENT_TEXT: Dict[str, str] = {}
_PROMPT_FRAGMENTS: Dict[str, str] = {}
_REGION_TEXT: Dict[str, str] = {}
_PROMPTS: Dict[str, str] = {}
_CURRENT_LOCALE = "ru"


def set_locale(locale: str = "ru") -> None:
    global _LOCALE, _AGENT_TEXT, _PROMPT_FRAGMENTS, _REGION_TEXT, _PROMPTS, _CURRENT_LOCALE
    _LOCALE = get_locale(locale)
    _AGENT_TEXT = _LOCALE["agent"]
    _PROMPT_FRAGMENTS = _LOCALE["prompt_fragments"]
    _REGION_TEXT = _LOCALE["regions"]
    _PROMPTS = _LOCALE["prompts"]
    _CURRENT_LOCALE = locale


set_locale()


def _format_articles(articles: List[ArticleRecord], limit_chars: int = 400) -> str:
    lines: List[str] = []
    for art in articles:
        summary = art.summary
        if len(summary) > limit_chars:
            summary = f"{summary[:limit_chars]}..."
        lines.append(
            f"- [{art.id}] ({art.norm_importance()}; {art.region_label()})\n"
            f"  {summary}\n"
            f"  [{art.display_title()}]({art.url})"
        )
    return "\n\n".join(lines)


def _format_sense_lines(lines: List[Dict[str, Any]]) -> str:
    formatted: List[str] = []
    for idx, line in enumerate(lines or [], start=1):
        articles = line.get("articles") or []
        articles_label = ", ".join(
            f"[{art.get('id', i)}] {art.get('title','')}" if isinstance(art, dict) else str(art)
            for i, art in enumerate(articles, 1)
        )
        formatted.append(
            f"{idx}) {line.get('id') or f'L{idx}'} | {line.get('short_title','')}\n"
            f"{line.get('description','')}\n"
            f"region: {line.get('region_note','')}\n"
            f"articles: {articles_label}"
        )
    return "\n\n".join(formatted)


def _format_ideas(ideas: List[Dict[str, Any]]) -> str:
    formatted: List[str] = []
    for idx, idea in enumerate(ideas or [], start=1):
        articles = idea.get("articles") or []
        articles_label = ", ".join(
            f"[{art.get('id', i)}] {art.get('title','')}" if isinstance(art, dict) else str(art)
            for i, art in enumerate(articles, 1)
        )
        formatted.append(
            f"{idx}) {idea.get('title','')}\n"
            f"{idea.get('summary','')}\n"
            f"region: {idea.get('region_note','')}\n"
            f"articles: {articles_label}"
        )
    return "\n\n".join(formatted)


def _fact_refs_for(report: IdeatorReport, article_ids: List[int]) -> List[str]:
    return [art.fact_ref() for art in report.filter_by_ids(article_ids)]


def _links_md_for(report: IdeatorReport, article_ids: List[int], limit: int = 5) -> str:
    links: List[str] = []
    links.extend(
        f"- [{art.display_title()}]({art.url})"
        for art in report.filter_by_ids(article_ids)[:limit]
        if art.url
    )
    return "\n".join(links)


def _fact_links_md(report: IdeatorReport, article_ids: List[int], limit: int = 5) -> str:
    links: List[str] = []
    for art in report.filter_by_ids(article_ids)[:limit]:
        if art.url:
            country = (art.search_country or "").lower()
            if country in {"ru", "rus", "russia", "рф", "россия"}:
                relevance = _REGION_TEXT["ru_relevant"]
            elif country:
                relevance = _REGION_TEXT["relevance_country"].format(country=country.upper())
            else:
                relevance = _REGION_TEXT["relevance_unknown"]
            importance = art.norm_importance()
            links.append(
                _AGENT_TEXT["fact_link_item"].format(
                    title=art.display_title(),
                    url=art.url,
                    relevance=relevance,
                    importance=importance,
                )
            )
    return "\n".join(links)


def _fact_links_from_articles(articles: List[ArticleRecord], limit: int = 5) -> str:
    links: List[str] = []
    for art in articles[:limit]:
        if art.url:
            country = (art.search_country or "").lower()
            if country in {"ru", "rus", "russia", "рф", "россия"}:
                relevance = _REGION_TEXT["ru_relevant"]
            elif country:
                relevance = _REGION_TEXT["relevance_country"].format(country=country.upper())
            else:
                relevance = _REGION_TEXT["relevance_unknown"]
            importance = art.norm_importance()
            links.append(
                _AGENT_TEXT["fact_link_item"].format(
                    title=art.display_title(),
                    url=art.url,
                    relevance=relevance,
                    importance=importance,
                )
            )
    return "\n".join(links)


def _extract_articles(obj: Dict[str, Any], report: IdeatorReport) -> List[ArticleRecord]:
    raw_articles = obj.get("articles")
    if isinstance(raw_articles, list) and raw_articles:
        if all(isinstance(a, ArticleRecord) for a in raw_articles):
            return list(raw_articles)  # type: ignore
        articles: List[ArticleRecord] = []
        for item in raw_articles:
            if isinstance(item, dict) and "id" in item:
                try:
                    articles.extend(report.filter_by_ids([int(item["id"])]))
                except Exception:
                    continue
        if articles:
            return articles
    ids = obj.get("article_ids") or []
    if isinstance(ids, list):
        return report.filter_by_ids([int(i) for i in ids if isinstance(i, (int, str))])
    return []


def _links_md_for(report: IdeatorReport, article_ids: List[int], limit: int = 5) -> str:
    links: List[str] = []
    links.extend(
        f"- [{art.display_title()}]({art.url})"
        for art in report.filter_by_ids(article_ids)[:limit]
        if art.url
    )
    return "\n".join(links)


def _build_sense_agent(model: BaseChatModel):
    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        state: IdeatorAgentState = request.state
        report: IdeatorReport | str = state.get("report")  # type: ignore
        use_report = isinstance(report, IdeatorReport)
        articles = report.sorted_articles()[:80] if use_report else []
        existing_lines = state.get("sense_lines") or []
        prompt = (
            _PROMPTS["ideator_core_prompt"]
            + "\n"
            + _PROMPTS["sense_line_stage_prompt"]
            + "\n"
            + _PROMPTS["sense_line_output_contract"]
            + "\n"
            + _PROMPTS["fact_ref_hint"]
            + "\n\n"
            + _PROMPT_FRAGMENTS["search_goal_line"].format(
                search_goal=report.search_goal if use_report else ""
            )
            + _PROMPT_FRAGMENTS["articles_stats_line"].format(
                total=report.total_articles if use_report else 0,
                count=len(articles),
            )
            + _PROMPT_FRAGMENTS["articles_list_block"].format(
                articles=_format_articles(articles)
            )
        )
        if existing_lines:
            prompt += _PROMPT_FRAGMENTS["existing_sense_lines_block"].format(
                lines=_format_sense_lines(existing_lines)
            )
            prompt += PRECOMPUTED_SENSE_LINES_NOTE
        prompt += f"{build_json_prompt(SenseLineResponse)}"
        return prompt + "\n\n" + _PROMPTS["format_instruction"] + "\n\n" + _PROMPTS["think_tool_policy_prompt"]

    return create_agent(
        model=model,
        tools=[_think_tool],
        middleware=[build_prompt, provider_then_tool],
        response_format=SenseLineResponse,
        state_schema=IdeatorAgentState,
        context_schema=IdeatorAgentContext,
    )


def _build_ideas_agent(model: BaseChatModel):
    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        state: IdeatorAgentState = request.state
        report: IdeatorReport | str = state.get("report")  # type: ignore
        use_report = isinstance(report, IdeatorReport)
        articles = state.get("filtered_articles") or []
        if not articles and use_report:
            filtered_ids = state.get("filtered_article_ids") or []
            articles = report.filter_by_ids(filtered_ids)
        existing_ideas = state.get("ideas") or []
        prompt = (
            _PROMPTS["ideator_core_prompt"]
            + "\n"
            + _PROMPTS["ideas_stage_prompt"]
            + "\n"
            + _PROMPTS["ideas_output_contract"]
            + "\n"
            + _PROMPTS["fact_ref_hint"]
            + "\n\n"
            + _PROMPT_FRAGMENTS["active_sense_line_line"].format(
                line=state.get("selected_line_id", "")
            )
            + _PROMPT_FRAGMENTS["articles_in_context_line"].format(count=len(articles))
            + _PROMPT_FRAGMENTS["available_articles_block"].format(
                articles=_format_articles(articles)
            )
        )
        if existing_ideas:
            prompt += _PROMPT_FRAGMENTS["existing_ideas_block"].format(
                ideas=_format_ideas(existing_ideas)
            )
        prompt += f"{build_json_prompt(IdeaListResponse)}"
        return prompt + _PROMPTS["format_instruction"] + _PROMPTS["think_tool_policy_prompt"] + "\n\n" + _PROMPTS["search_tool_policy_prompt"]

    return create_agent(
        model=model,
        tools=[_think_tool, _yandex_tool],
        middleware=[build_prompt, provider_then_tool],
        response_format=IdeaListResponse,
        state_schema=IdeatorAgentState,
        context_schema=IdeatorAgentContext,
    )


def create_init_node():
    def _find_report_path_from_attachments(state: IdeatorAgentState) -> Optional[str]:
        sources = []
        attachments = state.get("attachments")
        if isinstance(attachments, list):
            sources.extend(attachments)
        for item in sources:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            filename = (item.get("filename") or "").lower()
            ctype = (item.get("content_type") or "").lower()
            if not path:
                continue
            if filename.endswith(".json") or "json" in ctype:
                return path
        return None

    def init_node(
        state: IdeatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IdeatorAgentContext],
    ) -> IdeatorAgentState:
        report_path = None
        if runtime.context and runtime.context.get("report_path"):
            report_path = runtime.context["report_path"]
        if report_path is None:
            report_path = _find_report_path_from_attachments(state)
        if report_path:
            state["report_path"] = report_path
            state["report"] = load_report(report_path)
        if not report_path and not state.get("report"):
            messages = state.get("messages") or []
            last_message = messages[-1] if messages else None
            if last_message and getattr(last_message, "type", None) == "human":
                content = _extract_text(last_message)
                #content = content.replace("\xa0", "")
                #getattr(last_message, "content", "")
                if is_valid_json_string(content):
                    data = json.loads(content)
                    state["report"] = process_report(data)
                else:
                    if content and len(content) >= MIN_REPORT_LENGTH:
                        state["report"] = content

        if "ideas_cache" not in state:
            state["ideas_cache"] = {}
        if "force_regen_lines" not in state:
            state["force_regen_lines"] = False
        if "force_regen" not in state:
            state["force_regen"] = False
        if "phase" not in state:
            state["phase"] = "lines"
        if not state.get("greeted"):
            greet = (
                _AGENT_TEXT["greeting_with_report"]
                if state.get("report")
                else _AGENT_TEXT["greeting_no_report"]
            )
            state["messages"] = (state.get("messages") or []) + [AIMessage(content=greet)]
            state["greeted"] = True
        return state

    return init_node


def create_sense_lines_prepare_node():
    def sense_lines_prepare_node(
        state: IdeatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IdeatorAgentContext],
    ) -> IdeatorAgentState:
        report: IdeatorReport | str = state.get("report")  # type: ignore
        use_report = isinstance(report, IdeatorReport)
        if use_report and (state.get("force_regen_lines") or not state.get("sense_lines")):
            precomputed_lines = build_sense_lines_from_report(report, locale=_CURRENT_LOCALE)
            if precomputed_lines:
                return {
                    "sense_lines": precomputed_lines,
                    "sense_lines_precomputed": True,
                    "skip_sense_llm": True,
                }
        return {"sense_lines_precomputed": False, "skip_sense_llm": False}

    return sense_lines_prepare_node


def route_sense_lines(state: IdeatorAgentState) -> str:
    return "sense_lines_post" if state.get("skip_sense_llm") else "sense_lines_llm"


def create_sense_lines_llm_node(model: BaseChatModel):
    return _build_sense_agent(model)


def create_sense_lines_post_node():
    def sense_lines_post_node(
        state: IdeatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IdeatorAgentContext],
    ) -> IdeatorAgentState:
        LOG.info("sense_lines_post_node: start, messages=%d", len(state.get("messages", [])))

        report: IdeatorReport | str = state.get("report")  # type: ignore
        use_report = isinstance(report, IdeatorReport)
        structured = state.get("structured_response") or {}
        sense_lines: List[Dict[str, Any]] = structured.get("sense_lines") or state.get("sense_lines") or []
        assistant_message: str = structured.get("assistant_message") or ""
        decision: Decision = structured.get("decision") or {}
        precomputed = bool(state.get("sense_lines_precomputed"))

        result: IdeatorAgentState = IdeatorAgentState()
        result["sense_lines"] = sense_lines
        result["force_regen_lines"] = False
        result["phase"] = state.get("phase") or "lines"
        result["skip_sense_llm"] = False
        result["sense_lines_precomputed"] = False
        result["structured_response"] = None
        LOG.info("sense_lines_post_node: generated %d lines", len(sense_lines))

        content = ""
        if assistant_message:
            content = assistant_message
        if sense_lines and (not assistant_message or precomputed):
            formatted_lines = []
            for idx, line in enumerate(sense_lines, start=1):
                articles = _extract_articles(line, report) if use_report else []
                fact_links_md = _fact_links_from_articles(articles) if use_report else ""
                extras: List[str] = []
                if fact_links_md:
                    extras.append(f"{_AGENT_TEXT['fact_links_label']}\n{fact_links_md}")
                formatted_lines.append(
                    f"{idx}) {line.get('short_title','')}\n"
                    f"{line.get('description','')}\n"
                    f"{line.get('region_note','')}\n"
                    + "\n".join(extras)
                )
            fallback = f"{_AGENT_TEXT['sense_lines_label']}\n" + "\n\n".join(formatted_lines)
            if content:
                content += "\n" + fallback
            else:
                content = fallback

        result["messages"] = state.get("messages", []) + [AIMessage(content=content)]

        if decision.get("regen_lines"):
            result["force_regen_lines"] = True
            result.pop("selected_line_id", None)
            result.pop("filtered_article_ids", None)
            result.pop("filtered_articles", None)
            result["phase"] = "lines"

        selected_idx = decision.get("selected_line_index")
        custom_line_text = decision.get("custom_line_text")
        ready_for_ideas = decision.get("consent_generate")
        if ready_for_ideas and selected_idx and 1 <= selected_idx <= len(sense_lines):
            selected = sense_lines[selected_idx - 1]
            result["selected_line_id"] = selected.get("id") or f"L{selected_idx}"
            articles = _extract_articles(selected, report) if use_report else []
            result["filtered_articles"] = articles
            result["filtered_article_ids"] = []
            result["force_regen"] = True
            result["phase"] = "ideas"
        elif ready_for_ideas and custom_line_text:
            top_articles = report.sorted_articles()[:20] if use_report else []
            result["selected_line_id"] = "custom_line"
            result["filtered_articles"] = top_articles
            result["filtered_article_ids"] = []
            result["force_regen"] = True
            result["phase"] = "ideas"
        elif decision.get("finish"):
            result["phase"] = "finish"
        return result

    return sense_lines_post_node


def create_ideas_llm_node(model: BaseChatModel):
    return _build_ideas_agent(model)


def create_ideas_post_node():
    def ideas_post_node(
        state: IdeatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IdeatorAgentContext],
    ) -> IdeatorAgentState:
        LOG.info(
            "ideas_post_node: start, selected_line=%s, filtered_ids=%s",
            state.get("selected_line_id"),
            state.get("filtered_article_ids"),
        )
        structured = state.get("structured_response") or {}
        ideas = structured.get("ideas") or state.get("ideas") or []
        assistant_message: str = structured.get("assistant_message") or ""
        decision: Decision = structured.get("decision") or {}

        result: IdeatorAgentState = IdeatorAgentState()
        result["ideas"] = ideas
        result["force_regen"] = False
        result["phase"] = "ideas"
        result["structured_response"] = None
        LOG.info("ideas_post_node: generated %d ideas", len(ideas))

        content = ""
        if assistant_message:
            content = assistant_message
        elif ideas:
            report: IdeatorReport | str = state.get("report")  # type: ignore
            use_report = isinstance(report, IdeatorReport)
            formatted = []
            for idea in ideas:
                articles = _extract_articles(idea, report) if use_report else []
                fact_links = _fact_links_from_articles(articles) if use_report else ""
                extras = f"{idea.get('region_note','')}\n"
                if fact_links:
                    extras += f"{_AGENT_TEXT['fact_links_label']}\n{fact_links}"
                formatted.append(
                    f"- {idea.get('title','')}: {idea.get('summary','')}\n"
                    f"{extras}"
                )
            if formatted:
                fallback = f"{_AGENT_TEXT['ideas_label']}\n" + "\n\n".join(formatted)
            else:
                fallback = _AGENT_TEXT["ideas_generation_failed"]
            content += "\n" + fallback

        result["messages"] = state.get("messages", []) + [AIMessage(content=content)]

        if decision.get("more_ideas"):
            result["force_regen"] = True
        selected_idea_index = decision.get("selected_idea_index")
        if selected_idea_index:
            result["active_idea_id"] = selected_idea_index
        if decision.get("custom_idea_text"):
            result["custom_idea_text"] = decision.get("custom_idea_text")
        if decision.get("finish"):
            result["phase"] = "finish"
        return result

    return ideas_post_node




def route(state: IdeatorAgentState) -> str:
    if not state.get("greeted"):
        return "init"
    phase = state.get("phase") or ("ideas" if state.get("selected_line_id") else "lines")
    state["phase"] = phase

    if state.get("force_regen_lines"):
        state["phase"] = "lines"
        return "sense_lines"

    if phase == "lines":
        return "sense_lines"

    if phase == "ideas":
        return "ideas"

    if phase == "finish":
        return "await"

    return "await"


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    use_platform_store: bool = False,
    locale: str = "ru",
    checkpoint_saver=None,
):
    set_locale(locale)
    set_models_locale(locale)
    log_name = f"ideator_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    callback_handlers = [StreamWriterCallbackHandler(), json_handler]
    if config.LANGFUSE_URL and len(config.LANGFUSE_URL) > 0:
        langfuse = Langfuse(
            public_key=config.LANGFUSE_PUBLIC,
            secret_key=config.LANGFUSE_SECRET,
            host=config.LANGFUSE_URL,
        )
        lf_handler = CallbackHandler()
        callback_handlers += [lf_handler]

    memory = None if use_platform_store else checkpoint_saver or MemorySaver()
    llm = get_llm(model="base", provider=provider.value, temperature=0.4, streaming=True)
    #llm = get_llm(model="base", provider="openai_4", temperature=0.4)

    builder = StateGraph(IdeatorAgentState)
    builder.add_node("init", create_init_node())
    builder.add_node("sense_lines_prepare", create_sense_lines_prepare_node())
    builder.add_node("sense_lines_llm", create_sense_lines_llm_node(llm))
    builder.add_node("sense_lines_post", create_sense_lines_post_node())
    builder.add_node("ideas_llm", create_ideas_llm_node(llm))
    builder.add_node("ideas_post", create_ideas_post_node())

    builder.add_conditional_edges(
        START,
        route,
        {
            "init": "init",
            "sense_lines": "sense_lines_prepare",
            "ideas": "ideas_llm",
            "await": END,
        },
    )
    builder.add_edge("init", END)
    builder.add_conditional_edges(
        "sense_lines_prepare",
        route_sense_lines,
        {
            "sense_lines_llm": "sense_lines_llm",
            "sense_lines_post": "sense_lines_post",
        },
    )
    builder.add_edge("sense_lines_llm", "sense_lines_post")
    builder.add_edge("sense_lines_post", END)
    builder.add_edge("ideas_llm", "ideas_post")
    builder.add_edge("ideas_post", END)

    graph = builder.compile(checkpointer=memory, debug=False).with_config({"callbacks": callback_handlers})
    return graph


if __name__ == "__main__":
    print("Initializing Ideator Agent...")
    graph = initialize_agent()
    config_run = {"configurable": {"thread_id": "ideator_demo"}}
    print(graph.get_graph().draw_ascii())
