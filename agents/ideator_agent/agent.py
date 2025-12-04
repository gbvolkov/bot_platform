from __future__ import annotations

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

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

import config

from agents.tools.think import ThinkTool
from agents.utils import ModelType, get_llm
from agents.structured_prompt_utils import build_json_prompt, provider_then_tool
from platform_utils.llm_logger import JSONFileTracer

from .models import ArticleRecord, IdeatorReport
from .prompts import (
    FACT_REF_HINT,
    IDEAS_INSTRUCTION,
    IDEATOR_SYSTEM_PROMPT,
    SENSE_LINE_INSTRUCTION,
)
from .report_loader import load_report
from .state import IdeatorAgentContext, IdeatorAgentState

LOG = logging.getLogger(__name__)


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
    article_ids: Annotated[List[int], "Ids from the provided list only"]
    region_note: Annotated[str, "Region applicability note"]


class SenseLineResponse(TypedDict):
    assistant_message: Annotated[str, "Natural language reply for the user"]
    sense_lines: List[SenseLineItem]
    decision: NotRequired[Decision]


class IdeaItem(TypedDict):
    title: Annotated[str, "Idea headline (1 short sentence)"]
    summary: Annotated[str, "1-2 sentences grounded strictly in provided articles"]
    article_ids: Annotated[List[int], "Ids from the provided list only"]
    region_note: Annotated[str, "Region applicability note"]
    importance_hint: Annotated[NotRequired[str], "High/Medium/Low or empty"]


class IdeaListResponse(TypedDict):
    assistant_message: Annotated[str, "Natural language reply for the user."]
    ideas: Annotated[List[IdeaItem], "List of generated ideas."]
    decision: NotRequired[Decision]


_think_tool = ThinkTool()


def _format_articles(articles: List[ArticleRecord], limit_chars: int = 400) -> str:
    lines: List[str] = []
    for art in articles:
        summary = art.summary
        if len(summary) > limit_chars:
            summary = summary[:limit_chars] + "..."
        lines.append(
            f"[{art.id}] ({art.norm_importance()}; {art.region_label()}) "
            f"{art.display_title()} — {summary} | {art.url}"
        )
    return "\n".join(lines)


def _format_sense_lines(lines: List[Dict[str, Any]]) -> str:
    formatted: List[str] = []
    for idx, line in enumerate(lines or [], start=1):
        formatted.append(
            f"{idx}) {line.get('id') or f'L{idx}'} | {line.get('short_title','')}\n"
            f"{line.get('description','')}\n"
            f"region: {line.get('region_note','')}\n"
            f"articles: {line.get('article_ids', [])}"
        )
    return "\n\n".join(formatted)


def _format_ideas(ideas: List[Dict[str, Any]]) -> str:
    formatted: List[str] = []
    for idx, idea in enumerate(ideas or [], start=1):
        formatted.append(
            f"{idx}) {idea.get('title','')}\n"
            f"{idea.get('summary','')}\n"
            f"region: {idea.get('region_note','')}\n"
            f"articles: {idea.get('article_ids', [])}"
        )
    return "\n\n".join(formatted)


def _fact_refs_for(report: IdeatorReport, article_ids: List[int]) -> List[str]:
    return [art.fact_ref() for art in report.filter_by_ids(article_ids)]


def _build_sense_agent(model: BaseChatModel):
    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        state: IdeatorAgentState = request.state
        report: IdeatorReport = state.get("report")  # type: ignore
        articles = report.sorted_articles()[:80] if report else []
        existing_lines = state.get("sense_lines") or []
        prompt = (
            IDEATOR_SYSTEM_PROMPT
            + "\n"
            + SENSE_LINE_INSTRUCTION
            + "\n"
            + FACT_REF_HINT
            + "\n\n"
            f"search_goal: {report.search_goal if report else ''}\n"
            f"Всего статей в отчёте: {report.total_articles if report else 0}. В выборке для анализа: {len(articles)}.\n"
            f"Список статей (id, importance, region, title, summary, url):\n{_format_articles(articles)}\n\n"
        )
        if existing_lines:
            prompt += (
                "Текущие смысловые линии (сохраняй id и порядок, если идёт обсуждение прошлых вариантов):\n"
                f"{_format_sense_lines(existing_lines)}\n\n"
            )
        prompt += f"{build_json_prompt(SenseLineResponse)}"
        return prompt

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
        report: IdeatorReport = state.get("report")  # type: ignore
        filtered_ids = state.get("filtered_article_ids") or []
        articles = report.filter_by_ids(filtered_ids) if report else []
        existing_ideas = state.get("ideas") or []
        prompt = (
            IDEATOR_SYSTEM_PROMPT
            + "\n"
            + IDEAS_INSTRUCTION
            + "\n"
            + FACT_REF_HINT
            + "\n\n"
            f"Активная смысловая линия: {state.get('selected_line_id', '')}\n"
            f"Статей в контексте: {len(articles)}.\n"
            f"Доступные статьи (id, importance, region, title, summary, url):\n{_format_articles(articles)}\n\n"
        )
        if existing_ideas:
            prompt += (
                "Текущие идеи (сохраняй порядок при обсуждении и уточнениях):\n"
                f"{_format_ideas(existing_ideas)}\n\n"
            )
        prompt += f"{build_json_prompt(IdeaListResponse)}"
        return prompt

    return create_agent(
        model=model,
        tools=[_think_tool],
        middleware=[build_prompt, provider_then_tool],
        response_format=IdeaListResponse,
        state_schema=IdeatorAgentState,
        context_schema=IdeatorAgentContext,
    )


def create_init_node():
    def init_node(
        state: IdeatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IdeatorAgentContext],
    ) -> IdeatorAgentState:
        if runtime.context and runtime.context.get("report_path"):
            report_path = runtime.context["report_path"]
            state["report_path"] = report_path
            state["report"] = load_report(report_path)
        if "ideas_cache" not in state:
            state["ideas_cache"] = {}
        if "force_regen_lines" not in state:
            state["force_regen_lines"] = False
        if "force_regen" not in state:
            state["force_regen"] = False
        if "phase" not in state:
            state["phase"] = "lines"
        if not state.get("greeted"):
            greet = "Привет! Я — Генератор идей. Отчёт загружен, готов выделить смысловые линии."
            state["messages"] = (state.get("messages") or []) + [AIMessage(content=greet)]
            state["greeted"] = True
        return state

    return init_node


def create_sense_lines_node(model: BaseChatModel):
    agent = _build_sense_agent(model)

    def sense_lines_node(
        state: IdeatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IdeatorAgentContext],
    ) -> IdeatorAgentState:
        LOG.info("sense_lines_node: start, messages=%d", len(state.get("messages", [])))

        result: IdeatorAgentState = agent.invoke(state, config=config, context=runtime.context)
        structured = result.get("structured_response") or {}
        sense_lines: List[Dict[str, Any]] = structured.get("sense_lines") or state.get("sense_lines") or []
        assistant_message: str = structured.get("assistant_message") or ""
        decision: Decision = structured.get("decision") or {}
        result["sense_lines"] = sense_lines
        result["force_regen_lines"] = False
        result["phase"] = result.get("phase") or "lines"
        result.pop("structured_response", None)
        LOG.info("sense_lines_node: generated %d lines", len(sense_lines))

        if assistant_message:
            result["messages"] = result.get("messages", []) + [AIMessage(content=assistant_message)]
        elif sense_lines:
            report: IdeatorReport = state.get("report")  # type: ignore
            formatted_lines = []
            for idx, line in enumerate(sense_lines, start=1):
                fact_refs = _fact_refs_for(report, line.get("article_ids", [])) if report else []
                formatted_lines.append(
                    f"{idx}) {line.get('short_title','')}\n"
                    f"{line.get('description','')}\n"
                    f"{line.get('region_note','')}\n"
                    + "\n".join(fact_refs)
                )
            fallback = "Смысловые линии:\n" + "\n\n".join(formatted_lines)
            result["messages"] = result.get("messages", []) + [AIMessage(content=fallback)]

        if decision.get("regen_lines"):
            result["force_regen_lines"] = True
            result.pop("selected_line_id", None)
            result.pop("filtered_article_ids", None)
            result["phase"] = "lines"

        selected_idx = decision.get("selected_line_index")
        custom_line_text = decision.get("custom_line_text")
        ready_for_ideas = decision.get("consent_generate")
        report: IdeatorReport = state.get("report")  # type: ignore
        if ready_for_ideas and selected_idx and 1 <= selected_idx <= len(sense_lines):
            selected = sense_lines[selected_idx - 1]
            result["selected_line_id"] = selected.get("id") or f"L{selected_idx}"
            result["filtered_article_ids"] = [int(i) for i in selected.get("article_ids", [])]
            result["force_regen"] = True
            result["phase"] = "ideas"
        elif ready_for_ideas and custom_line_text:
            top_ids = [a.id for a in (report.sorted_articles()[:20] if report else [])]
            result["selected_line_id"] = "custom_line"
            result["filtered_article_ids"] = top_ids
            result["force_regen"] = True
            result["phase"] = "ideas"
        elif decision.get("finish"):
            result["phase"] = "finish"
        return result

    return sense_lines_node


def create_ideas_node(model: BaseChatModel):
    agent = _build_ideas_agent(model)

    def ideas_node(
        state: IdeatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IdeatorAgentContext],
    ) -> IdeatorAgentState:
        LOG.info(
            "ideas_node: start, selected_line=%s, filtered_ids=%s",
            state.get("selected_line_id"),
            state.get("filtered_article_ids"),
        )
        result: IdeatorAgentState = agent.invoke(state, config=config, context=runtime.context)
        structured = result.get("structured_response") or {}
        ideas = structured.get("ideas") or state.get("ideas") or []
        assistant_message: str = structured.get("assistant_message") or ""
        decision: Decision = structured.get("decision") or {}
        result["ideas"] = ideas
        result["force_regen"] = False
        result["phase"] = "ideas"
        result.pop("structured_response", None)
        LOG.info("ideas_node: generated %d ideas", len(ideas))

        if assistant_message:
            result["messages"] = result.get("messages", []) + [AIMessage(content=assistant_message)]
        else:
            report: IdeatorReport = state.get("report")  # type: ignore
            formatted = []
            for idea in ideas:
                fact_refs = _fact_refs_for(report, idea.get("article_ids", [])) if report else []
                formatted.append(
                    f"- {idea.get('title','')}: {idea.get('summary','')}\n"
                    f"{idea.get('region_note','')}\n" + "\n".join(fact_refs)
                )
            if formatted:
                fallback = "Идеи по выбранной линии:\n" + "\n\n".join(formatted)
            else:
                fallback = "Не удалось сгенерировать идеи по выбранной линии. Попробуйте выбрать другую линию или уточнить запрос."
            result["messages"] = result.get("messages", []) + [AIMessage(content=fallback)]

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

    return ideas_node




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
        if state.get("force_regen") or not state.get("ideas"):
            return "ideas"
        return "await"

    if phase == "finish":
        return "await"

    return "await"


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    use_platform_store: bool = False,
):
    log_name = f"ideator_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    callback_handlers = [json_handler]
    if config.LANGFUSE_URL and len(config.LANGFUSE_URL) > 0:
        langfuse = Langfuse(
            public_key=config.LANGFUSE_PUBLIC,
            secret_key=config.LANGFUSE_SECRET,
            host=config.LANGFUSE_URL,
        )
        lf_handler = CallbackHandler()
        callback_handlers += [lf_handler]

    memory = None if use_platform_store else MemorySaver()
    llm = get_llm(model="base", provider=provider.value, temperature=0.4)

    builder = StateGraph(IdeatorAgentState)
    builder.add_node("init", create_init_node())
    builder.add_node("sense_lines", create_sense_lines_node(llm))
    builder.add_node("ideas", create_ideas_node(llm))

    builder.add_conditional_edges(
        START,
        route,
        {
            "init": "init",
            "sense_lines": "sense_lines",
            "ideas": "ideas",
            "await": END,
        },
    )
    builder.add_edge("init", END)
    builder.add_edge("sense_lines", END)
    builder.add_edge("ideas", END)

    graph = builder.compile(checkpointer=memory, debug=False).with_config({"callbacks": callback_handlers})
    return graph


if __name__ == "__main__":
    print("Initializing Ideator Agent...")
    graph = initialize_agent()
    config_run = {"configurable": {"thread_id": "ideator_demo"}}
    print(graph.get_graph().draw_ascii())
