from __future__ import annotations

from deepagents.middleware.subagents import SubAgent

import config
from agents.tools.yandex_search import YandexSearchTool


WEB_SEARCH_AGENT_ID = "web_search_agent"

WEB_SEARCH_AGENT_DESCRIPTION = (
    "Stateless web research specialist using Yandex Search. Use for isolated public "
    "web lookup, multi-query search, source selection, and compact sourced reports."
)

WEB_SEARCH_AGENT_SYSTEM_PROMPT = """You are web_search_agent, a stateless web research specialist.

Your job is to use Yandex web search to answer the manager's research request with current public-source information.

Rules:
- Use the `web_search` tool for external web information. Do not answer from memory when the manager asks for public/current sources.
- You may run up to 3 focused searches with different query wording when the first search is insufficient.
- Prefer precise queries over broad ones. Use the user's language unless an English query is more likely to find authoritative sources.
- Return only the final research report to the manager.
- Include source links from the search results in the final report whenever you use external information.
- Keep the answer compact: key findings, relevant caveats, and sources.
- If search fails or evidence is weak, say that explicitly instead of inventing facts.
"""


def build_web_search_tool(
    *,
    max_results: int = 5,
    summarize: bool = True,
) -> YandexSearchTool:
    missing: list[str] = []
    if not config.YA_API_KEY:
        missing.append("YA_API_KEY")
    if not config.YA_FOLDER_ID:
        missing.append("YA_FOLDER_ID")
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"{WEB_SEARCH_AGENT_ID} requires environment variables: {missing_text}"
        )

    return YandexSearchTool(
        api_key=config.YA_API_KEY,
        folder_id=config.YA_FOLDER_ID,
        max_results=max_results,
        summarize=summarize,
    )


def build_web_search_subagent(
    *,
    max_results: int = 5,
    summarize: bool = True,
) -> SubAgent:
    return SubAgent(
        name=WEB_SEARCH_AGENT_ID,
        description=WEB_SEARCH_AGENT_DESCRIPTION,
        system_prompt=WEB_SEARCH_AGENT_SYSTEM_PROMPT,
        tools=[build_web_search_tool(max_results=max_results, summarize=summarize)],
    )
