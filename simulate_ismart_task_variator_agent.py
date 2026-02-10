from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from agents.utils import extract_text
from agents.ismart_task_variator_agent.state import VariatorAgentContext

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate ismart_task_variator_agent with one greeting step and "
            "multiple JSON payload steps."
        )
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        default=Path("data/tasks.json"),
        help="Path to a JSON file containing a list of objects.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/variator.md"),
        help="Path to markdown output file with all agent responses.",
    )
    parser.add_argument(
        "--thread-id",
        default="ismart_task_variator_simulation",
        help="Thread id used for graph memory checkpointing.",
    )
    return parser.parse_args()


def _load_payloads(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON file not found: {path}")

    raw = path.read_text(encoding="utf-8-sig")
    data = json.loads(raw)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    payloads: list[dict[str, Any]] = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item #{idx} is not an object.")
        payloads.append(item)

    return payloads


def _extract_last_ai_text(result: dict[str, Any]) -> str:
    messages = result.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = extract_text(message)
            return content
    return ""


def _append_step(
    lines: list[str],
    step: int,
    user_content: str,
    ai_content: str,
    *,
    user_block_lang: str,
) -> None:
    lines.append(f"## Step {step}")
    lines.append("")
    lines.append("### User")
    lines.append(f"```{user_block_lang}")
    lines.append(user_content)
    lines.append("```")
    lines.append("")
    lines.append("### Agent")
    lines.append(ai_content or "_(empty response)_")
    lines.append("")


def run_simulation(input_json: Path, output_path: Path, thread_id: str) -> None:
    try:
        from agents.ismart_task_variator_agent.agent import initialize_agent
        from agents.utils import ModelType
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime environment dependent
        missing = getattr(exc, "name", "unknown")
        raise RuntimeError(
            f"Missing required dependency '{missing}'. Install project dependencies first."
        ) from exc

    payloads = _load_payloads(input_json)

    agent_graph = initialize_agent(
        provider=ModelType.GPT_THINK,
        locale="ru",
        streaming=False,
    )
    config = {"configurable": {"thread_id": thread_id}}
    ctx = VariatorAgentContext(mode="auto")

    report_lines: list[str] = [
        "# iSmart Task Variator Simulation",
        "",
        f"- Agent: `ismart_task_variator_agent`",
        f"- Input: `{input_json.as_posix()}`",
        f"- Total JSON items: `{len(payloads)}`",
        "",
    ]

    #_append_step(report_lines, 1, greeting, greeting_response, user_block_lang="text")

    for idx, payload in enumerate(payloads, start=2):
        thread_iid = f"{thread_id}_{idx}"
        config = {"configurable": {"thread_id": thread_iid}}

        payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=payload_text)]},
            config=config,
            context=ctx,
        )
        response = _extract_last_ai_text(result)
        _append_step(report_lines, idx, payload_text, response, user_block_lang="json")

    output_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")
    print(f"Saved simulation report to: {output_path}")


if __name__ == "__main__":
    args = _parse_args()
    run_simulation(args.input_json, args.output, args.thread_id)
