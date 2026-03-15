from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import uuid
from typing import Any, Mapping

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools.base import ToolException
from langgraph.types import Command

from agents.sysadmin_agent.agent import initialize_agent
from agents.utils import ModelType, extract_text, get_llm


DEFAULT_MODE = "human"
DEFAULT_SCENARIO = (
    "You are an office employee asking a sysadmin for help with a Linux server. "
    "The current issue is that after a docker compose restart the web application "
    "started returning 502 errors. You can run commands when asked and paste short, "
    "plausible outputs."
)
CUSTOMER_SYSTEM_PROMPT = """
You are simulating the customer in a sysadmin support dialog.

Rules:
- You are the customer, not the sysadmin.
- Send exactly one natural next message.
- Keep it short: 1 to 4 sentences.
- Stay consistent with the scenario and prior conversation.
- If the sysadmin asked you to run a command or check something, reply with a plausible result.
- If the problem looks solved, say it is fixed and thank the sysadmin.
- Do not use markdown fences or speaker labels.
""".strip()
HELP_TEXT = """
Commands:
  /mode human        switch customer mode to human
  /mode ai           switch customer mode to AI
  /scenario <text>   replace the AI customer scenario
  /reset             start a fresh dialog session
  /help              show commands
  /exit              leave the simulator

In AI mode, press Enter to generate the next customer turn.
""".strip()
_MCP_AUTH_ERROR_TEXT = "Sysadmin MCP auth is not configured."
_MCP_AUTH_FAILED_TEXT = "Sysadmin MCP authentication failed."
_MCP_AUTH_REFRESH_ERROR_TEXT = "Sysadmin MCP authentication failed after refreshing the access token."
_MCP_URL = os.environ.get("SYSADMIN_MCP_URL", "http://127.0.0.1:8000/mcp")
_SECRET_HINTS = ("password", "passphrase", "secret")


def _parse_provider(raw_value: str) -> ModelType:
    normalized = raw_value.strip().lower()
    for provider in ModelType:
        if normalized in {provider.value.lower(), provider.name.lower()}:
            return provider
    available = ", ".join(sorted(provider.value for provider in ModelType))
    raise ValueError(f"Unknown provider '{raw_value}'. Available values: {available}")


def _new_config() -> dict[str, Any]:
    return {"configurable": {"thread_id": f"sysadmin-sim-{uuid.uuid4().hex}"}}


def _latest_ai_text(result: dict[str, Any]) -> str:
    for message in reversed(result.get("messages") or []):
        if isinstance(message, AIMessage):
            return extract_text(message)
    return ""


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return None
    payload = getattr(interrupts[-1], "value", interrupts[-1])
    return payload if isinstance(payload, dict) else {"content": str(payload)}


def _is_secret_field(field_name: str, field_schema: Mapping[str, Any]) -> bool:
    if str(field_schema.get("format") or "").strip().lower() == "password":
        return True
    hint_text = " ".join(
        [
            field_name,
            str(field_schema.get("title") or ""),
            str(field_schema.get("description") or ""),
        ]
    ).lower()
    return any(token in hint_text for token in _SECRET_HINTS)


def _interrupt_fields(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_fields = payload.get("fields")
    if isinstance(raw_fields, list):
        explicit_fields = [field for field in raw_fields if isinstance(field, dict)]
        if explicit_fields:
            return [dict(field) for field in explicit_fields]

    requested_schema = payload.get("requestedSchema")
    if not isinstance(requested_schema, Mapping):
        return []

    properties = requested_schema.get("properties")
    required = requested_schema.get("required")
    if not isinstance(properties, Mapping):
        return []

    required_fields = set()
    if isinstance(required, list):
        required_fields = {str(field_name) for field_name in required}

    fields: list[dict[str, Any]] = []
    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, Mapping):
            continue
        fields.append(
            {
                "name": str(field_name),
                "title": field_schema.get("title") or str(field_name),
                "description": field_schema.get("description"),
                "required": str(field_name) in required_fields,
                "type": field_schema.get("type"),
                "format": field_schema.get("format"),
                "secret": _is_secret_field(str(field_name), field_schema),
            }
        )
    return fields


def _read_interrupt_input(prompt: str, *, secret: bool) -> str:
    raw_value = getpass.getpass(prompt) if secret else input(prompt)
    if raw_value.strip().lower() in {"/exit", "exit"}:
        raise KeyboardInterrupt
    return raw_value


def _interrupt_resume_value(payload: Mapping[str, Any]) -> Any:
    fields = _interrupt_fields(payload)
    non_approval_fields = [
        field for field in fields if str(field.get("name") or "") != "approve"
    ]

    if payload.get("responseMode") == "text" and len(non_approval_fields) == 1:
        field = non_approval_fields[0]
        label = str(field.get("title") or field.get("name") or "Value")
        value = _read_interrupt_input(
            f"{label}> ",
            secret=bool(field.get("secret")),
        )
        return {"value": value}

    if fields:
        response: dict[str, Any] = {}
        for field in fields:
            field_name = str(field.get("name") or "").strip()
            if not field_name:
                continue
            label = str(field.get("title") or field_name)
            field_type = str(field.get("type") or "").lower()
            value = _read_interrupt_input(
                f"{label}> ",
                secret=bool(field.get("secret")),
            )
            if not value and not field.get("required"):
                continue
            if field_type == "boolean":
                response[field_name] = value.strip().lower() in {"y", "yes", "true", "1"}
            else:
                response[field_name] = value
        return response

    return _read_interrupt_input("Approval> ", secret=False).strip()


def _format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "(conversation has not started yet)"
    return "\n".join(f"{speaker}: {text}" for speaker, text in history[-12:])


def _is_missing_mcp_auth(error: BaseException) -> bool:
    return _MCP_AUTH_ERROR_TEXT in str(error)


def _is_failed_mcp_auth_refresh(error: BaseException) -> bool:
    return _MCP_AUTH_REFRESH_ERROR_TEXT in str(error)


def _is_failed_mcp_auth(error: BaseException) -> bool:
    return _MCP_AUTH_FAILED_TEXT in str(error)


def _walk_exceptions(error: BaseException) -> list[BaseException]:
    nested = getattr(error, "exceptions", None)
    if not nested:
        return [error]

    flat: list[BaseException] = []
    for item in nested:
        if isinstance(item, BaseException):
            flat.extend(_walk_exceptions(item))
    return flat


def _translate_agent_error(error: BaseException) -> RuntimeError | None:
    if _is_missing_mcp_auth(error):
        return RuntimeError(
            "Simulator stopped: sysadmin MCP auth is not configured.\n"
            "Configure the agent environment with SYSADMIN_MCP_OAUTH_ISSUER_URL, "
            "SYSADMIN_MCP_OAUTH_CLIENT_ID, and SYSADMIN_MCP_OAUTH_CLIENT_SECRET."
        )
    if _is_failed_mcp_auth_refresh(error):
        return RuntimeError(
            "Simulator stopped: sysadmin MCP authentication failed.\n"
            "The agent refreshed its access token, but the MCP server still returned 401. "
            "Check MCP OAuth credentials or MCP server authorization."
        )
    if _is_failed_mcp_auth(error):
        return RuntimeError(
            "Simulator stopped: sysadmin MCP authentication failed.\n"
            "Check the MCP bearer token or MCP OAuth credentials."
        )

    for item in _walk_exceptions(error):
        if isinstance(item, ToolException):
            return RuntimeError(
                "Simulator stopped: sysadmin agent tool call failed.\n"
                f"{item}"
            )
        if isinstance(item, httpx.HTTPStatusError) and item.response is not None:
            if item.response.status_code == 401:
                return RuntimeError(
                    "Simulator stopped: sysadmin MCP authentication failed.\n"
                    "The agent could not authenticate to the MCP server. "
                    "Check MCP OAuth credentials or MCP server authorization."
                )
            if item.response.status_code == 503:
                return RuntimeError(
                    "Simulator stopped: sysadmin MCP server is unavailable.\n"
                    f"Check that the MCP server is running and reachable at {_MCP_URL}."
                )
        if isinstance(item, (httpx.ConnectError, httpx.ConnectTimeout)):
            return RuntimeError(
                "Simulator stopped: sysadmin MCP server is unreachable.\n"
                f"Check that the MCP server is running and reachable at {_MCP_URL}."
            )

    return None


async def _generate_customer_reply(
    *,
    customer_llm: Any,
    scenario: str,
    history: list[tuple[str, str]],
) -> str:
    prompt = (
        f"Scenario:\n{scenario}\n\n"
        f"Conversation so far:\n{_format_history(history)}\n\n"
        "Write the next customer message."
    )
    response = await customer_llm.ainvoke(
        [
            SystemMessage(content=CUSTOMER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    text = extract_text(response).strip()
    return text or "I checked it. What should I do next?"


async def _invoke_agent_until_complete(
    *,
    graph: Any,
    config: dict[str, Any],
    customer_text: str | None,
) -> str:
    pending_input: Any
    if customer_text is None:
        pending_input = {}
    else:
        pending_input = {"messages": [HumanMessage(content=customer_text)]}

    while True:
        try:
            result = await graph.ainvoke(pending_input, config=config)
        except Exception as exc:
            translated = _translate_agent_error(exc)
            if translated is not None:
                raise translated from None
            raise
        interrupt_payload = _interrupt_payload(result)
        if interrupt_payload is None:
            return _latest_ai_text(result)

        question = (
            interrupt_payload.get("question")
            or interrupt_payload.get("content")
            or "Agent requested approval."
        )
        label = (
            "[Approval required]"
            if interrupt_payload.get("type") == "approval"
            else "[Additional input required]"
        )
        print(f"\n{label}\n{question}\n")
        pending_input = Command(resume=_interrupt_resume_value(interrupt_payload))


def _handle_command(
    raw_text: str,
    *,
    mode: str,
    scenario: str,
) -> tuple[bool, str, str, bool]:
    command = raw_text.strip()
    if not command.startswith("/"):
        return False, mode, scenario, False

    known_commands = {
        "/help",
        "/reset",
        "/mode human",
        "/mode ai",
        "/exit",
        "exit",
    }
    if command.lower() not in known_commands and not command.lower().startswith("/scenario "):
        return False, mode, scenario, False

    lowered = command.lower()
    if lowered in {"/exit", "exit"}:
        raise KeyboardInterrupt
    if lowered == "/help":
        print(f"\n{HELP_TEXT}\n")
        return True, mode, scenario, False
    if lowered == "/reset":
        return True, mode, scenario, True
    if lowered == "/mode human":
        print("\n[Simulator] customer mode: human\n")
        return True, "human", scenario, False
    if lowered == "/mode ai":
        print("\n[Simulator] customer mode: ai\n")
        return True, "ai", scenario, False
    if lowered.startswith("/scenario "):
        updated_scenario = command[len("/scenario ") :].strip()
        if updated_scenario:
            print("\n[Simulator] AI customer scenario updated.\n")
            return True, mode, updated_scenario, False
    print("\n[Simulator] unknown command. Use /help.\n")
    return True, mode, scenario, False


async def run_dialog_simulator(
    *,
    provider: ModelType,
    mode: str = DEFAULT_MODE,
    scenario: str = DEFAULT_SCENARIO,
) -> None:
    graph = initialize_agent(provider=provider, streaming=False)
    customer_llm = get_llm(
        model="mini",
        provider=provider.value,
        temperature=0.8,
        streaming=False,
    )

    history: list[tuple[str, str]] = []
    config = _new_config()

    print("\nSysadmin dialog simulator")
    print(f"Customer mode: {mode}")
    print(HELP_TEXT)

    opening = await _invoke_agent_until_complete(graph=graph, config=config, customer_text=None)
    if opening:
        print(f"\nSysadmin agent: {opening}\n")
        history.append(("agent", opening))

    while True:
        prompt = "Customer (human)> " if mode == "human" else "Customer (AI, Enter=generate)> "
        raw_text = input(prompt)
        handled, mode, scenario, should_reset = _handle_command(raw_text, mode=mode, scenario=scenario)
        if should_reset:
            history = []
            config = _new_config()
            print("\n[Simulator] new dialog session started.\n")
            opening = await _invoke_agent_until_complete(graph=graph, config=config, customer_text=None)
            if opening:
                print(f"Sysadmin agent: {opening}\n")
                history.append(("agent", opening))
            continue
        if handled:
            continue

        customer_text = raw_text.strip()
        if mode == "ai" and not customer_text:
            customer_text = await _generate_customer_reply(
                customer_llm=customer_llm,
                scenario=scenario,
                history=history,
            )
            print(f"\nCustomer (AI): {customer_text}\n")

        if not customer_text:
            continue

        history.append(("customer", customer_text))
        agent_reply = await _invoke_agent_until_complete(
            graph=graph,
            config=config,
            customer_text=customer_text,
        )
        if agent_reply:
            print(f"\nSysadmin agent: {agent_reply}\n")
            history.append(("agent", agent_reply))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dialog simulator for the sysadmin agent.")
    parser.add_argument(
        "--provider",
        default=ModelType.GPT.value,
        help="LLM provider for both the sysadmin agent and AI customer.",
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=["human", "ai"],
        help="Initial customer mode. Default: human.",
    )
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        help="Scenario prompt used when the customer mode is AI.",
    )
    args = parser.parse_args()

    provider = _parse_provider(args.provider)
    try:
        asyncio.run(
            run_dialog_simulator(
                provider=provider,
                mode=args.mode,
                scenario=args.scenario,
            )
        )
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
    except RuntimeError as exc:
        if _translate_agent_error(exc) is not None or "Simulator stopped:" in str(exc):
            print(
                f"\n{exc}\n"
            )
            return
        raise


if __name__ == "__main__":
    main()
