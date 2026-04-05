from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agents.utils import ModelType, extract_text
from bot_service.initialize_agents import parse_yaml
from langchain_core.messages import AIMessage, HumanMessage


EXIT_COMMANDS = {"exit", "/exit", "quit", "/quit"}
RESET_COMMANDS = {"reset", "/reset"}
HELP_COMMANDS = {"help", "/help"}
MULTILINE_START_COMMANDS = {"/multi", "/multiline", "<<<"}
MULTILINE_END_COMMANDS = {"/send", "/end", ">>>"}
MULTILINE_CANCEL_COMMANDS = {"/cancel", "/abort"}
CHOOSE_PRODUCT_COMMANDS = {"/choose-product", "/product"}
LIST_PRODUCTS_COMMANDS = {"/list-products", "/products"}
DEFAULT_AGENT_CONFIG_PATH = "data/config/bot_service/load.json"
DEFAULT_USER_ID = "ingos-product-cli"
DEFAULT_USER_ROLE = "default"
PRODUCT_AGENT_ID_PREFIX = "product_"
PRODUCT_AGENT_MODULE_PREFIX = "agents.ingos_product_agent"
REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProductAgentSpec:
    id: str
    name: str
    description: str
    product: str
    module_path: str
    provider: ModelType
    is_active: bool
    init_params: dict[str, Any]


def _parse_provider(raw_value: str) -> ModelType:
    normalized = raw_value.strip().lower()
    for provider in ModelType:
        if normalized in {provider.value.lower(), provider.name.lower()}:
            return provider
    available = ", ".join(sorted(provider.value for provider in ModelType))
    raise ValueError(f"Unknown provider '{raw_value}'. Available values: {available}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct interactive CLI for the ingos_product_agent family."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional prompt for a single non-interactive run.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_AGENT_CONFIG_PATH,
        help="Path to the bot-service agent JSON/YAML config.",
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--agent-id",
        default=None,
        help="Start with a specific product agent id, for example product_Car.",
    )
    selection_group.add_argument(
        "--product",
        default=None,
        help="Start with a specific product name, for example Car or Инголаб.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Optional provider override for the selected product agent.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Optional thread id. If omitted, a random one is generated.",
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help="User id passed to the agent config.",
    )
    parser.add_argument(
        "--user-role",
        default=DEFAULT_USER_ROLE,
        help="User role passed to the agent config.",
    )
    parser.add_argument(
        "--prefetch-top-k",
        type=int,
        default=3,
        help="Vector prefetch document count passed to ingos_product_agent.",
    )
    parser.add_argument(
        "--list-products",
        action="store_true",
        help="Print the available ingos product agents and exit.",
    )
    return parser.parse_args()


def _resolve_config_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _load_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = parse_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"Agent config must be a JSON/YAML object: {path}")
    return data


def _load_product_agent_specs(raw_config_path: str) -> list[ProductAgentSpec]:
    path = _resolve_config_path(raw_config_path)
    config = _load_config_file(path)
    raw_agents = config.get("agents")
    if not isinstance(raw_agents, list):
        raise ValueError(f"Agent config must contain an 'agents' list: {path}")

    specs: list[ProductAgentSpec] = []
    for entry in raw_agents:
        if not isinstance(entry, dict):
            continue

        agent_id = entry.get("id")
        module_path = entry.get("module") or entry.get("path")
        params = entry.get("params") or {}

        if not isinstance(agent_id, str) or not agent_id.startswith(PRODUCT_AGENT_ID_PREFIX):
            continue
        if not isinstance(module_path, str) or not module_path.startswith(PRODUCT_AGENT_MODULE_PREFIX):
            continue
        if not isinstance(params, dict):
            raise ValueError(f"Agent '{agent_id}' params must be a mapping.")

        product = params.get("product")
        if not isinstance(product, str) or not product.strip():
            raise ValueError(f"Agent '{agent_id}' is missing params.product in {path}.")

        raw_provider = params.get("provider") or ModelType.GPT.value
        provider = _parse_provider(str(raw_provider))

        init_params = dict(params)
        init_params.pop("provider", None)
        init_params.pop("checkpoint_saver", None)

        specs.append(
            ProductAgentSpec(
                id=agent_id,
                name=str(entry.get("name") or agent_id),
                description=str(entry.get("description") or ""),
                product=product,
                module_path=module_path,
                provider=provider,
                is_active=bool(entry.get("is_active", True)),
                init_params=init_params,
            )
        )

    return specs


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _find_product_spec(
    specs: list[ProductAgentSpec],
    *,
    agent_id: Optional[str] = None,
    product_name: Optional[str] = None,
) -> ProductAgentSpec | None:
    if agent_id:
        target = _normalize_text(agent_id)
        for spec in specs:
            if _normalize_text(spec.id) == target:
                return spec
        return None

    if product_name:
        target = _normalize_text(product_name)
        for spec in specs:
            if _normalize_text(spec.product) == target or _normalize_text(spec.name) == target:
                return spec
        return None

    return None


def _print_product_list(specs: list[ProductAgentSpec]) -> None:
    print("Available ingos_product_agent family agents:")
    print("")
    for index, spec in enumerate(specs, start=1):
        status = "" if spec.is_active else " [inactive in service config]"
        print(f"{index}. {spec.name} ({spec.id}){status}")
    print("")


def _choose_product_spec(specs: list[ProductAgentSpec]) -> ProductAgentSpec:
    if not specs:
        raise ValueError("No ingos_product_agent family agents found in the config.")

    _print_product_list(specs)
    while True:
        raw_choice = input("Choose product number> ").strip()
        if raw_choice.lower() in EXIT_COMMANDS:
            raise KeyboardInterrupt
        try:
            choice = int(raw_choice)
        except ValueError:
            print(f"Enter a number from 1 to {len(specs)}.")
            continue
        if 1 <= choice <= len(specs):
            return specs[choice - 1]
        print(f"Enter a number from 1 to {len(specs)}.")


def _build_human_message(*, user_text: str, reset: bool = False) -> HumanMessage:
    if reset:
        return HumanMessage(content=[{"type": "reset", "text": user_text or "RESET"}])
    return HumanMessage(content=[{"type": "text", "text": user_text}])


def _extract_last_ai_text(result: dict[str, Any]) -> str:
    messages = result.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return extract_text(message)
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
    return ""


def _new_config(
    *,
    user_id: str,
    user_role: str,
    thread_id: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    resolved_thread_id = thread_id or f"ingos-product-cli-{uuid.uuid4().hex}"
    return resolved_thread_id, {
        "configurable": {
            "thread_id": resolved_thread_id,
            "user_id": user_id,
            "user_role": user_role,
        }
    }


def _initialize_agent_for_spec(
    spec: ProductAgentSpec,
    *,
    provider: ModelType,
    prefetch_top_k: int,
) -> Any:
    module = importlib.import_module(spec.module_path)
    init_fn = getattr(module, "initialize_agent", None)
    if not callable(init_fn):
        raise AttributeError(f"{spec.module_path} does not expose initialize_agent().")

    params = dict(spec.init_params)
    params["provider"] = provider
    params["prefetch_top_k"] = prefetch_top_k

    signature = inspect.signature(init_fn)
    accepted_names = set(signature.parameters)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_kwargs:
        params = {key: value for key, value in params.items() if key in accepted_names}

    return init_fn(**params)


def _invoke_turn(
    agent: Any,
    *,
    user_text: str,
    config: dict[str, Any],
    reset: bool = False,
) -> str:
    result = agent.invoke(
        {"messages": [_build_human_message(user_text=user_text, reset=reset)]},
        config=config,
    )
    if not isinstance(result, dict):
        return ""
    return _extract_last_ai_text(result)


def _print_help() -> None:
    print("Commands:")
    print("  /help            show commands")
    print("  /list-products   show product agents")
    print("  /choose-product  switch product agent using a numbered list")
    print("  /multi           enter multiline mode")
    print("  /reset           start a fresh thread for the current product")
    print("  /exit            leave the CLI")


def _collect_multiline_input() -> Optional[str]:
    print("Multiline mode: finish with /send, cancel with /cancel.")
    lines: list[str] = []
    while True:
        line = input("...> ")
        command = line.strip().lower()
        if command in EXIT_COMMANDS:
            return None
        if command in MULTILINE_CANCEL_COMMANDS:
            return ""
        if command in MULTILINE_END_COMMANDS:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _selected_provider(spec: ProductAgentSpec, provider_override: Optional[str]) -> ModelType:
    if provider_override:
        return _parse_provider(provider_override)
    return spec.provider


def _activate_spec(
    *,
    spec: ProductAgentSpec,
    provider_override: Optional[str],
    prefetch_top_k: int,
    agent_cache: dict[tuple[str, str, int], Any],
) -> tuple[Any, ModelType]:
    provider = _selected_provider(spec, provider_override)
    cache_key = (spec.id, provider.value, prefetch_top_k)
    agent = agent_cache.get(cache_key)
    if agent is None:
        print(f"Loading {spec.id} directly from {spec.module_path}...")
        agent = _initialize_agent_for_spec(
            spec,
            provider=provider,
            prefetch_top_k=prefetch_top_k,
        )
        agent_cache[cache_key] = agent
    return agent, provider


def main() -> int:
    args = _parse_args()

    try:
        specs = _load_product_agent_specs(args.config)
        if not specs:
            raise ValueError("No ingos_product_agent family agents found in the config.")
        if args.list_products:
            _print_product_list(specs)
            return 0

        selected_spec = _find_product_spec(
            specs,
            agent_id=args.agent_id,
            product_name=args.product,
        )
        if selected_spec is None:
            selected_spec = _choose_product_spec(specs)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
    except Exception as exc:
        print(f"Failed to start CLI: {exc}", file=sys.stderr)
        return 1

    agent_cache: dict[tuple[str, str, int], Any] = {}

    try:
        agent, provider = _activate_spec(
            spec=selected_spec,
            provider_override=args.provider,
            prefetch_top_k=args.prefetch_top_k,
            agent_cache=agent_cache,
        )
        thread_id, run_config = _new_config(
            user_id=args.user_id,
            user_role=args.user_role,
            thread_id=args.thread_id,
        )
    except Exception as exc:
        print(f"Failed to initialize agent: {exc}", file=sys.stderr)
        return 1

    prompt = " ".join(args.prompt).strip()
    if prompt:
        try:
            answer = _invoke_turn(agent, user_text=prompt, config=run_config)
        except KeyboardInterrupt:
            print("Goodbye!")
            return 0
        except Exception as exc:
            print(f"Assistant error: {exc}", file=sys.stderr)
            return 1
        if answer:
            print(answer)
        return 0

    print(
        "ingos_product_agent CLI started "
        f"(thread_id={thread_id}, agent_id={selected_spec.id}, provider={provider.value})."
    )
    print("Direct mode: agents are initialized in-process without bot_service.")
    _print_help()
    print("")

    while True:
        try:
            user_input = input(f"{selected_spec.name}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if not user_input:
            continue

        user_input_lower = user_input.lower()
        if user_input_lower in EXIT_COMMANDS:
            print("Goodbye!")
            return 0
        if user_input_lower in HELP_COMMANDS:
            _print_help()
            print("")
            continue
        if user_input_lower in LIST_PRODUCTS_COMMANDS:
            _print_product_list(specs)
            continue
        if user_input_lower in CHOOSE_PRODUCT_COMMANDS:
            try:
                selected_spec = _choose_product_spec(specs)
                agent, provider = _activate_spec(
                    spec=selected_spec,
                    provider_override=args.provider,
                    prefetch_top_k=args.prefetch_top_k,
                    agent_cache=agent_cache,
                )
                thread_id, run_config = _new_config(
                    user_id=args.user_id,
                    user_role=args.user_role,
                )
            except KeyboardInterrupt:
                print("\nGoodbye!")
                return 0
            except Exception as exc:
                print(f"Failed to switch product agent: {exc}", file=sys.stderr)
                continue
            print(
                f"Switched to {selected_spec.id}. Started fresh thread: {thread_id} "
                f"(provider={provider.value}).\n"
            )
            continue
        if user_input_lower in MULTILINE_START_COMMANDS:
            multiline_input = _collect_multiline_input()
            if multiline_input is None:
                print("Goodbye!")
                return 0
            if not multiline_input:
                print("Canceled.\n")
                continue
            user_input = multiline_input
            user_input_lower = user_input.lower()
        if user_input_lower in RESET_COMMANDS:
            thread_id, run_config = _new_config(
                user_id=args.user_id,
                user_role=args.user_role,
            )
            print(f"Started fresh thread: {thread_id}\n")
            continue

        try:
            answer = _invoke_turn(agent, user_text=user_input, config=run_config)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return 0
        except Exception as exc:
            print(f"\nAssistant error: {exc}\n", file=sys.stderr)
            continue

        print(f"\nAssistant: {answer or '(empty response)'}\n")


if __name__ == "__main__":
    raise SystemExit(main())
