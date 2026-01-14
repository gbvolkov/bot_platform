from __future__ import annotations
from typing import Optional
import os

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tomllib  # built-in on Python 3.11+


import config

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.language_models.chat_models import BaseChatModel

from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
#from langchain_gigachat import GigaChat
from .yandex_tools.yandex_tooling import ChatYandexGPTWithTools as ChatYandexGPT



VALID_MODES = {"base", "mini", "nano"}

@dataclass(frozen=True)
class ModelRegistry:
    providers: dict[str, dict[str, str]]

    @classmethod
    def from_toml(cls, path: Path) -> "ModelRegistry":
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("rb") as f:
            data = tomllib.load(f)
        providers = data.get("providers")
        if not isinstance(providers, dict):
            raise ValueError("Invalid config: missing top-level [providers] table.")

        # normalize to lowercase keys and validate modes
        norm: dict[str, dict[str, str]] = {}
        for prov, modes in providers.items():
            if not isinstance(modes, dict):
                raise ValueError(f"Invalid provider section for '{prov}'.")
            lower_modes = {k.lower(): v for k, v in modes.items()}
            # ensure exactly the three modes exist
            missing = VALID_MODES - set(lower_modes.keys())
            extra = set(lower_modes.keys()) - VALID_MODES
            if missing:
                raise ValueError(f"Provider '{prov}' missing modes: {sorted(missing)}.")
            if extra:
                raise ValueError(f"Provider '{prov}' has unknown modes: {sorted(extra)}.")
            norm[prov.lower()] = lower_modes
        return cls(norm)

    def get(self, provider: str, mode: str) -> str:
        p, m = provider.lower(), mode.lower()
        if p not in self.providers:
            known = ", ".join(sorted(self.providers.keys()))
            raise KeyError(f"Unknown provider '{provider}'. Known: {known}")
        if m not in VALID_MODES:
            raise KeyError(f"Unknown mode '{mode}'. Use one of: {', '.join(sorted(VALID_MODES))}")
        return self.providers[p][m]

@lru_cache(maxsize=1)
def _load_registry(config_path: str = "models.toml") -> ModelRegistry:
    return ModelRegistry.from_toml(Path(config_path))

def get_model(provider: str, mode: str, config_path: str = "models.toml") -> str:
    """
    Return the model string for a given provider and mode ('base' | 'mini' | 'nano').
    """
    # Include config_path in the cache key by passing it through _load_registry
    registry = _load_registry(config_path)
    return registry.get(provider, mode)

# Example:
# model = get_model("openai", "mini")  # -> "gpt-4o-mini"

def get_llm(
        model: str = "base", 
        provider: str = None,
        temperature: Optional[float] = 0, 
        frequency_penalty: Optional[float] = None 
    ):
    if provider is None:
        provider = config.LLM_PROVIDER
    llm_model = get_model(provider, model)
    if provider == "openai":
        #TODO: model=="base" is a temporary fix for verbosity issue and sgall be removed in future
        verbosity = "low" if model == "base" else "medium"
        return ChatOpenAI(model=llm_model, 
                        temperature=temperature, 
                        frequency_penalty=frequency_penalty,
                        verbosity=verbosity
                        )
    elif provider == "openai_4":
        return ChatOpenAI(model=llm_model, 
                          temperature=temperature, 
                          frequency_penalty=frequency_penalty)
    elif provider == "openai_gv":
        return ChatOpenAI(model=llm_model, 
                          api_key=os.getenv("OPENAI_API_KEY_PERSONAL"),
                          temperature=temperature, 
                          frequency_penalty=frequency_penalty)
    elif provider == "gigachat":
        raise NotImplementedError
        #return GigaChat(
        #    credentials=config.GIGA_CHAT_AUTH, 
        #    model=llm_model,
        #    verify_ssl_certs=False,
        #    temperature=temperature,
        #    frequency_penalty=frequency_penalty,
        #    scope = config.GIGA_CHAT_SCOPE)
    elif provider == "mistral":
        return ChatMistralAI(model=llm_model, temperature=temperature, frequency_penalty=frequency_penalty)
    elif provider == "yandex":
        model_name=f'gpt://{config.YA_FOLDER_ID}/{llm_model}'
        return ChatYandexGPT(
            #iam_token = None,
            api_key = config.YA_API_KEY, 
            folder_id=config.YA_FOLDER_ID, 
            model_uri=model_name,
            temperature=temperature
            )


def build_model_fallback_middleware(
    primary_llm: BaseChatModel,
    *,
    alternative_llm: BaseChatModel,
    primary_retries: int = 3,
) -> ModelFallbackMiddleware:
    """
    Create a ModelFallbackMiddleware that retries the primary model first.

    It will retry the primary model `primary_retries` times (so the first N
    fallbacks are still the same model) before switching to the alternative.
    """
    if primary_retries < 0:
        raise ValueError("primary_retries must be >= 0")
    fallback_chain = [primary_llm] * primary_retries + [alternative_llm]
    return ModelFallbackMiddleware(*fallback_chain)


def with_llm_fallbacks(
    primary_llm: BaseChatModel,
    *,
    alternative_llm: BaseChatModel,
    primary_retries: int = 3,
) -> BaseChatModel:
    """
    Wrap an LLM runnable with retry-then-fallback behavior using instances.

    The primary model will be retried `primary_retries` times before the
    alternative model is attempted.
    """
    if primary_retries < 0:
        raise ValueError("primary_retries must be >= 0")
    fallback_chain = [primary_llm] * primary_retries + [alternative_llm]
    return primary_llm.with_fallbacks(fallback_chain)
