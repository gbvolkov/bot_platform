from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import ingos_product_agent_cli as cli
from agents.utils import ModelType


def test_load_product_agent_specs_reads_inactive_family_entries(tmp_path):
    config_path = tmp_path / "load.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "id": "product_Car",
                        "name": "Car",
                        "description": "Car expert.",
                        "module": "agents.ingos_product_agent",
                        "is_active": True,
                        "params": {
                            "provider": "openai",
                            "product": "Car",
                            "use_platform_store": False,
                            "checkpoint_saver": "SQLite",
                            "role": "default",
                        },
                    },
                    {
                        "id": "product_Инголаб",
                        "name": "Инголаб",
                        "description": "Инголаб expert.",
                        "module": "agents.ingos_product_agent",
                        "is_active": False,
                        "params": {
                            "provider": "openai",
                            "product": "Инголаб",
                            "use_platform_store": False,
                        },
                    },
                    {
                        "id": "simple_agent",
                        "name": "Simple",
                        "description": "Other agent.",
                        "module": "agents.simple_agent",
                        "is_active": True,
                        "params": {
                            "provider": "openai",
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = cli._load_product_agent_specs(str(config_path))

    assert [spec.id for spec in specs] == ["product_Car", "product_Инголаб"]
    assert specs[0].provider == ModelType.GPT
    assert specs[0].init_params == {
        "product": "Car",
        "use_platform_store": False,
        "role": "default",
    }
    assert specs[1].product == "Инголаб"
    assert specs[1].is_active is False


def test_find_product_spec_matches_agent_id_and_product_name():
    specs = [
        cli.ProductAgentSpec(
            id="product_Car",
            name="Car",
            description="Car expert.",
            product="Car",
            module_path="agents.ingos_product_agent",
            provider=ModelType.GPT,
            is_active=True,
            init_params={"product": "Car"},
        ),
        cli.ProductAgentSpec(
            id="product_Инголаб",
            name="Инголаб",
            description="Инголаб expert.",
            product="Инголаб",
            module_path="agents.ingos_product_agent",
            provider=ModelType.GPT,
            is_active=False,
            init_params={"product": "Инголаб"},
        ),
    ]

    assert cli._find_product_spec(specs, agent_id="product_Car").id == "product_Car"
    assert cli._find_product_spec(specs, product_name="инголаб").id == "product_Инголаб"


def test_choose_product_spec_retries_until_valid(monkeypatch, capsys):
    prompts = iter(["bad", "7", "2"])
    specs = [
        cli.ProductAgentSpec(
            id="product_Car",
            name="Car",
            description="Car expert.",
            product="Car",
            module_path="agents.ingos_product_agent",
            provider=ModelType.GPT,
            is_active=True,
            init_params={"product": "Car"},
        ),
        cli.ProductAgentSpec(
            id="product_Household",
            name="Household",
            description="Household expert.",
            product="Household",
            module_path="agents.ingos_product_agent",
            provider=ModelType.GPT,
            is_active=False,
            init_params={"product": "Household"},
        ),
    ]

    monkeypatch.setattr("builtins.input", lambda _prompt="": next(prompts))

    selected = cli._choose_product_spec(specs)

    assert selected.id == "product_Household"
    captured = capsys.readouterr()
    assert "1. Car (product_Car)" in captured.out
    assert "2. Household (product_Household) [inactive in service config]" in captured.out
    assert captured.out.count("Enter a number from 1 to 2.") == 2


def test_initialize_agent_for_spec_filters_unaccepted_params(monkeypatch):
    captured: dict[str, object] = {}

    def fake_initialize_agent(provider, product, use_platform_store=False, prefetch_top_k=3):
        captured["provider"] = provider
        captured["product"] = product
        captured["use_platform_store"] = use_platform_store
        captured["prefetch_top_k"] = prefetch_top_k
        return "fake-agent"

    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda _module_path: SimpleNamespace(initialize_agent=fake_initialize_agent),
    )

    spec = cli.ProductAgentSpec(
        id="product_Car",
        name="Car",
        description="Car expert.",
        product="Car",
        module_path="agents.ingos_product_agent",
        provider=ModelType.GPT,
        is_active=True,
        init_params={
            "product": "Car",
            "use_platform_store": False,
            "role": "default",
        },
    )

    agent = cli._initialize_agent_for_spec(
        spec,
        provider=ModelType.GPT4,
        prefetch_top_k=5,
    )

    assert agent == "fake-agent"
    assert captured == {
        "provider": ModelType.GPT4,
        "product": "Car",
        "use_platform_store": False,
        "prefetch_top_k": 5,
    }


def test_choose_product_spec_raises_keyboard_interrupt_on_exit(monkeypatch):
    specs = [
        cli.ProductAgentSpec(
            id="product_Car",
            name="Car",
            description="Car expert.",
            product="Car",
            module_path="agents.ingos_product_agent",
            provider=ModelType.GPT,
            is_active=True,
            init_params={"product": "Car"},
        )
    ]

    monkeypatch.setattr("builtins.input", lambda _prompt="": "/exit")

    with pytest.raises(KeyboardInterrupt):
        cli._choose_product_spec(specs)
