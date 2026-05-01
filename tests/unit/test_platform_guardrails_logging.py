from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from platform_guardrails.logging import RedactingJSONFileTracer, redact_value


def test_redact_value_redacts_sensitive_keys_and_common_secret_patterns():
    payload = {
        "metadata": {
            "email": "alice@example.com",
            "api_key": "sk-test-secret-value-that-is-very-long",
            "attachment": {
                "filename": "sample.txt",
                "data": "YmFzZTY0LXJhdw==",
            },
        },
        "safe": "visible",
    }

    redacted = redact_value(payload)

    assert redacted["metadata"]["email"] == "<redacted-email>"
    assert redacted["metadata"]["api_key"]["redacted"] is True
    assert redacted["metadata"]["attachment"]["data"]["redacted"] is True
    assert redacted["safe"] == "visible"


def test_redacting_json_file_tracer_does_not_write_raw_prompts_outputs_or_tokens(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    tracer = RedactingJSONFileTracer(str(trace_path))
    response = SimpleNamespace(
        generations=[
            [
                SimpleNamespace(text="Answer for alice@example.com with token secret-token-value-1234567890"),
            ]
        ]
    )

    tracer.on_llm_start(
        {"name": "fake", "kwargs": {"api_key": "sk-test-secret-value-that-is-very-long"}},
        ["Email alice@example.com with phone +7 999 123-45-67"],
    )
    tracer.on_llm_new_token("raw-token-value")
    tracer.on_llm_end(response)
    tracer.on_tool_start({"name": "lookup"}, "alice@example.com")
    tracer.on_tool_end(ToolMessage(content="Alice +7 999 123-45-67", tool_call_id="call-1"))

    text = trace_path.read_text(encoding="utf-8")
    assert "alice@example.com" not in text
    assert "+7 999 123-45-67" not in text
    assert "raw-token-value" not in text
    assert "secret-token-value" not in text
    assert "prompt_chars" in text
    assert "response_chars" in text
    assert "token_chars" in text
    assert "input_chars" in text
