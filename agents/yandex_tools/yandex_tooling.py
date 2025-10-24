import json
import requests
from typing import Sequence, Union, Callable, Any, List, Dict

from langchain_community.chat_models import ChatYandexGPT
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

class ChatYandexGPTWithTools(ChatYandexGPT):
    """
    YandexGPT via REST /completion, with function-calling per Yandex spec,
    support for API-key auth, and full two-step tool invocation with
    tool-output truncation to respect model token limits.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tools_payload: List[Dict[str, Any]] = []
        self._tool_funcs: Dict[str, Callable[..., Any]] = {}

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], BaseTool, Callable]],
    ) -> "ChatYandexGPTWithTools":
        self._tools_payload = []
        self._tool_funcs = {}
        for tool in tools:
            if isinstance(tool, dict):
                spec = tool
            elif isinstance(tool, BaseTool):
                spec = {
                    "name": tool.name,
                    "description": getattr(tool, "description", "") or "",
                    "parameters": tool.args_schema.schema(),
                }
                def make_run(t: BaseTool) -> Callable[..., Any]:
                    def _run(**kwargs: Any) -> Any:
                        if "query" in kwargs:
                            return t.run(kwargs["query"])
                        return t.run(kwargs)
                    return _run
                self._tool_funcs[tool.name] = make_run(tool)
            else:
                o = convert_to_openai_tool(tool)
                spec = {"name": o["name"], "description": o.get("description", ""), "parameters": o["parameters"]}
                self._tool_funcs[o["name"]] = tool
            self._tools_payload.append({"function": spec})
        return self

    def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if getattr(self, "api_key", None):
            key = self.api_key.get_secret_value() if hasattr(self.api_key, "get_secret_value") else str(self.api_key)
            headers = {"Authorization": f"Api-Key {key}", "Content-Type": "application/json"}
        else:
            headers = {"Authorization": f"Bearer {self.iam_token}", "x-folder-id": self.folder_id, "Content-Type": "application/json"}
        resp = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def _generate(
        self,
        messages: List[Any],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any
    ) -> ChatResult:
        def _flatten(c: Any) -> str:
            if isinstance(c, list):
                return "".join(
                    seg.get("text", str(seg)) if isinstance(seg, dict) else str(seg)
                    for seg in c
                )
            return str(c)

        # Prepare & trim history
        body_msgs = []
        for m in messages:
            role = getattr(m, "type", getattr(m, "role", None))
            role = role if role in {"system", "user", "assistant"} else "user"
            body_msgs.append({"role": role, "text": _flatten(m.content)})
        sys_msgs = [m for m in body_msgs if m["role"] == "system"]
        hist = [m for m in body_msgs if m["role"] != "system"]
        max_turns = 50 if "32k" in self.model_uri else 10
        if len(hist) > max_turns:
            hist = hist[-max_turns:]
        body_msgs = sys_msgs + hist

        # First pass: ask for function call
        payload1 = {"modelUri": self.model_uri, "messages": body_msgs, "tools": self._tools_payload}
        try:
            data1 = self._send_request(payload1)
        except requests.HTTPError as e:
            err = getattr(e.response, 'json', lambda: {})()
            msg = err.get('error', {}).get('message', '')
            if 'does not support function calling' in msg:
                data_no = self._send_request({"modelUri": self.model_uri, "messages": body_msgs})
                alt = data_no.get("result", {}).get("alternatives", [])
                text = alt[0].get("message", {}).get("text", "") if alt else ""
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
            raise
        alts1 = data1.get("result", {}).get("alternatives", [])
        if not alts1:
            raise RuntimeError("No alternatives in Yandex response")
        msg1 = alts1[0].get("message", {})
        calls = msg1.get("toolCallList", {}).get("toolCalls", [])
        if not calls:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=msg1.get("text", "")))])

        # Execute tools, truncating output
        call_msg = {"role": "assistant", "toolCallList": {"toolCalls": []}}
        results = []
        # limit tool content length by chars to approx tokens
        limit_chars = 30000 if "32k" in self.model_uri else 8000
        for call in calls:
            fc = call.get("functionCall", {})
            name = fc.get("name")
            args = fc.get("arguments", {}) or {}
            call_msg["toolCallList"]["toolCalls"].append({"functionCall": {"name": name, "arguments": args}})
            func = self._tool_funcs.get(name)
            out = func(**args) if func else None
            res_str = out if isinstance(out, str) else json.dumps(out)
            if len(res_str) > limit_chars:
                res_str = res_str[:limit_chars] + "... (truncated)"
            results.append({"functionResult": {"name": name, "content": res_str}})
        result_msg = {"role": "assistant", "toolResultList": {"toolResults": results}}

        # Second pass: include tool messages + results
        payload2 = {"modelUri": self.model_uri, "tools": self._tools_payload, "messages": body_msgs + [call_msg, result_msg]}
        data2 = self._send_request(payload2)
        alt2 = data2.get("result", {}).get("alternatives", [])
        msg2 = alt2[0].get("message", {}) if alt2 else {}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=msg2.get("text", "")))])
