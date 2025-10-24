from __future__ import annotations
from typing import Any, Dict, List, Callable, Union, Iterable
from copy import copy
from pydantic import BaseModel

from langchain.agents import create_agent
#from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.simple import Tool  # concrete Tool implementation
from langchain_core.runnables import RunnableConfig


# ---------- helpers: deep map over common JSON-ish shapes ----------

JSONLike = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]

def _map_strings(obj: JSONLike, fn: Callable[[str], str]) -> JSONLike:
    """Recursively apply fn to every string in nested dict/list structures."""
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _map_strings(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_strings(v, fn) for v in obj]
    return obj  # ints, floats, bools, None left as-is


def _anonymize_message_content(content: Any, anonymize: Callable[[str], str]) -> Any:
    """
    LangChain message content may be:
      - plain str
      - list of content parts: [{"type":"text","text":"..."}, {"type":"input_text", ...}, ...]
    We only transform text-bearing fields.
    """
    if isinstance(content, str):
        return anonymize(content)

    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                p = dict(part)
                # Common keys that carry text
                for key in ("text", "content", "input", "title", "caption", "markdown"):
                    if isinstance(p.get(key), str):
                        p[key] = anonymize(p[key])
                out.append(p)
            else:
                out.append(part)
        return out

    # any other structure: best effort
    return content


def _transform_messages(messages: List[BaseMessage], transform_text: Callable[[str], str]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for m in messages:
        m2 = copy(m)
        m2.content = _anonymize_message_content(m.content, transform_text)
        out.append(m2)
    return out


# ---------- tool wrapper: de-anonymize args -> call -> anonymize result ----------

def wrap_tool_for_privacy(tool: BaseTool, anonymize: Callable[[str], str], deanonymize: Callable[[str], str]) -> BaseTool:
    """
    Returns a new Tool that:
      - de-anonymizes string arguments before calling the underlying tool
      - anonymizes any string content in the tool's return value before returning it
    The name/description/args_schema are preserved so the model schema doesn't change.
    """

    # preserve important metadata so the model/tool-calling keeps working
    name = getattr(tool, "name", tool.__class__.__name__)
    description = getattr(tool, "description", "") or getattr(tool, "__doc__", "") or ""
    args_schema = getattr(tool, "args_schema", None)
    response_format = getattr(tool, "response_format", "content")
    return_direct = getattr(tool, "return_direct", False)
    metadata = getattr(tool, "metadata", None)

    async def _ainvoke(payload: JSONLike, config: RunnableConfig | None = None) -> Any:
        real_args = _map_strings(payload, deanonymize)
        result = await tool.ainvoke(real_args, config=config)  # call the real tool
        return _map_strings(result, anonymize)

    def _invoke(payload: JSONLike, config: RunnableConfig | None = None) -> Any:
        real_args = _map_strings(payload, deanonymize)
        result = tool.invoke(real_args, config=config)
        return _map_strings(result, anonymize)

    # Build a new Tool around our wrapper functions while preserving schema
    wrapped = Tool.from_function(
        func=_invoke,
        coroutine=_ainvoke,
        name=name,
        description=description,
        args_schema=args_schema,
        return_direct=return_direct,
        metadata=metadata,
    )
    # keep response format consistent if the underlying tool uses content+artifact
    wrapped.response_format = response_format
    return wrapped


def wrap_all_tools(tools: List[BaseTool], anonymize: Callable[[str], str], deanonymize: Callable[[str], str]) -> List[BaseTool]:
    return [wrap_tool_for_privacy(t, anonymize, deanonymize) for t in tools]


# ---------- Middleware: anonymize what the model sees + install wrapped tools ----------

class AnonymizationMiddleware(): #AgentMiddleware):
    """
    - modify_model_request:
        * Replace request.messages with anonymized copies (model sees only anonymized)
        * Replace request.tools with wrapped tools that deanonymize inputs & anonymize outputs
    """

    def __init__(self, anonymizer: Any, *, anonymize_llm_input: bool = True):
        super().__init__()
        self._anon = anonymizer
        self._anonymize_llm_input = anonymize_llm_input

    # This hook edits ONLY the model call (not permanent agent state), perfect for privacy.
    # It runs just before the LLM is invoked, and before tools execute.
    def modify_model_request(self, request, state) -> Any:
        if self._anonymize_llm_input:
            request.messages = _transform_messages(request.messages, self._anon.anonymize)

        # Swap in wrapped tools for THIS model step (schema preserved)
        if request.tools:
            request.tools = wrap_all_tools(
                request.tools, anonymize=self._anon.anonymize, deanonymize=self._anon.deanonymize
            )
        return request

if __name__ == "__main__":
    # ---------- Usage example ----------

    # 1) Define some tools (normal, untrusted I/O)
    from langchain_core.tools import tool

    @tool
    def fetch_profile(email: str) -> dict:
        """Fetches a user profile by email (demo)."""
        # Imagine this hits your DB or SaaS API using the real email.
        return {"email": email, "full_name": "Alice Customer", "notes": "Met at FOSDEM 2024"}

    @tool
    def send_email(to: str, subject: str, body: str) -> str:
        """Sends an email (demo)."""
        # Just echo back what would be sent
        return f"queued: to={to}, subject={subject}, body={body[:120]}"


    # 2) Create your agent with middleware
    # (You can pass either a model string or a BaseChatModel here.)
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini")  # example; use whatever you prefer

    # Your anonymizer instance (must be stateful across the session)
    anonymizer = ...  # e.g., Presidio-based or your custom reversible mapper

    mw = AnonymizationMiddleware(anonymizer)

    agent = create_agent(
        model=llm,
        tools=[fetch_profile, send_email],
        prompt="You are a helpful assistant. Use tools when helpful.",
        middleware=[mw],  # <-- the key line
        # NOTE: when using middleware, you should NOT also pass pre/post model hooks
    )

    # 3) Run it like normal
    from langchain_core.messages import HumanMessage

    res = agent.invoke({"messages": [HumanMessage("Email Alice at alice@example.com: 'Lunch next week?'")]})
    print(res)
    # At this point:
    #  - The LLM saw an anonymized version (e.g., "Email [EMAIL_1] ...")
    #  - The actual tool functions received de-anonymized args (real email, etc.)
    #  - The tool return values were re-anonymized before being stored/returned