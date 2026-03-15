from __future__ import annotations

import asyncio
from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import AsyncExitStack
import json
import os
from pathlib import PurePosixPath
import secrets
import threading
import time
from typing import Any, Dict, List

import httpx
import mcp.types as mcp_types
from mcp.shared._httpx_utils import MCP_DEFAULT_SSE_READ_TIMEOUT, MCP_DEFAULT_TIMEOUT
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.callbacks import CallbackContext, Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, ToolCallInterceptor
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.config import get_stream_writer
from langgraph.types import interrupt


SYSADMIN_MCP_URL = os.environ.get("SYSADMIN_MCP_URL", "http://127.0.0.1:8000/mcp")
SYSADMIN_MCP_SCOPE = os.environ.get("SYSADMIN_MCP_OAUTH_SCOPE", "sysadmin:mcp")
_APPROVE_ANSWERS = {"y", "yes", "approve", "approved", "ok", "okay", "continue"}
_DECLINE_ANSWERS = {"n", "no", "decline", "cancel", "stop", "abort"}
_PASSWORD_SECRET_STORE: Dict[str, str] = {}
_PASSWORD_SECRET_STORE_LOCK = threading.Lock()
_TOKEN_EXPIRY_SKEW_SECONDS = 60.0


def _safe_stream_writer():
    try:
        return get_stream_writer()
    except Exception:
        return lambda *_args, **_kwargs: None


def _resolve_token_endpoint() -> str | None:
    token_endpoint = os.environ.get("SYSADMIN_MCP_OAUTH_TOKEN_ENDPOINT")
    if token_endpoint:
        return token_endpoint
    issuer_url = os.environ.get("SYSADMIN_MCP_OAUTH_ISSUER_URL")
    if issuer_url:
        return issuer_url.rstrip("/") + "/protocol/openid-connect/token"
    return None


async def get_access_token() -> str:
    access_token = os.environ.get("SYSADMIN_MCP_TOKEN") or os.environ.get("SYSADMIN_MCP_BEARER_TOKEN")
    if access_token:
        return access_token

    client_id = os.environ.get("SYSADMIN_MCP_OAUTH_CLIENT_ID")
    client_secret = os.environ.get("SYSADMIN_MCP_OAUTH_CLIENT_SECRET")
    token_endpoint = _resolve_token_endpoint()
    if not client_id or not client_secret or not token_endpoint:
        raise RuntimeError(
            "Sysadmin MCP auth is not configured. Set SYSADMIN_MCP_TOKEN or "
            "SYSADMIN_MCP_OAUTH_CLIENT_ID, SYSADMIN_MCP_OAUTH_CLIENT_SECRET, and "
            "SYSADMIN_MCP_OAUTH_TOKEN_ENDPOINT/SYSADMIN_MCP_OAUTH_ISSUER_URL."
        )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            token_endpoint,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": SYSADMIN_MCP_SCOPE,
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()

    payload = response.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise RuntimeError("Sysadmin MCP OAuth response did not include access_token.")
    return str(access_token)


class _AccessTokenProvider:
    def __init__(self, session_context: MutableMapping[str, Any]) -> None:
        self._session_context = session_context
        self._lock = asyncio.Lock()
        self._static_token = os.environ.get("SYSADMIN_MCP_TOKEN") or os.environ.get("SYSADMIN_MCP_BEARER_TOKEN")
        self._access_token = str(self._static_token or "").strip()
        self._expires_at = float("inf") if self._access_token else 0.0

    @property
    def refreshable(self) -> bool:
        return not bool(self._static_token)

    async def get_valid_token(self, *, force_refresh: bool = False) -> str:
        if not self.refreshable:
            token = str(self._static_token or "").strip()
            if not token:
                raise RuntimeError(
                    "Sysadmin MCP auth is not configured. Set SYSADMIN_MCP_TOKEN or "
                    "SYSADMIN_MCP_OAUTH_CLIENT_ID, SYSADMIN_MCP_OAUTH_CLIENT_SECRET, and "
                    "SYSADMIN_MCP_OAUTH_TOKEN_ENDPOINT/SYSADMIN_MCP_OAUTH_ISSUER_URL."
                )
            _set_mcp_debug_auth(self._session_context, token)
            return token

        now = time.monotonic()
        if (
            not force_refresh
            and self._access_token
            and now < (self._expires_at - _TOKEN_EXPIRY_SKEW_SECONDS)
        ):
            return self._access_token

        async with self._lock:
            now = time.monotonic()
            if (
                not force_refresh
                and self._access_token
                and now < (self._expires_at - _TOKEN_EXPIRY_SKEW_SECONDS)
            ):
                return self._access_token
            return await self._refresh_locked()

    async def _refresh_locked(self) -> str:
        client_id = os.environ.get("SYSADMIN_MCP_OAUTH_CLIENT_ID")
        client_secret = os.environ.get("SYSADMIN_MCP_OAUTH_CLIENT_SECRET")
        token_endpoint = _resolve_token_endpoint()
        if not client_id or not client_secret or not token_endpoint:
            raise RuntimeError(
                "Sysadmin MCP auth is not configured. Set SYSADMIN_MCP_TOKEN or "
                "SYSADMIN_MCP_OAUTH_CLIENT_ID, SYSADMIN_MCP_OAUTH_CLIENT_SECRET, and "
                "SYSADMIN_MCP_OAUTH_TOKEN_ENDPOINT/SYSADMIN_MCP_OAUTH_ISSUER_URL."
            )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                token_endpoint,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": SYSADMIN_MCP_SCOPE,
                },
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()

        payload = response.json()
        access_token = str(payload.get("access_token") or "").strip()
        if not access_token:
            raise RuntimeError("Sysadmin MCP OAuth response did not include access_token.")

        expires_in = payload.get("expires_in")
        try:
            expires_in_seconds = float(expires_in)
        except Exception:
            expires_in_seconds = MCP_DEFAULT_TIMEOUT
        self._access_token = access_token
        self._expires_at = time.monotonic() + max(expires_in_seconds, 1.0)
        _set_mcp_debug_auth(self._session_context, access_token)
        return access_token


class _RefreshingBearerAuth(httpx.Auth):
    def __init__(
        self,
        token_provider: _AccessTokenProvider,
        session_context: MutableMapping[str, Any],
    ) -> None:
        self._token_provider = token_provider
        self._session_context = session_context

    async def async_auth_flow(self, request: httpx.Request):
        token = await self._token_provider.get_valid_token()
        authorization = f"Bearer {token}"
        request.headers["Authorization"] = authorization
        self._session_context["_mcp_auth_retry_attempted"] = False
        response = yield request

        if response.status_code != 401:
            return

        if not self._token_provider.refreshable:
            return

        refreshed_token = await self._token_provider.get_valid_token(force_refresh=True)
        refreshed_authorization = f"Bearer {refreshed_token}"
        self._session_context["_mcp_auth_retry_attempted"] = True

        retry_debug = {
            "method": request.method,
            "url": str(request.url),
            "authorization": refreshed_authorization,
            "content": _request_content_text(request),
        }
        print("[sysadmin-agent][debug] Retrying MCP HTTP request with refreshed token:")
        try:
            print(json.dumps(retry_debug, ensure_ascii=False, indent=2, default=str))
        except Exception:
            print(str(retry_debug))

        request.headers["Authorization"] = refreshed_authorization
        yield request


def _is_approved_response(value: Any) -> bool:
    return str(value or "").strip().lower() in _APPROVE_ANSWERS


def _is_declined_response(value: Any) -> bool:
    return str(value or "").strip().lower() in _DECLINE_ANSWERS


def merge_state_into_mcp_context(
    state: Mapping[str, Any],
    session_context: MutableMapping[str, Any],
) -> None:
    for field_name in ("server", "target_id", "working_dir", "execution_id"):
        value = state.get(field_name)
        if value:
            session_context[field_name] = value
    allowed_paths = state.get("allowed_paths")
    if isinstance(allowed_paths, list):
        session_context["allowed_paths"] = [str(path) for path in allowed_paths if path]

    password_secret_id = str(state.get("password_secret_id") or "").strip()
    if password_secret_id:
        cached_password = _load_password_secret(password_secret_id)
        if cached_password:
            session_context["password_secret_id"] = password_secret_id
            session_context["accepted_password"] = cached_password
            return

    session_context.pop("password_secret_id", None)
    session_context.pop("accepted_password", None)


def apply_mcp_context_to_state(
    state: Mapping[str, Any],
    session_context: Mapping[str, Any],
) -> Dict[str, Any]:
    updated = dict(state)
    for field_name in ("server", "target_id", "working_dir", "execution_id"):
        value = session_context.get(field_name)
        if value:
            updated[field_name] = value
    allowed_paths = session_context.get("allowed_paths")
    if isinstance(allowed_paths, list):
        updated["allowed_paths"] = [str(path) for path in allowed_paths if path]
    else:
        updated.pop("allowed_paths", None)
    password_secret_id = str(session_context.get("password_secret_id") or "").strip()
    if password_secret_id and _load_password_secret(password_secret_id):
        updated["password_secret_id"] = password_secret_id
    else:
        updated.pop("password_secret_id", None)
    return updated


def _load_password_secret(secret_id: str) -> str | None:
    with _PASSWORD_SECRET_STORE_LOCK:
        return _PASSWORD_SECRET_STORE.get(secret_id)


def _store_password_secret(
    password: str,
    session_context: MutableMapping[str, Any],
) -> str:
    secret_id = str(session_context.get("password_secret_id") or "").strip()
    if not secret_id:
        secret_id = secrets.token_urlsafe(24)
    with _PASSWORD_SECRET_STORE_LOCK:
        _PASSWORD_SECRET_STORE[secret_id] = password
    session_context["password_secret_id"] = secret_id
    session_context["accepted_password"] = password
    return secret_id


def _set_mcp_debug_auth(
    session_context: MutableMapping[str, Any],
    access_token: str,
) -> None:
    authorization = f"Bearer {access_token}"
    session_context["_mcp_debug_auth"] = {
        "url": SYSADMIN_MCP_URL,
        "authorization": authorization,
    }
    session_context["_last_mcp_request"] = {
        "phase": "session_open",
        "url": SYSADMIN_MCP_URL,
        "authorization": authorization,
    }


def _record_mcp_request(
    session_context: MutableMapping[str, Any],
    request: MCPToolCallRequest,
) -> None:
    debug_auth = session_context.get("_mcp_debug_auth")
    request_debug: Dict[str, Any] = {
        "phase": "tool_call",
        "url": SYSADMIN_MCP_URL,
        "server_name": request.server_name,
        "tool_name": request.name,
        "args": dict(request.args or {}),
    }
    if isinstance(debug_auth, Mapping):
        authorization = debug_auth.get("authorization")
        if authorization:
            request_debug["authorization"] = authorization
    session_context["_last_mcp_request"] = request_debug


def _request_content_text(request: httpx.Request) -> str | None:
    try:
        content = request.content
    except Exception:
        return None
    if not content:
        return None
    if isinstance(content, bytes):
        try:
            return content.decode("utf-8")
        except Exception:
            return repr(content)
    return str(content)


def _build_debug_httpx_client_factory(
    session_context: MutableMapping[str, Any],
):
    async def on_response(response: httpx.Response) -> None:
        request = response.request
        http_debug = {
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "authorization": request.headers.get("Authorization"),
            "content": _request_content_text(request),
        }
        session_context["_last_mcp_http_status"] = response.status_code
        session_context["_last_mcp_http_request"] = http_debug

        if response.status_code == 401:
            print("[sysadmin-agent][debug] MCP returned HTTP 401")
            print("[sysadmin-agent][debug] MCP HTTP request parameters:")
            try:
                print(json.dumps(http_debug, ensure_ascii=False, indent=2, default=str))
            except Exception:
                print(str(http_debug))

            logical_request = session_context.get("_last_mcp_request")
            if logical_request is not None:
                print("[sysadmin-agent][debug] MCP logical request parameters:")
                try:
                    print(json.dumps(logical_request, ensure_ascii=False, indent=2, default=str))
                except Exception:
                    print(str(logical_request))

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        effective_timeout = timeout or httpx.Timeout(
            MCP_DEFAULT_TIMEOUT,
            read=MCP_DEFAULT_SSE_READ_TIMEOUT,
        )
        kwargs: Dict[str, Any] = {
            "follow_redirects": True,
            "timeout": effective_timeout,
            "event_hooks": {"response": [on_response]},
        }
        if headers is not None:
            kwargs["headers"] = headers
        if auth is not None:
            kwargs["auth"] = auth
        return httpx.AsyncClient(**kwargs)

    return factory


def _normalize_server_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _select_target_from_result(
    targets: Sequence[Mapping[str, Any]],
    desired_server: str | None,
) -> Mapping[str, Any] | None:
    normalized_server = _normalize_server_name(desired_server)
    if normalized_server:
        for target in targets:
            if _normalize_server_name(target.get("target_id")) == normalized_server:
                return target
            if _normalize_server_name(target.get("ssh_alias")) == normalized_server:
                return target
    if len(targets) == 1:
        return targets[0]
    return None


def _path_within_allowed_base(path: str, allowed_base: str) -> bool:
    normalized_path = PurePosixPath(path)
    normalized_base = PurePosixPath(allowed_base)
    try:
        normalized_path.relative_to(normalized_base)
        return True
    except ValueError:
        return False


def _is_allowed_remote_path(path: str | None, allowed_paths: Sequence[str] | None) -> bool:
    if not path:
        return False
    if not allowed_paths:
        return True
    return any(
        _path_within_allowed_base(path, allowed_base)
        for allowed_base in allowed_paths
        if allowed_base
    )


def _disallowed_path_result(
    *,
    field_name: str,
    path: str,
    allowed_paths: Sequence[str],
) -> mcp_types.CallToolResult:
    allowed_text = ", ".join(allowed_paths) if allowed_paths else "(no allowlist reported)"
    return _plain_text_result(
        f"{field_name} is outside the allowlist: {path}. "
        f"Allowed paths: {allowed_text}. "
        "Choose one of the allowed paths for browse_files/read_file, or inspect the path via run_command instead."
    )


def _parent_directory(path: str | None) -> str | None:
    if not path:
        return None
    parent = str(PurePosixPath(path).parent)
    return parent if parent and parent != "." else None


def _update_mcp_context_from_result(
    request_name: str,
    request_args: Mapping[str, Any],
    result: Any,
    session_context: MutableMapping[str, Any],
) -> None:
    target_id = request_args.get("target_id")
    if target_id:
        session_context["target_id"] = target_id

    if request_name == "run_command":
        working_dir = request_args.get("working_dir")
        if working_dir:
            session_context["working_dir"] = working_dir
    elif request_name == "read_file":
        parent = _parent_directory(request_args.get("path"))
        if parent:
            session_context["working_dir"] = parent

    structured = getattr(result, "structuredContent", None)
    if not isinstance(structured, dict):
        return

    if request_name == "list_targets":
        targets = structured.get("targets")
        if isinstance(targets, list):
            selected_target = _select_target_from_result(
                [item for item in targets if isinstance(item, Mapping)],
                str(session_context.get("server") or ""),
            )
            if selected_target:
                selected_target_id = selected_target.get("target_id")
                if selected_target_id:
                    session_context["target_id"] = selected_target_id
                ssh_alias = selected_target.get("ssh_alias")
                if ssh_alias:
                    session_context["server"] = ssh_alias
                allowed_paths = selected_target.get("allowed_paths")
                if isinstance(allowed_paths, list):
                    session_context["allowed_paths"] = [str(path) for path in allowed_paths if path]
        return

    if request_name == "browse_files":
        summary = structured.get("summary")
        if isinstance(summary, Mapping):
            if summary.get("target_id"):
                session_context["target_id"] = summary["target_id"]
            if summary.get("directory"):
                session_context["working_dir"] = summary["directory"]
        return

    if request_name == "run_command":
        summary = structured.get("summary")
        if isinstance(summary, Mapping) and summary.get("target_id"):
            session_context["target_id"] = summary["target_id"]
        if structured.get("execution_id"):
            session_context["execution_id"] = structured["execution_id"]
        if structured.get("current_directory"):
            session_context["working_dir"] = structured["current_directory"]
        return

    if request_name == "read_command_output" and request_args.get("execution_id"):
        session_context["execution_id"] = request_args["execution_id"]


def _plain_text_result(text: str) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=text)],
        isError=False,
    )


def _call_tool_result_text(result: mcp_types.CallToolResult) -> str:
    parts: List[str] = []
    for item in result.content:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def _is_call_tool_error_result(result: Any) -> bool:
    return bool(getattr(result, "isError", False)) and hasattr(result, "content")


def _recoverable_tool_error_result(
    request_name: str,
    result: Any,
) -> mcp_types.CallToolResult:
    error_text = _call_tool_result_text(result) or f"{request_name} failed."

    if "Remote path does not exist" in error_text:
        return _plain_text_result(
            f"{error_text}\n"
            "Choose another existing path, or use run_command to search the filesystem."
        )

    if "outside the allowlist" in error_text:
        return _plain_text_result(
            f"{error_text}\n"
            "Use one of the allowed paths for browse_files/read_file, or inspect the path via run_command instead."
        )

    return _plain_text_result(error_text)


def _build_run_command_approval_payload(
    request: MCPToolCallRequest,
) -> Dict[str, Any]:
    args = request.args or {}
    details: List[str] = ["Approve remote command execution."]

    if args.get("target_id"):
        details.append(f"Target: {args['target_id']}")
    timeout_seconds = args.get("timeout_seconds") or 300
    details.append(f"Timeout: {timeout_seconds}s")
    if args.get("working_dir"):
        details.append(f"Working directory: {args['working_dir']}")
    details.append("Command:")
    details.append(str(args.get("command") or ""))
    details.append("Reply `yes` to approve or `no` to cancel.")
    prompt = "\n".join(details)

    return {
        "type": "approval",
        "question": prompt,
        "content": prompt,
    }


def _requested_schema(params: mcp_types.ElicitRequestParams) -> Mapping[str, Any]:
    requested_schema = getattr(params, "requestedSchema", {}) or {}
    return requested_schema if isinstance(requested_schema, Mapping) else {}


def _requested_properties(params: mcp_types.ElicitRequestParams) -> Dict[str, Mapping[str, Any]]:
    properties = _requested_schema(params).get("properties", {})
    if not isinstance(properties, Mapping):
        return {}
    return {
        str(field_name): field_schema
        for field_name, field_schema in properties.items()
        if isinstance(field_schema, Mapping)
    }


def _required_fields(params: mcp_types.ElicitRequestParams) -> List[str]:
    required = _requested_schema(params).get("required", [])
    if not isinstance(required, list):
        return []
    return [str(field_name) for field_name in required]


def _response_matches_schema(
    response_content: Mapping[str, Any],
    params: mcp_types.ElicitRequestParams,
) -> bool:
    required_fields = _required_fields(params)
    return all(field_name in response_content for field_name in required_fields)


def _is_secret_field(
    field_name: str,
    field_schema: Mapping[str, Any],
) -> bool:
    field_format = str(field_schema.get("format") or "").strip().lower()
    if field_format == "password":
        return True

    hint_parts = [
        field_name,
        str(field_schema.get("title") or ""),
        str(field_schema.get("description") or ""),
    ]
    hint_text = " ".join(hint_parts).lower()
    return any(token in hint_text for token in ("password", "passphrase", "secret"))


def _build_elicitation_fields(
    params: mcp_types.ElicitRequestParams,
) -> List[Dict[str, Any]]:
    required_fields = set(_required_fields(params))
    fields: List[Dict[str, Any]] = []
    for field_name, field_schema in _requested_properties(params).items():
        fields.append(
            {
                "name": field_name,
                "title": field_schema.get("title") or field_name,
                "description": field_schema.get("description"),
                "required": field_name in required_fields,
                "type": field_schema.get("type"),
                "format": field_schema.get("format"),
                "writeOnly": field_schema.get("writeOnly"),
                "secret": _is_secret_field(field_name, field_schema),
            }
        )
    return fields


def _build_elicitation_response_content(
    params: mcp_types.ElicitRequestParams,
    user_response: Any,
) -> Dict[str, Any] | None:
    properties = _requested_properties(params)
    if not properties:
        return None

    non_approval_fields = [field_name for field_name in properties if field_name != "approve"]
    response_content: Dict[str, Any] = {}

    if isinstance(user_response, Mapping):
        if "approve" in properties and "approve" in user_response:
            response_content["approve"] = bool(user_response["approve"])
        elif "approve" in properties:
            response_content["approve"] = True

        for field_name in non_approval_fields:
            if field_name in user_response:
                response_content[field_name] = user_response[field_name]

        if len(non_approval_fields) == 1 and non_approval_fields[0] not in response_content:
            value = user_response.get("value")
            if value is None:
                value = user_response.get("text")
            if value is not None:
                response_content[non_approval_fields[0]] = value

        if response_content and _response_matches_schema(response_content, params):
            return response_content

        response_text = str(user_response.get("value") or user_response.get("text") or "").strip()
    else:
        response_text = str(user_response or "").strip()

    if not response_text or _is_declined_response(response_text):
        return None

    if "approve" in properties:
        response_content["approve"] = True
        if not non_approval_fields:
            return response_content if _is_approved_response(response_text) else None

    if len(non_approval_fields) == 1:
        response_content[non_approval_fields[0]] = response_text
        return response_content

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, Mapping):
        return None

    for field_name in non_approval_fields:
        if field_name in parsed:
            response_content[field_name] = parsed[field_name]

    return response_content if _response_matches_schema(response_content, params) else None


def _extract_password_from_response_content(
    params: mcp_types.ElicitRequestParams,
    response_content: Mapping[str, Any],
) -> str | None:
    for field_name, field_schema in _requested_properties(params).items():
        if not _is_secret_field(field_name, field_schema):
            continue
        if str(field_schema.get("format") or "").strip().lower() != "password" and field_name != "password":
            continue
        password = response_content.get(field_name)
        if password is not None:
            return str(password)
    return None


def _pending_password_requested(pending_elicitation: Any) -> bool:
    if not isinstance(pending_elicitation, Mapping):
        return False
    params = pending_elicitation.get("params")
    if not params:
        return False
    for field_name, field_schema in _requested_properties(params).items():
        if _is_secret_field(field_name, field_schema):
            if str(field_schema.get("format") or "").strip().lower() == "password" or field_name == "password":
                return True
    return False


def _build_elicitation_payload(
    params: mcp_types.ElicitRequestParams,
) -> Dict[str, Any]:
    if isinstance(params, mcp_types.ElicitRequestURLParams):
        prompt = (
            "Open the URL below and reply `yes` to continue or `no` to cancel.\n\n"
            f"{params.url}"
        )
        return {
            "type": "approval",
            "question": prompt,
            "content": prompt,
        }

    requested_schema = dict(_requested_schema(params))
    properties = _requested_properties(params)
    fields = _build_elicitation_fields(params)
    details: List[str] = [params.message]
    non_approval_fields = [field_name for field_name in properties if field_name != "approve"]

    if len(non_approval_fields) == 1 and non_approval_fields[0] == "confirmation_token":
        details.append("Reply with the requested confirmation token to continue, or `cancel`.")
    elif len(non_approval_fields) == 1 and non_approval_fields[0] == "password":
        details.append("Reply with the requested password to continue, or `cancel`.")
    elif len(non_approval_fields) == 1:
        details.append(f"Reply with the value for `{non_approval_fields[0]}` to continue, or `cancel`.")
    elif "approve" in properties and not non_approval_fields:
        details.append("Reply `yes` to approve or `no` to cancel.")
    else:
        details.append("Reply with a JSON object that matches the requested schema, or `cancel`.")

    for field_name, field_schema in properties.items():
        description = field_schema.get("description")
        if description:
            details.append(f"{field_name}: {description}")
    prompt = "\n".join(details)

    return {
        "type": "approval" if "approve" in properties and not non_approval_fields else "elicitation",
        "question": prompt,
        "content": prompt,
        "message": params.message,
        "requestedSchema": requested_schema,
        "fields": fields,
        "responseMode": "text" if len(non_approval_fields) == 1 else "json",
    }


class SysAdminContextInterceptor:
    def __init__(self, session_context: MutableMapping[str, Any]) -> None:
        self._session_context = session_context

    async def _run_command_with_hitl(
        self,
        request: MCPToolCallRequest,
        handler: Any,
    ) -> Any:
        approval_response = interrupt(_build_run_command_approval_payload(request))
        if _is_declined_response(approval_response):
            return _plain_text_result("Command execution cancelled by user.")
        if not _is_approved_response(approval_response):
            return _plain_text_result("Command execution cancelled because approval was not granted.")

        attempts = 0
        while True:
            self._session_context["_auto_approve_run_command"] = True
            try:
                result = await handler(request)
            finally:
                self._session_context.pop("_auto_approve_run_command", None)
                self._session_context.pop("_elicitation_response", None)

            pending_elicitation = self._session_context.pop("_pending_elicitation", None)
            candidate_password = str(self._session_context.pop("_candidate_password", "") or "").strip()
            if candidate_password and not _pending_password_requested(pending_elicitation):
                _store_password_secret(candidate_password, self._session_context)

            if not pending_elicitation or not getattr(result, "isError", False):
                return result

            attempts += 1
            if attempts >= 5:
                return result

            user_response = interrupt(pending_elicitation["payload"])
            response_content = _build_elicitation_response_content(
                pending_elicitation["params"],
                user_response,
            )
            if response_content is None:
                return _plain_text_result("Command execution cancelled by user.")
            password = _extract_password_from_response_content(
                pending_elicitation["params"],
                response_content,
            )
            if password:
                self._session_context["_candidate_password"] = password
            self._session_context["_elicitation_response"] = response_content

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Any,
    ) -> Any:
        updated_args = dict(request.args or {})

        if request.name in {"browse_files", "read_file", "run_command"}:
            if not updated_args.get("target_id") and self._session_context.get("target_id"):
                updated_args["target_id"] = self._session_context["target_id"]

        if request.name == "run_command":
            if not updated_args.get("working_dir") and self._session_context.get("working_dir"):
                updated_args["working_dir"] = self._session_context["working_dir"]

        if request.name == "read_command_output":
            if not updated_args.get("execution_id") and self._session_context.get("execution_id"):
                updated_args["execution_id"] = self._session_context["execution_id"]

        request = request.override(args=updated_args)
        _record_mcp_request(self._session_context, request)

        allowed_paths = self._session_context.get("allowed_paths")
        normalized_allowed_paths = (
            [str(path) for path in allowed_paths if path]
            if isinstance(allowed_paths, list)
            else []
        )

        if request.name in {"browse_files", "read_file", "run_command"} and not updated_args.get("target_id"):
            return _plain_text_result(
                "target_id is required. Call list_targets first and select a target before using this tool."
            )

        if request.name == "browse_files":
            directory = str(updated_args.get("directory") or "").strip()
            if directory and not _is_allowed_remote_path(directory, normalized_allowed_paths):
                return _disallowed_path_result(
                    field_name="directory",
                    path=directory,
                    allowed_paths=normalized_allowed_paths,
                )

        if request.name == "read_file":
            path = str(updated_args.get("path") or "").strip()
            if path and not _is_allowed_remote_path(path, normalized_allowed_paths):
                return _disallowed_path_result(
                    field_name="path",
                    path=path,
                    allowed_paths=normalized_allowed_paths,
                )

        if request.name == "read_command_output" and not updated_args.get("execution_id"):
            return _plain_text_result(
                "execution_id is required. Run a command first or provide execution_id explicitly."
            )

        if request.name == "run_command":
            result = await self._run_command_with_hitl(request, handler)
        else:
            result = await handler(request)

        if _is_call_tool_error_result(result):
            result = _recoverable_tool_error_result(request.name, result)

        _update_mcp_context_from_result(request.name, request.args, result, self._session_context)
        return result


async def _on_mcp_progress(
    progress: float,
    total: float | None,
    message: str | None,
    context: CallbackContext,
) -> None:
    writer = _safe_stream_writer()
    payload: Dict[str, Any] = {
        "type": "tool_progress",
        "server": context.server_name,
        "tool": context.tool_name,
        "progress": progress,
    }
    if total is not None:
        payload["total"] = total
    if message:
        payload["message"] = message
    writer(payload)


async def _on_mcp_elicitation(
    _mcp_context: Any,
    params: mcp_types.ElicitRequestParams,
    session_context: MutableMapping[str, Any],
) -> mcp_types.ElicitResult:
    if isinstance(params, mcp_types.ElicitRequestURLParams):
        return mcp_types.ElicitResult(action="decline")

    properties = _requested_properties(params)
    non_approval_fields = [field_name for field_name in properties if field_name != "approve"]

    if "approve" in properties and not non_approval_fields:
        if session_context.pop("_auto_approve_run_command", False):
            return mcp_types.ElicitResult(action="accept", content={"approve": True})
        return mcp_types.ElicitResult(action="decline")

    stored_response = session_context.pop("_elicitation_response", None)
    if isinstance(stored_response, Mapping) and _response_matches_schema(stored_response, params):
        return mcp_types.ElicitResult(action="accept", content=dict(stored_response))

    accepted_password = str(session_context.get("accepted_password") or "").strip()
    if accepted_password:
        for field_name, field_schema in properties.items():
            if not _is_secret_field(field_name, field_schema):
                continue
            if str(field_schema.get("format") or "").strip().lower() != "password" and field_name != "password":
                continue
            content: Dict[str, Any] = {field_name: accepted_password}
            if "approve" in properties:
                content["approve"] = True
            return mcp_types.ElicitResult(action="accept", content=content)

    if "confirmation_token" in properties:
        confirmation_token = str(os.environ.get("SYSADMIN_MCP_CONFIRMATION_TOKEN") or "").strip()
        if confirmation_token:
            content: Dict[str, Any] = {"confirmation_token": confirmation_token}
            if "approve" in properties:
                content["approve"] = True
            return mcp_types.ElicitResult(action="accept", content=content)

    if properties:
        session_context["_pending_elicitation"] = {
            "params": params,
            "payload": _build_elicitation_payload(params),
        }
        return mcp_types.ElicitResult(action="cancel")

    return mcp_types.ElicitResult(action="cancel")


def build_mcp_callbacks(session_context: MutableMapping[str, Any]) -> Callbacks:
    async def on_elicitation(
        mcp_context: Any,
        params: mcp_types.ElicitRequestParams,
        _context: CallbackContext,
    ) -> mcp_types.ElicitResult:
        return await _on_mcp_elicitation(mcp_context, params, session_context)

    return Callbacks(
        on_progress=_on_mcp_progress,
        on_elicitation=on_elicitation,
    )


async def create_mcp_session_agent(
    *,
    model: BaseChatModel,
    middleware: List[Any],
    state_schema: Any,
    context_schema: Any,
    session_context: MutableMapping[str, Any],
) -> tuple[AsyncExitStack, Any]:
    token_provider = _AccessTokenProvider(session_context)
    await token_provider.get_valid_token()
    callbacks = build_mcp_callbacks(session_context)
    tool_interceptors: List[ToolCallInterceptor] = [SysAdminContextInterceptor(session_context)]
    auth = _RefreshingBearerAuth(token_provider, session_context)

    client = MultiServerMCPClient(
        {
            "sysadmin": {
                "transport": "http",
                "url": SYSADMIN_MCP_URL,
                "headers": {},
                "auth": auth,
                "httpx_client_factory": _build_debug_httpx_client_factory(session_context),
            }
        },
        callbacks=callbacks,
        tool_interceptors=tool_interceptors,
    )

    stack = AsyncExitStack()
    session = await stack.enter_async_context(client.session("sysadmin"))
    tools = await load_mcp_tools(
        session,
        callbacks=callbacks,
        server_name="sysadmin",
        tool_interceptors=tool_interceptors,
    )
    agent = create_agent(
        model=model,
        tools=tools,
        middleware=middleware,
        state_schema=state_schema,
        context_schema=context_schema,
    )
    return stack, agent
