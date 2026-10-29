from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 Request model."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str
    method: str
    params: dict[str, Any] | list[Any] | None = None


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 Error model."""

    code: int
    message: str
    data: Any = None


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 Response model."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None = None
    result: Any = None
    error: JsonRpcError | None = None


class JsonRpcNotification(BaseModel):
    """JSON-RPC 2.0 Notification model."""

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] | list[Any] | None = None


def encode_message(msg: BaseModel | dict[str, Any]) -> str:
    """Encode a JSON-RPC message into a newline-delimited JSON string."""
    if isinstance(msg, BaseModel):
        data = msg.model_dump(exclude_none=True)
    else:
        data = msg
    return json.dumps(data) + "\n"


def decode_message(raw: str | bytes) -> dict[str, Any]:
    """Decode a raw line or byte string into a JSON dictionary."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    line = raw.strip()
    if not line:
        raise ValueError("Cannot decode empty message line")
    try:
        parsed: Any = json.loads(line)
        if not isinstance(parsed, dict):
            raise TypeError(f"Expected JSON object, got {type(parsed).__name__}")
        return parsed
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON: {err}") from err


def build_initialize_request(
    req_id: int | str,
    client_name: str = "hermes-agy-acp",
    client_version: str = "1.0.0",
) -> JsonRpcRequest:
    """Build an ACP initialize JSON-RPC request."""
    return JsonRpcRequest(
        id=req_id,
        method="initialize",
        params={
            "protocolVersion": 1,
            "clientInfo": {
                "name": client_name,
                "version": client_version,
            },
            "capabilities": {
                "streaming": True,
            },
        },
    )


def build_session_new_request(
    req_id: int | str,
    cwd: str,
    model: str | None = None,
    mcp_servers: list[Any] | None = None,
) -> JsonRpcRequest:
    """Build an ACP session/new JSON-RPC request."""
    params: dict[str, Any] = {
        "cwd": cwd,
        "mcpServers": mcp_servers if mcp_servers is not None else [],
    }
    if model is not None:
        params["model"] = model
    return JsonRpcRequest(id=req_id, method="session/new", params=params)


def build_session_prompt_request(
    req_id: int | str,
    session_id: str,
    prompt: str | list[dict[str, Any]],
) -> JsonRpcRequest:
    """Build an ACP session/prompt JSON-RPC request."""
    return JsonRpcRequest(
        id=req_id,
        method="session/prompt",
        params={
            "sessionId": session_id,
            "prompt": prompt,
        },
    )


def build_session_cancel_notification(session_id: str) -> JsonRpcNotification:
    """Build an ACP session/cancel notification."""
    return JsonRpcNotification(
        method="session/cancel",
        params={
            "sessionId": session_id,
        },
    )


def build_permission_approval_response(
    req_id: int | str,
    decision: str = "allow",
) -> JsonRpcResponse:
    """Build an auto-approval response for session/request_permission."""
    return JsonRpcResponse(
        id=req_id,
        result={"decision": decision},
    )


AUTH_KEYWORDS = {
    "unauthenticated",
    "oauth",
    "unauthorized",
    "token expired",
    "auth failed",
    "login required",
    "not authenticated",
}


def is_auth_error(error: JsonRpcError | dict[str, Any] | str | Exception) -> bool:
    """Determine if a given error, string, or exception indicates authentication failure."""
    if isinstance(error, JsonRpcError):
        if error.code in (401, -32001):
            return True
        text = f"{error.message} {error.data}".lower()
    elif isinstance(error, dict):
        raw_code = error.get("code")
        err_obj = error.get("error")
        if isinstance(err_obj, dict):
            raw_code = err_obj.get("code", raw_code)
        if raw_code in (401, -32001):
            return True
        text = json.dumps(error).lower()
    elif isinstance(error, Exception):
        text = str(error).lower()
    elif isinstance(error, str):
        text = error.lower()
    else:
        text = str(error).lower()

    return any(keyword in text for keyword in AUTH_KEYWORDS)
