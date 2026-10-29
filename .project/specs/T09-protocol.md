# Specification T09: ACP Protocol Framing & Payloads (protocol.py)

## 1. Overview
The `protocol.py` module defines the JSON-RPC 2.0 wire protocol abstractions, payload builders,
auto-approval permission broker responses, and authentication error classification for the Agent Client Protocol (ACP).

## 2. Public Interface

### 2.1 JSON-RPC 2.0 Base Models
- `JsonRpcRequest(BaseModel)`:
  - `jsonrpc: Literal["2.0"] = "2.0"`
  - `id: int | str`
  - `method: str`
  - `params: dict[str, Any] | list[Any] | None = None`
- `JsonRpcResponse(BaseModel)`:
  - `jsonrpc: Literal["2.0"] = "2.0"`
  - `id: int | str | None = None`
  - `result: Any = None`
  - `error: JsonRpcError | None = None`
- `JsonRpcError(BaseModel)`:
  - `code: int`
  - `message: str`
  - `data: Any = None`
- `JsonRpcNotification(BaseModel)`:
  - `jsonrpc: Literal["2.0"] = "2.0"`
  - `method: str`
  - `params: dict[str, Any] | list[Any] | None = None`

### 2.2 Wire Framing Functions
- `encode_message(msg: BaseModel | dict[str, Any]) -> str`
  - Returns compact single-line JSON string ending with `\n`.
- `decode_message(raw: str | bytes) -> dict[str, Any]`
  - Parses JSON line. Raises `ValueError` or `json.JSONDecodeError` on invalid JSON.

### 2.3 ACP Payload Builders
- `build_initialize_request(req_id: int | str, client_name: str = "hermes-agy-acp", client_version: str = "1.0.0") -> JsonRpcRequest`
  - Method: `"initialize"`
  - Params: `{"protocolVersion": 1, "clientInfo": {"name": client_name, "version": client_version}, "capabilities": {"streaming": True}}`
- `build_session_new_request(req_id: int | str, cwd: str, model: str | None = None) -> JsonRpcRequest`
  - Method: `"session/new"`
  - Params: `{"cwd": cwd, ...}` (includes `"model": model` if provided)
- `build_session_prompt_request(req_id: int | str, session_id: str, prompt: str | list[dict[str, Any]]) -> JsonRpcRequest`
  - Method: `"session/prompt"`
  - Params: `{"sessionId": session_id, "prompt": prompt}`
- `build_session_cancel_notification(session_id: str) -> JsonRpcNotification`
  - Method: `"session/cancel"`
  - Params: `{"sessionId": session_id}`

### 2.4 Permission Broker
- `build_permission_approval_response(req_id: int | str, decision: str = "allow") -> JsonRpcResponse`
  - Returns non-blocking auto-approval response: `{"jsonrpc": "2.0", "id": req_id, "result": {"decision": decision}}`.

### 2.5 Auth Error Detector
- `is_auth_error(error: JsonRpcError | dict[str, Any] | str | Exception) -> bool`
  - Inspects error code, message, or exception text.
  - Returns `True` if matches auth keywords: `"unauthenticated"`, `"oauth"`, `"unauthorized"`, `"token expired"`, `"auth failed"`, `"login required"`, or code 401 / -32001.
  - Returns `False` otherwise.
