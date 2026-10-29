from __future__ import annotations

import logging
import os
import threading
from types import SimpleNamespace
from collections.abc import Iterator
from typing import Any

from hermes_agy_acp.config import Settings, get_settings
from hermes_agy_acp.process_pool import ACPProcessSupervisor
from hermes_agy_acp.protocol import (
    build_initialize_request,
    build_permission_approval_response,
    build_session_new_request,
    build_session_prompt_request,
    decode_message,
    encode_message,
    is_auth_error,
)
from hermes_agy_acp.session_map import SessionRegistry

logger = logging.getLogger(__name__)


class ResponseNamespace(SimpleNamespace):
    """Namespace that supports attribute access, dict indexing, and vars()."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and isinstance(args[0], dict):
            kwargs.update(args[0])
            args = args[1:]
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: str) -> Any:
        val = getattr(self, key)
        if isinstance(val, dict):
            return ResponseNamespace(**val)
        return val

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _build_completion(
    content: str,
    model: str = "gemini-3.8-flash-high",
    tool_calls: list[Any] | None = None,
) -> ResponseNamespace:
    msg = ResponseNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = ResponseNamespace(
        message=msg,
        index=0,
        finish_reason="stop",
    )
    usage = ResponseNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )
    return ResponseNamespace(
        choices=[choice],
        model=model,
        usage=usage,
    )


class BaseACPClient:
    """Base class declaring Hermes ACP provider contracts."""

    HERMES_SKIP_TRANSPORT_WRAP: bool = True
    HERMES_SKIP_ASYNC_WRAP: bool = True
    api_key: str = "antigravity"
    base_url: str = "acp://antigravity"


class AntigravityACPClient(BaseACPClient):
    """Client adapter connecting Hermes Agent to Antigravity ACP server."""

    def __init__(
        self,
        supervisor: ACPProcessSupervisor | None = None,
        registry: SessionRegistry | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or ACPProcessSupervisor()
        self.registry = registry or SessionRegistry()
        self._id_counter = 0
        self._lock = threading.Lock()
        self._active_sessions: set[str] = set()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create_chat_completion))

    def _next_id(self) -> int:
        with self._lock:
            self._id_counter += 1
            return self._id_counter

    def _send_and_receive(self, req: Any) -> dict[str, Any]:
        """Send a request and wait for matching response, handling permission requests."""
        self.supervisor.write_line(encode_message(req))
        req_id = req.id if hasattr(req, "id") else req.get("id")

        while True:
            raw_line = self.supervisor.read_line(timeout=30.0)
            msg = decode_message(raw_line)

            # Auto-approval of session permission broker requests
            method = msg.get("method")
            if method in ("session/request_permission", "session/requestPermission"):
                perm_id = msg.get("id")
                if perm_id is not None:
                    approval = build_permission_approval_response(perm_id)
                    self.supervisor.write_line(encode_message(approval))
                continue

            if msg.get("id") == req_id:
                if "error" in msg:
                    err = msg["error"]
                    if is_auth_error(err):
                        raise PermissionError(f"Antigravity authentication error: {err}")
                    raise RuntimeError(f"ACP error response: {err}")
                result = msg.get("result", {})
                if isinstance(result, dict):
                    return result
                return {"value": result}

    def initialize(self) -> dict[str, Any]:
        """Send initialize handshake to ACP server."""
        req = build_initialize_request(self._next_id())
        return self._send_and_receive(req)

    def ensure_session(
        self,
        hermes_session_id: str,
        cwd: str | None = None,
        model: str | None = None,
        force_new: bool = False,
    ) -> str:
        """Ensure ACP session is created and mapped to hermes_session_id."""
        from hermes_agy_acp.protocol import JsonRpcRequest

        target_cwd = cwd or os.getcwd()
        target_model = model or self.settings.agy_default_model

        if not force_new:
            existing = self.registry.get_acp_session_id(hermes_session_id)
            if existing:
                if existing in self._active_sessions:
                    return existing
                # Attempt to load session into this process
                load_req = JsonRpcRequest(
                    id=self._next_id(),
                    method="session/load",
                    params={"sessionId": existing, "cwd": target_cwd, "mcpServers": []},
                )
                try:
                    self._send_and_receive(load_req)
                    self._active_sessions.add(existing)
                    return existing
                except Exception as err:
                    logger.debug("session/load failed for %s, recreating: %s", existing, err)
                    self.registry.reset_session(hermes_session_id)

        req = build_session_new_request(
            self._next_id(),
            cwd=target_cwd,
            model=target_model,
        )
        res = self._send_and_receive(req)
        acp_session_id = str(res.get("sessionId") or res.get("session_id"))
        self._active_sessions.add(acp_session_id)
        self.registry.save_mapping(
            hermes_session_id,
            acp_session_id,
            metadata={"model": target_model, "cwd": target_cwd},
        )
        return acp_session_id

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        stream: bool = False,
        session_id: str | None = None,
        cwd: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Create chat completion translating OpenAI messages to ACP prompt stream."""
        h_sess = session_id or "default"
        acp_sess = self.ensure_session(h_sess, cwd=cwd, model=model)

        # Build prompt string from messages
        prompt_parts: list[str] = []
        for m in messages:
            content = m.get("content", "")
            role = m.get("role", "user")
            if isinstance(content, str):
                prompt_parts.append(f"[{role}]: {content}")
            elif isinstance(content, list):
                text_blocks = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                prompt_parts.append(f"[{role}]: {' '.join(text_blocks)}")

        prompt_str = "\n\n".join(prompt_parts)
        prompt_payload = [{"type": "text", "text": prompt_str}]
        req_id = self._next_id()
        prompt_req = build_session_prompt_request(req_id, session_id=acp_sess, prompt=prompt_payload)
        self.supervisor.write_line(encode_message(prompt_req))

        def _stream() -> Iterator[ResponseNamespace]:
            while True:
                raw_line = self.supervisor.read_line(timeout=30.0)
                msg = decode_message(raw_line)

                method = msg.get("method")
                if method in ("session/request_permission", "session/requestPermission"):
                    perm_id = msg.get("id")
                    if perm_id is not None:
                        approval = build_permission_approval_response(perm_id)
                        self.supervisor.write_line(encode_message(approval))
                    continue

                if method in ("session/update", "session/notification"):
                    params = msg.get("params", {})
                    text_delta = ""
                    if isinstance(params, dict):
                        text_delta = params.get("text") or params.get("delta") or params.get("content") or ""
                        if not text_delta and "update" in params:
                            upd = params["update"]
                            if isinstance(upd, dict):
                                cnt = upd.get("content")
                                if isinstance(cnt, dict):
                                    text_delta = cnt.get("text") or ""
                                elif isinstance(cnt, str):
                                    text_delta = cnt
                    if text_delta:
                        yield ResponseNamespace(
                            choices=[
                                ResponseNamespace(
                                    delta=ResponseNamespace(content=text_delta),
                                    index=0,
                                    finish_reason=None,
                                )
                            ]
                        )

                if msg.get("id") == req_id:
                    if "error" in msg:
                        err = msg["error"]
                        if is_auth_error(err):
                            raise PermissionError(f"Antigravity authentication error: {err}")
                        raise RuntimeError(f"ACP prompt error: {err}")
                    yield ResponseNamespace(
                        choices=[
                            ResponseNamespace(
                                delta=ResponseNamespace({}),
                                index=0,
                                finish_reason="stop",
                            )
                        ]
                    )
                    break

        if stream:
            return _stream()

        accumulated: list[str] = []
        for chunk in _stream():
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None) or choice.get("delta")
            c = getattr(delta, "content", None) or (delta.get("content") if isinstance(delta, dict) else None)
            if c:
                accumulated.append(c)

        full_text = "".join(accumulated)
        return _build_completion(
            content=full_text,
            model=model or self.settings.agy_default_model,
        )
    def reset(self, hermes_session_id: str) -> bool:
        """Reset session mapping in registry."""
        return self.registry.reset_session(hermes_session_id)

    def close(self) -> None:
        """Close supervisor process and registry."""
        self.supervisor.terminate()
        self.registry.close()
