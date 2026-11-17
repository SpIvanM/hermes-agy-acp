from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hermes_agy_acp.config import Settings, get_settings, parse_model_and_thinking
from hermes_agy_acp.orchestrator_bridge import (
    StreamChunks,
    completion_to_stream_chunks,
    extract_tool_calls_from_text,
    filter_hermes_tools,
    inject_bootstrap_prompt,
    process_proxy_response,
    render_tool_bridge_sections,
)
from hermes_agy_acp.process_pool import ACPProcessSupervisor
from hermes_agy_acp.protocol import (
    build_initialize_request,
    build_jsonrpc_error,
    build_jsonrpc_result,
    build_permission_approval_response,
    build_session_cancel_notification,
    build_session_load_request,
    build_session_new_request,
    build_session_prompt_request,
    build_session_set_config_option_request,
    build_session_set_mode_request,
    build_session_set_model_request,
    decode_message,
    encode_message,
    is_auth_error,
)
from hermes_agy_acp.session_map import SessionRegistry

logger = logging.getLogger(__name__)

_ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
    "context": "Context",
}

_PROMPT_PREAMBLE = (
    "You are being used as the active ACP agent backend for Hermes.",
    "Use your native ACP tools and subagents (including invoke_subagent) to complete tasks.",
)


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
    reasoning: str | None = None,
) -> ResponseNamespace:
    finish_reason = "tool_calls" if tool_calls else "stop"
    msg = ResponseNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls or [],
        reasoning=reasoning or None,
        reasoning_content=reasoning or None,
        reasoning_details=None,
    )
    choice = ResponseNamespace(
        message=msg,
        index=0,
        finish_reason=finish_reason,
    )
    usage = ResponseNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        prompt_tokens_details=ResponseNamespace(cached_tokens=0),
    )
    return ResponseNamespace(
        choices=[choice],
        model=model,
        usage=usage,
    )


def _resolve_timeout(timeout: Any, default_seconds: float) -> float:
    if timeout is None:
        return default_seconds
    if isinstance(timeout, (int, float)):
        return float(timeout) if float(timeout) > 0 else default_seconds
    candidates = [
        getattr(timeout, attr, None)
        for attr in ("read", "write", "connect", "pool", "timeout")
    ]
    valid = [float(v) for v in candidates if isinstance(v, (int, float)) and float(v) > 0]
    return max(valid) if valid else default_seconds


def _is_writable_acp_dir(path: Path) -> bool:
    return path != Path("/") and path.is_dir() and os.access(path, os.W_OK | os.X_OK)


def _default_acp_cwd(explicit: str | None) -> str:
    candidate: Path | None = None
    if explicit:
        try:
            candidate = Path(explicit).expanduser().resolve()
        except Exception:
            candidate = None
    else:
        try:
            from agent.runtime_cwd import resolve_agent_cwd  # type: ignore[import-not-found]

            candidate = Path(resolve_agent_cwd()).expanduser().resolve()
        except Exception:
            try:
                candidate = Path.cwd().resolve()
            except Exception:
                candidate = None

    if candidate is not None and _is_writable_acp_dir(candidate):
        return str(candidate)

    home = Path.home().resolve()
    work_dir = home / "work"
    if _is_writable_acp_dir(work_dir):
        return str(work_dir)
    return str(home)


def _ensure_path_within_cwd(path_text: str, cwd: str, *, verb: str) -> Path:
    if not path_text:
        raise PermissionError(f"{verb} denied: empty path.")
    if path_text.startswith(("\\\\?\\", "\\\\.\\", "\\\\")):
        raise PermissionError(f"{verb} denied: NT namespace or UNC paths are not allowed.")
    try:
        from agent.file_safety import get_nt_namespace_error  # type: ignore[import-not-found]

        if nt_err := get_nt_namespace_error(path_text, verb=verb):
            raise PermissionError(nt_err)
    except ImportError:
        pass

    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = candidate.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


def _fs_read_text_file(params: dict[str, Any], cwd: str) -> dict[str, str]:
    path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd, verb="Read")
    try:
        from agent.file_safety import get_read_block_error  # type: ignore[import-not-found]

        if block_err := get_read_block_error(str(path)):
            raise PermissionError(block_err)
    except ImportError:
        pass

    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        content = ""

    raw_line = params.get("line")
    raw_limit = params.get("limit")
    start = raw_line - 1 if isinstance(raw_line, int) and not isinstance(raw_line, bool) and raw_line > 1 else 0
    valid_limit = raw_limit if isinstance(raw_limit, int) and not isinstance(raw_limit, bool) and raw_limit > 0 else None
    if start > 0 or valid_limit is not None:
        end = start + valid_limit if valid_limit is not None else None
        content = "".join(content.splitlines(keepends=True)[start:end])

    if content:
        try:
            from agent.redact import redact_sensitive_text  # type: ignore[import-not-found]

            content = redact_sensitive_text(content, force=True)
        except ImportError:
            pass
    return {"content": content}


def _fs_write_text_file(params: dict[str, Any], cwd: str) -> None:
    path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd, verb="Write")
    try:
        from agent.file_safety import (  # type: ignore[import-not-found]
            get_write_denied_error,
            is_write_approval_required,
        )

        if denied := get_write_denied_error(str(path)):
            raise PermissionError(denied)
        if is_write_approval_required(str(path)):
            raise PermissionError(
                f"Write denied: '{path}' requires interactive approval and cannot be written via ACP."
            )
    except ImportError:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(params.get("content") or ""), encoding="utf-8")
    return None


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if isinstance(content.get("content"), str):
            return content["content"].strip()
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                parts.append(item["text"].strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    filtered_tools = filter_hermes_tools(tools)
    sections: list[str] = [*_PROMPT_PREAMBLE]
    if filtered_tools:
        sections.extend(render_tool_bridge_sections(filtered_tools, tool_choice))
    transcript: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "unknown").strip().lower()
        rendered = _render_message_content(msg.get("content"))
        if rendered:
            label = _ROLE_LABELS.get(role, "Context")
            transcript.append(f"{label}:\n{rendered}")
    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    return "\n\n".join(s.strip() for s in sections if s and s.strip())


def _extract_chunk_text(content: Any) -> str:
    if isinstance(content, dict):
        return str(content.get("text") or "")
    if isinstance(content, str):
        return content
    return ""


@dataclass
class _ActiveRequest:
    request_key: int
    supervisor: ACPProcessSupervisor
    hermes_session_id: str | None = None
    acp_session_id: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


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
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | tuple[str, ...] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | tuple[str, ...] | None = None,
        cwd: str | None = None,
        **_: Any,
    ) -> None:
        self.settings = settings or get_settings()
        self._injected_supervisor = supervisor
        self.supervisor = supervisor
        self.registry = registry or SessionRegistry(db_path=self.settings.state_db_path)

        self.api_key = api_key or "antigravity"
        self.base_url = base_url or "acp://antigravity"
        self._default_headers = dict(default_headers or {})

        raw_cmd = acp_command or command
        raw_args = acp_args if acp_args is not None else args
        self._acp_command: str | None = str(raw_cmd).strip() if raw_cmd else None
        if raw_args is not None:
            self._acp_args: list[str] = [str(a) for a in raw_args]
        else:
            self._acp_args = list(self.settings.agy_acp_args)
        self._acp_cwd = _default_acp_cwd(acp_cwd or cwd)

        self._id_counter = 0
        self._req_seq = 0
        self._lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active_requests: dict[int, _ActiveRequest] = {}
        self._thread_local = threading.local()
        self.is_closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create_chat_completion))

    def _next_id(self) -> int:
        with self._lock:
            self._id_counter += 1
            return self._id_counter

    def _next_request_key(self) -> int:
        with self._lock:
            self._req_seq += 1
            return self._req_seq

    def _build_command(self) -> list[str]:
        if self._acp_command:
            return [self._acp_command, *self._acp_args]
        return self.settings.get_server_command()

    def _make_supervisor(self, target_cwd: str) -> tuple[ACPProcessSupervisor, bool]:
        if self._injected_supervisor is not None:
            return self._injected_supervisor, False
        return ACPProcessSupervisor(command=self._build_command(), cwd=target_cwd, auto_respawn=False), True

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        supervisor: ACPProcessSupervisor,
        cwd: str,
        expected_session_id: str | None = None,
        text_parts: list[str] | None = None,
        reasoning_parts: list[str] | None = None,
    ) -> bool:
        """Process server->client notifications and requests. Return True if consumed."""
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}

        if method in ("session/request_permission", "session/requestPermission"):
            perm_id = msg.get("id")
            if perm_id is not None:
                options = params.get("options") if isinstance(params.get("options"), list) else None
                approval = build_permission_approval_response(perm_id, options=options)
                supervisor.write_line(encode_message(approval))
            return True

        if method == "fs/read_text_file":
            req_id = msg.get("id")
            if req_id is not None:
                try:
                    res = _fs_read_text_file(params, cwd)
                    supervisor.write_line(encode_message(build_jsonrpc_result(req_id, res)))
                except Exception as exc:
                    supervisor.write_line(encode_message(build_jsonrpc_error(req_id, -32602, str(exc))))
            return True

        if method == "fs/write_text_file":
            req_id = msg.get("id")
            if req_id is not None:
                try:
                    res = _fs_write_text_file(params, cwd)
                    supervisor.write_line(encode_message(build_jsonrpc_result(req_id, res)))
                except Exception as exc:
                    supervisor.write_line(encode_message(build_jsonrpc_error(req_id, -32602, str(exc))))
            return True

        if method in ("session/update", "session/notification"):
            msg_sess = params.get("sessionId")
            if expected_session_id and msg_sess and msg_sess != expected_session_id:
                return True

            update = params.get("update")
            if isinstance(update, dict):
                kind = str(update.get("sessionUpdate") or "").strip()
                chunk = _extract_chunk_text(update.get("content"))
                if kind == "agent_message_chunk":
                    if chunk and text_parts is not None:
                        text_parts.append(chunk)
                elif kind == "agent_thought_chunk":
                    if chunk and reasoning_parts is not None:
                        reasoning_parts.append(chunk)
                return True

            if "sessionUpdate" not in params and text_parts is not None:
                fallback = params.get("text") or params.get("delta") or ""
                if isinstance(fallback, str) and fallback:
                    text_parts.append(fallback)
            return True

        req_id = msg.get("id")
        if req_id is not None:
            supervisor.write_line(
                encode_message(build_jsonrpc_error(req_id, -32601, f"Unsupported method '{method}'"))
            )
        return True

    def _send_and_receive(
        self,
        req: Any,
        *,
        supervisor: ACPProcessSupervisor,
        cwd: str,
        deadline: float,
        expected_session_id: str | None = None,
        text_parts: list[str] | None = None,
        reasoning_parts: list[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        supervisor.write_line(encode_message(req))
        req_id = req.id if hasattr(req, "id") else req.get("id")

        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("ACP request cancelled")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for ACP response")

            try:
                raw_line = supervisor.read_line(timeout=min(remaining, 1.0))
            except TimeoutError:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("ACP request cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for ACP response")
                continue
            except EOFError as err:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("ACP request cancelled") from err
                raise

            msg = decode_message(raw_line)
            if self._handle_server_message(
                msg,
                supervisor=supervisor,
                cwd=cwd,
                expected_session_id=expected_session_id,
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
            ):
                continue

            if msg.get("id") == req_id:
                if "error" in msg and msg["error"] is not None:
                    err = msg["error"]
                    if is_auth_error(err):
                        raise PermissionError(f"Antigravity authentication error: {err}")
                    raise RuntimeError(f"ACP error response: {err}")
                result = msg.get("result")
                if isinstance(result, dict):
                    return result
                return {"value": result}

    def _apply_mode_and_model(
        self,
        *,
        supervisor: ACPProcessSupervisor,
        cwd: str,
        deadline: float,
        acp_sess: str,
        session_result: dict[str, Any],
        requested_model: str | None,
        requested_thinking: str | None,
        cancel_event: threading.Event,
    ) -> None:
        acp_mode = str(self.settings.agy_acp_mode or "yolo").strip()
        if acp_mode:
            mode_req = build_session_set_mode_request(
                self._next_id(),
                session_id=acp_sess,
                mode_id=acp_mode,
            )
            try:
                self._send_and_receive(
                    mode_req,
                    supervisor=supervisor,
                    cwd=cwd,
                    deadline=deadline,
                    expected_session_id=acp_sess,
                    cancel_event=cancel_event,
                )
            except Exception as exc:
                logger.warning("session/set_mode(%s) failed; continuing: %s", acp_mode, exc)

        raw_model = str(requested_model or self.settings.agy_default_model or "").strip()
        if not raw_model or raw_model in ("acp", "antigravity-acp"):
            return

        base_model, parsed_thinking = parse_model_and_thinking(raw_model)
        target_thinking = str(requested_thinking or parsed_thinking or "").strip() or None

        config_options = session_result.get("configOptions")
        model_opt: dict[str, Any] | None = None
        thinking_opt: dict[str, Any] | None = None
        if isinstance(config_options, list):
            for opt in config_options:
                if not isinstance(opt, dict):
                    continue
                opt_id = str(opt.get("id") or "").strip()
                opt_cat = str(opt.get("category") or "").strip()
                if opt_id == "model" or opt_cat == "model":
                    model_opt = opt
                elif opt_id == "thinking_level" or opt_cat in ("thought_level", "thinking_level"):
                    thinking_opt = opt

        def _option_values(opt: dict[str, Any]) -> list[str]:
            vals: list[str] = []
            for entry in opt.get("options") or []:
                if isinstance(entry, dict) and entry.get("value"):
                    vals.append(str(entry["value"]).strip())
            return vals

        if model_opt is not None:
            allowed_models = _option_values(model_opt)
            chosen_model = base_model or raw_model
            if allowed_models:
                if base_model and base_model in allowed_models:
                    chosen_model = base_model
                elif raw_model in allowed_models:
                    chosen_model = raw_model
            cfg_id = str(model_opt.get("id") or "model")
            try:
                self._send_and_receive(
                    build_session_set_config_option_request(
                        self._next_id(),
                        session_id=acp_sess,
                        config_id=cfg_id,
                        value=chosen_model,
                    ),
                    supervisor=supervisor,
                    cwd=cwd,
                    deadline=deadline,
                    expected_session_id=acp_sess,
                    cancel_event=cancel_event,
                )
            except Exception as exc:
                logger.warning("session/set_config_option(model=%s) failed, trying session/set_model: %s", chosen_model, exc)
                with contextlib.suppress(Exception):
                    self._send_and_receive(
                        build_session_set_model_request(
                            self._next_id(),
                            session_id=acp_sess,
                            model_id=chosen_model,
                        ),
                        supervisor=supervisor,
                        cwd=cwd,
                        deadline=deadline,
                        expected_session_id=acp_sess,
                        cancel_event=cancel_event,
                    )
        elif isinstance(session_result.get("models"), dict):
            legacy_models = [
                str(m.get("modelId")).strip()
                for m in (session_result["models"].get("availableModels") or [])
                if isinstance(m, dict) and m.get("modelId")
            ]
            chosen_legacy = raw_model if raw_model in legacy_models else (base_model or raw_model)
            with contextlib.suppress(Exception):
                self._send_and_receive(
                    build_session_set_model_request(
                        self._next_id(),
                        session_id=acp_sess,
                        model_id=chosen_legacy,
                    ),
                    supervisor=supervisor,
                    cwd=cwd,
                    deadline=deadline,
                    expected_session_id=acp_sess,
                    cancel_event=cancel_event,
                )
        else:
            chosen_model = base_model or raw_model
            try:
                self._send_and_receive(
                    build_session_set_config_option_request(
                        self._next_id(),
                        session_id=acp_sess,
                        config_id="model",
                        value=chosen_model,
                    ),
                    supervisor=supervisor,
                    cwd=cwd,
                    deadline=deadline,
                    expected_session_id=acp_sess,
                    cancel_event=cancel_event,
                )
            except Exception:
                with contextlib.suppress(Exception):
                    self._send_and_receive(
                        build_session_set_model_request(
                            self._next_id(),
                            session_id=acp_sess,
                            model_id=chosen_model,
                        ),
                        supervisor=supervisor,
                        cwd=cwd,
                        deadline=deadline,
                        expected_session_id=acp_sess,
                        cancel_event=cancel_event,
                    )

        if target_thinking:
            if thinking_opt is not None:
                allowed_thinking = _option_values(thinking_opt)
                if not allowed_thinking or target_thinking in allowed_thinking:
                    cfg_id = str(thinking_opt.get("id") or "thinking_level")
                    with contextlib.suppress(Exception):
                        self._send_and_receive(
                            build_session_set_config_option_request(
                                self._next_id(),
                                session_id=acp_sess,
                                config_id=cfg_id,
                                value=target_thinking,
                            ),
                            supervisor=supervisor,
                            cwd=cwd,
                            deadline=deadline,
                            expected_session_id=acp_sess,
                            cancel_event=cancel_event,
                        )
            elif not isinstance(session_result.get("models"), dict) and config_options is None:
                with contextlib.suppress(Exception):
                    self._send_and_receive(
                        build_session_set_config_option_request(
                            self._next_id(),
                            session_id=acp_sess,
                            config_id="thinking_level",
                            value=target_thinking,
                        ),
                        supervisor=supervisor,
                        cwd=cwd,
                        deadline=deadline,
                        expected_session_id=acp_sess,
                        cancel_event=cancel_event,
                    )

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
    ) -> ResponseNamespace | StreamChunks:
        """Execute one isolated ACP turn (initialize -> session -> mode -> model/thinking -> prompt)."""
        target_cwd = _default_acp_cwd(cwd or self._acp_cwd)
        effective_timeout = _resolve_timeout(timeout, self.settings.agy_prompt_timeout)
        deadline = time.monotonic() + effective_timeout

        sup, owns_supervisor = self._make_supervisor(target_cwd)
        req_key = self._next_request_key()
        active_req = _ActiveRequest(
            request_key=req_key,
            supervisor=sup,
            hermes_session_id=session_id,
        )

        with self._active_lock:
            self._active_requests[req_key] = active_req
        prev_thread_key = getattr(self._thread_local, "active_request_key", None)
        self._thread_local.active_request_key = req_key

        try:
            sup.ensure_running()

            self._send_and_receive(
                build_initialize_request(self._next_id()),
                supervisor=sup,
                cwd=target_cwd,
                deadline=deadline,
                cancel_event=active_req.cancel_event,
            )

            session_result: dict[str, Any] = {}
            acp_sess: str | None = None
            if session_id:
                existing = self.registry.get_acp_session_id(session_id)
                if existing:
                    try:
                        session_result = self._send_and_receive(
                            build_session_load_request(self._next_id(), existing, target_cwd),
                            supervisor=sup,
                            cwd=target_cwd,
                            deadline=deadline,
                            cancel_event=active_req.cancel_event,
                        )
                        acp_sess = str(session_result.get("sessionId") or existing)
                    except Exception as exc:
                        logger.debug("session/load failed for %s, recreating: %s", existing, exc)
                        self.registry.reset_session(session_id)

            if not acp_sess:
                session_result = self._send_and_receive(
                    build_session_new_request(self._next_id(), cwd=target_cwd),
                    supervisor=sup,
                    cwd=target_cwd,
                    deadline=deadline,
                    cancel_event=active_req.cancel_event,
                )
                acp_sess = str(session_result.get("sessionId") or session_result.get("session_id") or "").strip()
                if not acp_sess:
                    raise RuntimeError("ACP session/new did not return a sessionId")
                if session_id:
                    self.registry.save_mapping(
                        session_id,
                        acp_sess,
                        metadata={"model": model or self.settings.agy_default_model, "cwd": target_cwd},
                    )

            active_req.acp_session_id = acp_sess

            self._apply_mode_and_model(
                supervisor=sup,
                cwd=target_cwd,
                deadline=deadline,
                acp_sess=acp_sess,
                session_result=session_result,
                requested_model=model,
                requested_thinking=kwargs.get("thinking_level"),
                cancel_event=active_req.cancel_event,
            )

            prepared_messages = inject_bootstrap_prompt(
                messages,
                bootstrap_text=self.settings.agy_bootstrap_prompt,
            )

            prompt_str = _format_messages_as_prompt(
                prepared_messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            prompt_req = build_session_prompt_request(
                self._next_id(),
                session_id=acp_sess,
                prompt=[{"type": "text", "text": prompt_str}],
            )

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            self._send_and_receive(
                prompt_req,
                supervisor=sup,
                cwd=target_cwd,
                deadline=deadline,
                expected_session_id=acp_sess,
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
                cancel_event=active_req.cancel_event,
            )

            raw_text = "".join(text_parts)
            reasoning_text = "".join(reasoning_parts)
            tool_calls, cleaned_text = extract_tool_calls_from_text(raw_text)

            completion = _build_completion(
                content=cleaned_text,
                model=model or self.settings.agy_default_model,
                tool_calls=tool_calls,
                reasoning=reasoning_text or None,
            )
            if stream:
                return completion_to_stream_chunks(completion)
            return completion
        finally:
            self._thread_local.active_request_key = prev_thread_key
            with self._active_lock:
                self._active_requests.pop(req_key, None)
            if owns_supervisor:
                sup.terminate()

    def cancel(
        self,
        request_key: int | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Cancel only the targeted in-flight ACP request (send session/cancel + terminate process)."""
        targets: list[_ActiveRequest] = []
        with self._active_lock:
            if request_key is not None:
                if req := self._active_requests.get(request_key):
                    targets = [req]
            elif session_id is not None:
                targets = [
                    req
                    for req in self._active_requests.values()
                    if req.hermes_session_id == session_id or req.acp_session_id == session_id
                ]
            else:
                thread_key = getattr(self._thread_local, "active_request_key", None)
                if thread_key is not None and thread_key in self._active_requests:
                    targets = [self._active_requests[thread_key]]
                elif self._active_requests:
                    latest_key = max(self._active_requests)
                    targets = [self._active_requests[latest_key]]

        if not targets:
            return False

        for req in targets:
            req.cancel_event.set()
            if req.acp_session_id:
                proc = getattr(req.supervisor, "process", None)
                stdin = getattr(proc, "stdin", None) if proc is not None else None
                if proc is not None and proc.poll() is None and stdin is not None:
                    line = encode_message(build_session_cancel_notification(req.acp_session_id))
                    payload = line if line.endswith("\n") else f"{line}\n"
                    write_lock = getattr(req.supervisor, "_write_lock", None) or contextlib.nullcontext()
                    try:
                        with write_lock:
                            if proc.poll() is None and not getattr(stdin, "closed", False):
                                stdin.write(payload.encode("utf-8"))
                                stdin.flush()
                        time.sleep(0.05)
                    except (BrokenPipeError, OSError, ValueError):
                        pass
            with contextlib.suppress(Exception):
                req.supervisor.terminate()
        return True

    def reset(self, hermes_session_id: str) -> bool:
        """Reset session mapping in registry."""
        return self.registry.reset_session(hermes_session_id)

    def close(self) -> None:
        """Terminate any active request processes owned by this client and close registry."""
        self.is_closed = True
        with self._active_lock:
            active = list(self._active_requests.values())
        for req in active:
            req.cancel_event.set()
            with contextlib.suppress(Exception):
                req.supervisor.terminate()
        if self._injected_supervisor is not None:
            with contextlib.suppress(Exception):
                self._injected_supervisor.terminate()
        self.registry.close()


__all__ = [
    "AntigravityACPClient",
    "BaseACPClient",
    "ResponseNamespace",
    "process_proxy_response",
]
