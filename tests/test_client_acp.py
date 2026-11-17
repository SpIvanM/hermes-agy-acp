from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_agy_acp.client import AntigravityACPClient
from hermes_agy_acp.config import Settings
from hermes_agy_acp.process_pool import ACPProcessSupervisor

FAKE_SERVER = str(Path(__file__).parent / "fake_acp_server.py")


def _read_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": {"type": "object", "properties": {"action": {"type": "string"}}},
        },
    }


def test_full_handshake_yolo_mode_and_combined_model_config(tmp_path, monkeypatch):
    log_path = tmp_path / "acp.log"
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "basic")
    monkeypatch.setenv("FAKE_ACP_LOG", str(log_path))

    client = AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[FAKE_SERVER],
        acp_cwd=str(tmp_path),
        settings=Settings(state_db_path=tmp_path / "state.db"),
    )
    resp = client.create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-3.8-flash-high",
    )
    assert resp.choices[0].message.content == "ok"

    entries = _read_log(log_path)
    methods = [e.get("method") for e in entries if "method" in e]
    assert methods[:3] == ["initialize", "session/new", "session/set_mode"]
    mode_req = next(e for e in entries if e.get("method") == "session/set_mode")
    assert mode_req["params"]["modeId"] == "yolo"

    cfg_reqs = [e for e in entries if e.get("method") == "session/set_config_option"]
    cfg_pairs = [(r["params"]["configId"], r["params"]["value"]) for r in cfg_reqs]
    assert ("model", "gemini-3.8-flash") in cfg_pairs
    assert ("thinking_level", "high") in cfg_pairs
    assert methods[-1] == "session/prompt"


def test_direct_models_and_legacy_set_model_fallback(tmp_path, monkeypatch):
    log_path = tmp_path / "acp.log"
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "legacy_models_only")
    monkeypatch.setenv("FAKE_ACP_LOG", str(log_path))

    client = AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[FAKE_SERVER],
        acp_cwd=str(tmp_path),
        settings=Settings(state_db_path=tmp_path / "state.db"),
    )
    resp = client.create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-4-hydrogen",
    )
    assert resp.choices[0].message.content == "ok"
    entries = _read_log(log_path)
    set_model = [e for e in entries if e.get("method") == "session/set_model"]
    assert len(set_model) == 1
    assert set_model[0]["params"]["modelId"] == "gemini-4-hydrogen"


def test_permission_v1_and_scoped_fs_rpc(tmp_path, monkeypatch):
    log_path = tmp_path / "acp.log"
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "permission_and_fs")
    monkeypatch.setenv("FAKE_ACP_LOG", str(log_path))
    monkeypatch.setenv("FAKE_ACP_CWD", str(tmp_path))

    client = AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[FAKE_SERVER],
        acp_cwd=str(tmp_path),
        settings=Settings(state_db_path=tmp_path / "state.db"),
    )
    resp = client.create_chat_completion(messages=[{"role": "user", "content": "fs test"}])
    assert resp.choices[0].message.content == "ok"
    assert (tmp_path / "created_by_acp.txt").read_text(encoding="utf-8") == "hello-fs"

    entries = _read_log(log_path)
    perm_reply = next(e["perm_reply"] for e in entries if "perm_reply" in e)
    assert perm_reply["result"] == {"outcome": {"outcome": "selected", "optionId": "allow_always"}}

    read_reply = next(e["read_reply"] for e in entries if "read_reply" in e)
    assert read_reply["result"] == {"content": "hello-fs"}

    bad_read_reply = next(e["bad_read_reply"] for e in entries if "bad_read_reply" in e)
    assert "error" in bad_read_reply


def test_session_update_filtering_and_thought_separation(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "noise_and_thought")
    client = AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[FAKE_SERVER],
        acp_cwd=str(tmp_path),
        settings=Settings(state_db_path=tmp_path / "state.db"),
    )
    resp = client.create_chat_completion(messages=[{"role": "user", "content": "think"}])
    msg = resp.choices[0].message
    assert msg.content == "clean-answer"
    assert "WRONG_SESSION" not in msg.content
    assert "SHOULD_NOT_APPEAR" not in msg.content
    assert msg.reasoning == "thinking-step"
    assert msg.reasoning_content == "thinking-step"


def test_tool_filtering_and_tool_call_bridge_roundtrip(tmp_path, monkeypatch):
    log_path = tmp_path / "acp.log"
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "tool_call")
    monkeypatch.setenv("FAKE_ACP_LOG", str(log_path))

    client = AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[FAKE_SERVER],
        acp_cwd=str(tmp_path),
        settings=Settings(state_db_path=tmp_path / "state.db"),
    )
    resp = client.create_chat_completion(
        messages=[{"role": "user", "content": "list memory"}],
        tools=[_tool("read_file"), _tool("terminal"), _tool("delegate_task"), _tool("memory")],
    )
    choice = resp.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert len(choice.message.tool_calls) == 1
    assert choice.message.tool_calls[0].function.name == "memory"
    assert "Checking memory now." in choice.message.content
    assert "<tool_call>" not in choice.message.content

    entries = _read_log(log_path)
    prompt_req = next(e for e in entries if e.get("method") == "session/prompt")
    prompt_text = prompt_req["params"]["prompt"][0]["text"]
    assert "memory" in prompt_text
    assert "read_file" not in prompt_text
    assert "delegate_task" not in prompt_text


def test_streaming_returns_stream_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "tool_call")
    client = AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[FAKE_SERVER],
        acp_cwd=str(tmp_path),
        settings=Settings(state_db_path=tmp_path / "state.db"),
    )
    chunks = client.create_chat_completion(
        messages=[{"role": "user", "content": "stream"}],
        tools=[_tool("memory")],
        stream=True,
    )
    chunk_list = list(chunks)
    assert len(chunk_list) >= 1
    data_chunk = chunk_list[0]
    assert data_chunk.choices[0].finish_reason == "tool_calls"
    assert data_chunk.choices[0].delta.tool_calls[0].function.name == "memory"


def test_concurrent_calls_are_isolated_without_stdout_crosstalk(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "echo_prompt_marker")
    client = AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[FAKE_SERVER],
        acp_cwd=str(tmp_path),
        settings=Settings(state_db_path=tmp_path / "state.db"),
    )

    def _call(idx: int) -> tuple[int, str]:
        marker = f"MARKER_{idx}"
        resp = client.create_chat_completion(
            messages=[{"role": "user", "content": marker}],
        )
        return idx, resp.choices[0].message.content

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_call, range(4)))

    for idx, content in results:
        assert f"ECHO:" in content
        assert f"MARKER_{idx}" in content
        for other in range(4):
            if other != idx:
                assert f"MARKER_{other}" not in content


def test_cancel_terminates_only_target_request_and_sends_session_cancel(tmp_path, monkeypatch):
    log_path = tmp_path / "cancel.log"
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "slow_hang")
    monkeypatch.setenv("FAKE_ACP_LOG", str(log_path))

    client = AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[FAKE_SERVER],
        acp_cwd=str(tmp_path),
        settings=Settings(state_db_path=tmp_path / "state.db"),
    )

    exc_holder: list[BaseException] = []

    def _worker() -> None:
        try:
            client.create_chat_completion(
                messages=[{"role": "user", "content": "hang"}],
                timeout=10.0,
            )
        except BaseException as exc:
            exc_holder.append(exc)

    t = threading.Thread(target=_worker)
    t.start()
    time.sleep(0.4)
    client.cancel()
    t.join(timeout=3.0)
    assert not t.is_alive()

    entries = _read_log(log_path)
    assert any(e.get("method") == "session/cancel" for e in entries)


def test_proxy_env_preserved_in_process_supervisor(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "env_dump")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:10808")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:10808")
    monkeypatch.setenv("ALL_PROXY", "socks5h://127.0.0.1:10808")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")

    client = AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[FAKE_SERVER],
        acp_cwd=str(tmp_path),
        settings=Settings(state_db_path=tmp_path / "state.db"),
    )
    resp = client.create_chat_completion(messages=[{"role": "user", "content": "env"}])
    dumped = json.loads(resp.choices[0].message.content)
    assert dumped["HTTP_PROXY"] == "http://127.0.0.1:10808"
    assert dumped["HTTPS_PROXY"] == "http://127.0.0.1:10808"
    assert dumped["ALL_PROXY"] == "socks5h://127.0.0.1:10808"
    assert dumped["NO_PROXY"] == "localhost,127.0.0.1"


def test_create_client_forwards_kwargs_and_profile_fields():
    import importlib.util

    root_init = Path(__file__).resolve().parent.parent / "__init__.py"
    spec = importlib.util.spec_from_file_location("hermes_agy_acp_plugin_root", root_init)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    client = mod.create_client(acp_command=sys.executable, acp_args=[FAKE_SERVER])
    assert client._acp_command == sys.executable
    assert client._acp_args == [FAKE_SERVER]


def test_cancel_one_session_does_not_affect_sibling_request(tmp_path, monkeypatch):
    client = AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[FAKE_SERVER],
        acp_cwd=str(tmp_path),
        settings=Settings(state_db_path=tmp_path / "state.db"),
    )

    monkeypatch.setenv("FAKE_ACP_SCENARIO", "slow_hang")
    slow_exc: list[BaseException] = []

    def _slow() -> None:
        try:
            client.create_chat_completion(
                messages=[{"role": "user", "content": "hang"}],
                session_id="slow-1",
                timeout=10.0,
            )
        except BaseException as exc:
            slow_exc.append(exc)

    t_slow = threading.Thread(target=_slow)
    t_slow.start()
    time.sleep(0.3)

    # Start a sibling request using basic scenario (override env before spawn)
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "basic")
    fast_resp = client.create_chat_completion(
        messages=[{"role": "user", "content": "fast"}],
        session_id="fast-2",
        timeout=10.0,
    )
    assert fast_resp.choices[0].message.content == "ok"

    assert client.cancel(session_id="slow-1") is True
    t_slow.join(timeout=3.0)
    assert not t_slow.is_alive()
    assert slow_exc


def test_root_plugin_profile_metadata_and_path_env(tmp_path, monkeypatch):
    import importlib.util

    hermes_root = Path("/home/h/.hermes/hermes-agent")
    if hermes_root.is_dir() and str(hermes_root) not in sys.path:
        sys.path.insert(0, str(hermes_root))

    root_init = Path(__file__).resolve().parent.parent / "__init__.py"
    spec = importlib.util.spec_from_file_location("hermes_agy_acp_plugin_meta", root_init)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    profile = mod._profile
    assert profile.supports_streaming is True
    assert profile.process_command == "agy_acp_server.par"
    assert profile.process_args == ("--uid=",)
    assert profile.process_command_env_vars == ("AGY_ACP_SERVER_PATH",)
    assert profile.process_args_env_var == "AGY_ACP_ARGS"
    assert "gemini-3.8-flash-high" in profile.fallback_models
    assert "gemini-4-hydrogen" in profile.fallback_models
    assert "gemini-4-argon" in profile.fallback_models

    from hermes_agy_acp.process_pool import build_subprocess_env

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:10808")
    env = build_subprocess_env()
    assert str(tmp_path / ".local" / "bin") in env["PATH"].split(":")
    assert env["http_proxy"] == "http://127.0.0.1:10808"


def test_cancel_after_terminate_does_not_respawn(tmp_path, monkeypatch):
    from hermes_agy_acp.client import _ActiveRequest

    monkeypatch.setenv("FAKE_ACP_SCENARIO", "slow_hang")
    client = AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[FAKE_SERVER],
        acp_cwd=str(tmp_path),
        settings=Settings(state_db_path=tmp_path / "state.db"),
    )
    sup, owns = client._make_supervisor(str(tmp_path))
    assert owns is True
    assert sup.auto_respawn is False

    start_calls = 0
    orig_start = ACPProcessSupervisor.start

    def _counting_start(self):
        nonlocal start_calls
        start_calls += 1
        return orig_start(self)

    ensure_calls = 0
    orig_ensure = ACPProcessSupervisor.ensure_running

    def _counting_ensure(self):
        nonlocal ensure_calls
        ensure_calls += 1
        return orig_ensure(self)

    monkeypatch.setattr(ACPProcessSupervisor, "start", _counting_start)
    monkeypatch.setattr(ACPProcessSupervisor, "ensure_running", _counting_ensure)

    sup.ensure_running()
    assert start_calls == 1
    assert ensure_calls == 1

    req_key = client._next_request_key()
    with client._active_lock:
        client._active_requests[req_key] = _ActiveRequest(
            request_key=req_key,
            supervisor=sup,
            acp_session_id="acp-sess-race",
        )

    # Simulate TOCTOU race: terminate() runs before or during cancel(), and cancel()
    # must never call ensure_running() or respawn a terminated process.
    orig_is_alive = sup.is_alive

    def _race_is_alive():
        alive = orig_is_alive()
        sup.terminate()
        return alive

    monkeypatch.setattr(sup, "is_alive", _race_is_alive)
    try:
        assert client.cancel(request_key=req_key) is True
        assert start_calls == 1
        assert ensure_calls == 1
        assert sup.process is None

        # Calling cancel() again after terminate() also must not respawn.
        assert client.cancel(request_key=req_key) is True
        assert start_calls == 1
        assert ensure_calls == 1
    finally:
        with client._active_lock:
            client._active_requests.pop(req_key, None)
        sup.terminate()


def test_early_eof_on_set_mode_and_cancel_do_not_respawn(tmp_path, monkeypatch):
    from hermes_agy_acp.client import _ActiveRequest

    log_path = tmp_path / "early_eof.log"
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "early_eof_on_set_mode")
    monkeypatch.setenv("FAKE_ACP_LOG", str(log_path))

    start_calls = 0
    orig_start = ACPProcessSupervisor.start

    def _counting_start(self):
        nonlocal start_calls
        start_calls += 1
        return orig_start(self)

    monkeypatch.setattr(ACPProcessSupervisor, "start", _counting_start)

    client = AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[FAKE_SERVER],
        acp_cwd=str(tmp_path),
        settings=Settings(state_db_path=tmp_path / "state.db"),
    )
    with pytest.raises(Exception):
        client.create_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-3.8-flash",
            timeout=5.0,
        )

    assert start_calls == 1
    entries = _read_log(log_path)
    init_reqs = [e for e in entries if e.get("method") == "initialize"]
    assert len(init_reqs) == 1

    # Also verify cancel() after an early process EOF (before terminate) does not respawn.
    sup, _ = client._make_supervisor(str(tmp_path))
    sup.ensure_running()
    assert start_calls == 2
    assert sup.process is not None
    sup.process.terminate()
    sup.process.wait(timeout=2.0)

    req_key = client._next_request_key()
    with client._active_lock:
        client._active_requests[req_key] = _ActiveRequest(
            request_key=req_key,
            supervisor=sup,
            acp_session_id="acp-sess-eof",
        )
    try:
        assert client.cancel(request_key=req_key) is True
        assert start_calls == 2
    finally:
        with client._active_lock:
            client._active_requests.pop(req_key, None)
        sup.terminate()


def test_fs_read_text_file_limit_with_line_none_one_and_bool_guards(tmp_path):
    from hermes_agy_acp.client import _fs_read_text_file

    target = tmp_path / "sample.txt"
    target.write_text("L1\nL2\nL3\nL4\nL5\n", encoding="utf-8")
    cwd = str(tmp_path)

    assert _fs_read_text_file({"path": str(target), "line": None, "limit": 2}, cwd) == {"content": "L1\nL2\n"}
    assert _fs_read_text_file({"path": str(target), "line": 1, "limit": 2}, cwd) == {"content": "L1\nL2\n"}
    assert _fs_read_text_file({"path": str(target), "line": 2, "limit": 2}, cwd) == {"content": "L2\nL3\n"}
    assert _fs_read_text_file({"path": str(target), "line": 2, "limit": True}, cwd) == {"content": "L2\nL3\nL4\nL5\n"}
    assert _fs_read_text_file({"path": str(target), "line": True, "limit": False}, cwd) == {"content": "L1\nL2\nL3\nL4\nL5\n"}

def test_acp_cwd_falls_back_from_root_and_non_writable_to_home_work(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    work_dir = fake_home / "work"
    work_dir.mkdir(parents=True)
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    ro_dir.chmod(0o555)

    log_path = tmp_path / "acp.log"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "basic")
    monkeypatch.setenv("FAKE_ACP_LOG", str(log_path))

    try:
        client_root = AntigravityACPClient(
            acp_command=sys.executable,
            acp_args=[FAKE_SERVER],
            acp_cwd="/",
            settings=Settings(state_db_path=tmp_path / "state1.db"),
        )
        assert client_root._acp_cwd == str(work_dir.resolve())
        client_root.create_chat_completion(messages=[{"role": "user", "content": "root init"}])

        client_ok = AntigravityACPClient(
            acp_command=sys.executable,
            acp_args=[FAKE_SERVER],
            acp_cwd=str(tmp_path),
            settings=Settings(state_db_path=tmp_path / "state2.db"),
        )
        client_ok.create_chat_completion(messages=[{"role": "user", "content": "root call"}], cwd="/")

        client_ro = AntigravityACPClient(
            acp_command=sys.executable,
            acp_args=[FAKE_SERVER],
            acp_cwd=str(ro_dir),
            settings=Settings(state_db_path=tmp_path / "state3.db"),
        )
        assert client_ro._acp_cwd == str(work_dir.resolve())
        client_ok.create_chat_completion(messages=[{"role": "user", "content": "ro call"}], cwd=str(ro_dir))

        session_news = [e for e in _read_log(log_path) if e.get("method") == "session/new"]
        assert len(session_news) == 3
        assert [e["params"]["cwd"] for e in session_news] == [str(work_dir.resolve())] * 3
    finally:
        ro_dir.chmod(0o755)


def test_acp_cwd_falls_back_to_home_when_work_missing_or_non_writable(tmp_path, monkeypatch):
    fake_home = tmp_path / "home_only"
    fake_home.mkdir()
    log_path = tmp_path / "acp.log"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "basic")
    monkeypatch.setenv("FAKE_ACP_LOG", str(log_path))

    client_missing_work = AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[FAKE_SERVER],
        acp_cwd="/",
        settings=Settings(state_db_path=tmp_path / "state1.db"),
    )
    assert client_missing_work._acp_cwd == str(fake_home.resolve())
    client_missing_work.create_chat_completion(messages=[{"role": "user", "content": "no work"}], cwd="/")

    work_dir = fake_home / "work"
    work_dir.mkdir()
    work_dir.chmod(0o555)
    try:
        monkeypatch.setitem(sys.modules, "agent.runtime_cwd", None)
        monkeypatch.chdir("/")
        client_ro_work = AntigravityACPClient(
            acp_command=sys.executable,
            acp_args=[FAKE_SERVER],
            settings=Settings(state_db_path=tmp_path / "state2.db"),
        )
        assert client_ro_work._acp_cwd == str(fake_home.resolve())
        client_ro_work.create_chat_completion(messages=[{"role": "user", "content": "ro work"}])

        session_news = [e for e in _read_log(log_path) if e.get("method") == "session/new"]
        assert len(session_news) == 2
        assert [e["params"]["cwd"] for e in session_news] == [str(fake_home.resolve())] * 2
    finally:
        work_dir.chmod(0o755)
