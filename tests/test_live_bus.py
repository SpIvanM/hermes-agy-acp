from __future__ import annotations

import importlib.util
import json
import sys
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from hermes_agy_acp.live_bus import (
    LiveStatusEvent,
    collect_child_session_events,
    publish,
    redact_args,
    redact_secrets,
    subscribe,
)


def test_singleton_bus_registered_in_sys_modules(tmp_path):
    assert "_hermes_live_status_bus" in sys.modules
    bus = sys.modules["_hermes_live_status_bus"]

    # Simulate loading live_bus via a different module name (plugin dynamic load)
    live_bus_path = Path(__file__).resolve().parents[1] / "src" / "hermes_agy_acp" / "live_bus.py"
    spec = importlib.util.spec_from_file_location("alt_plugin_ns.live_bus", live_bus_path)
    assert spec is not None and spec.loader is not None
    alt_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alt_mod)

    received: list[LiveStatusEvent] = []
    unsub = alt_mod.subscribe("*", received.append)
    try:
        ev = LiveStatusEvent(
            turn_id="turn-singleton-1",
            kind="tool_start",
            agent="Orchestrator",
            tool_name="read_file",
            args_summary="README.md",
            call_id="call-1",
            status="in_progress",
            timestamp=123.4,
        )
        publish(ev)
        assert len(received) == 1
        assert received[0].tool_name == "read_file"
        assert sys.modules["_hermes_live_status_bus"] is bus
    finally:
        unsub()


def test_live_status_event_is_immutable_and_filtered_by_turn_id():
    ev = LiveStatusEvent(turn_id="t1", kind="thinking", agent="Orchestrator")
    with pytest.raises(FrozenInstanceError):
        ev.agent = "Other"  # type: ignore[misc]

    all_events: list[LiveStatusEvent] = []
    t1_events: list[LiveStatusEvent] = []
    t2_events: list[LiveStatusEvent] = []

    unsub_all = subscribe("all", all_events.append)
    unsub_t1 = subscribe("t1", t1_events.append)
    unsub_t2 = subscribe("t2", t2_events.append)
    try:
        publish(LiveStatusEvent(turn_id="t1", kind="thinking", agent="Orchestrator"))
        publish(LiveStatusEvent(turn_id="t2", kind="tool_start", agent="Dev", tool_name="pytest"))
        assert len(all_events) == 2
        assert len(t1_events) == 1 and t1_events[0].turn_id == "t1"
        assert len(t2_events) == 1 and t2_events[0].agent == "Dev"

        unsub_t1()
        unsub_t1()  # idempotent unsubscribe
        publish(LiveStatusEvent(turn_id="t1", kind="turn_end", agent="Orchestrator", status="completed"))
        assert len(t1_events) == 1
        assert len(all_events) == 3
    finally:
        unsub_all()
        unsub_t1()
        unsub_t2()


def test_thread_safe_concurrent_publish_and_subscriber_isolation():
    received: list[LiveStatusEvent] = []
    lock = threading.Lock()

    def _bad_cb(_ev: LiveStatusEvent) -> None:
        raise RuntimeError("subscriber boom")

    def _good_cb(ev: LiveStatusEvent) -> None:
        with lock:
            received.append(ev)

    u1 = subscribe("concurrent-turn", _bad_cb)
    u2 = subscribe("concurrent-turn", _good_cb)
    try:
        def _worker(idx: int) -> None:
            for j in range(10):
                publish(
                    LiveStatusEvent(
                        turn_id="concurrent-turn",
                        kind="tool_update",
                        agent="Dev",
                        tool_name="run_command",
                        call_id=f"{idx}-{j}",
                    )
                )

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(received) == 40
    finally:
        u1()
        u2()


def test_redact_secrets_and_redact_args():
    raw = (
        'curl -H "Authorization: Bearer fake-jwt-abc123456" '
        "--token fake_cli_token_987 "
        "BOT_TOKEN=123456789:FAKE_TELEGRAM_BOT_TOKEN_XYZ123 "
        "sk-fakeproj-1234567890abcdef "
        "ghp_FakeGitHubToken1234567890 "
        "AIzaFakeGoogleApiKey1234567890\nnext line"
    )
    masked = redact_secrets(raw)
    for secret in (
        "fake-jwt-abc123456",
        "fake_cli_token_987",
        "123456789:FAKE_TELEGRAM_BOT_TOKEN_XYZ123",
        "sk-fakeproj-1234567890abcdef",
        "ghp_FakeGitHubToken1234567890",
        "AIzaFakeGoogleApiKey1234567890",
    ):
        assert secret not in masked
    assert "[REDACTED]" in masked

    compact = redact_args(raw, max_len=80)
    assert "\n" not in compact
    assert len(compact) <= 80


def test_collect_child_session_events_readonly_and_chmod_000(tmp_path):
    sessions_dir = tmp_path / "acpx_sessions"
    sessions_dir.mkdir()
    locked_dir = tmp_path / "locked_sessions"
    locked_dir.mkdir()
    locked_dir.chmod(0o000)

    session_file = sessions_dir / "researcher.jsonl"
    records = [
        {
            "agent": "Researcher",
            "tool": "read_file",
            "args": "spec.md --token fake_secret_tok_123",
            "status": "completed",
            "call_id": "sub-1",
            "timestamp": 100.0,
        },
        {
            "command": 'agy -p "[P:15ee R:Dev] pytest tests/"',
            "status": "in_progress",
            "timestamp": 101.0,
        },
    ]
    session_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    try:
        events = collect_child_session_events(
            session_dirs=[locked_dir, sessions_dir],
            since_ts=50.0,
            turn_id="turn-child-1",
        )
        assert len(events) == 2
        assert events[0].agent == "Researcher"
        assert events[0].tool_name == "read_file"
        assert "fake_secret_tok_123" not in events[0].args_summary
        assert "[REDACTED]" in events[0].args_summary
        assert events[0].kind == "tool_done"
        assert events[1].agent == "Dev"
        assert events[1].kind == "tool_start"
    finally:
        locked_dir.chmod(0o755)
