from __future__ import annotations

from hermes_agy_acp.config import Settings
from hermes_agy_acp.protocol import (
    build_initialize_request,
    build_permission_approval_response,
    build_session_cancel_notification,
    build_session_set_config_option_request,
    build_session_set_mode_request,
    build_session_set_model_request,
)


def test_initialize_request_declares_fs_capabilities():
    req = build_initialize_request(1)
    assert req.method == "initialize"
    assert req.params["protocolVersion"] == 1
    assert req.params["clientCapabilities"] == {
        "fs": {"readTextFile": True, "writeTextFile": True}
    }
    assert req.params["clientInfo"]["name"]


def test_session_mode_config_model_and_cancel_builders():
    mode_req = build_session_set_mode_request(2, session_id="s1", mode_id="yolo")
    assert mode_req.method == "session/set_mode"
    assert mode_req.params == {"sessionId": "s1", "modeId": "yolo"}

    cfg_req = build_session_set_config_option_request(3, session_id="s1", config_id="thinking_level", value="high")
    assert cfg_req.method == "session/set_config_option"
    assert cfg_req.params == {"sessionId": "s1", "configId": "thinking_level", "value": "high"}

    mdl_req = build_session_set_model_request(4, session_id="s1", model_id="gemini-4-hydrogen")
    assert mdl_req.method == "session/set_model"
    assert mdl_req.params == {"sessionId": "s1", "modelId": "gemini-4-hydrogen"}

    cancel_notif = build_session_cancel_notification("s1")
    assert cancel_notif.method == "session/cancel"
    assert cancel_notif.params == {"sessionId": "s1"}


def test_permission_approval_response_uses_acp_v1_selected_outcome():
    options = [
        {"optionId": "reject_once", "kind": "reject_once"},
        {"optionId": "allow_once", "kind": "allow_once"},
        {"optionId": "allow_always", "kind": "allow_always"},
    ]
    resp = build_permission_approval_response(7, options=options)
    assert resp.id == 7
    assert resp.result == {"outcome": {"outcome": "selected", "optionId": "allow_always"}}

    resp_once = build_permission_approval_response(
        8, options=[{"optionId": "allow_once_id", "kind": "allow_once"}]
    )
    assert resp_once.result == {"outcome": {"outcome": "selected", "optionId": "allow_once_id"}}


def test_settings_defaults_yolo_timeout_900_and_empty_bootstrap(monkeypatch):
    monkeypatch.delenv("AGY_ACP_MODE", raising=False)
    monkeypatch.delenv("AGY_PROMPT_TIMEOUT", raising=False)
    monkeypatch.delenv("AGY_BOOTSTRAP_PROMPT", raising=False)
    settings = Settings()
    assert settings.agy_acp_mode == "yolo"
    assert settings.agy_prompt_timeout == 900.0
    assert settings.agy_bootstrap_prompt == ""
