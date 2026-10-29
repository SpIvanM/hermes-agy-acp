from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from hermes_agy_acp.client import AntigravityACPClient
from hermes_agy_acp.config import Settings, get_settings
from hermes_agy_acp.process_pool import ACPProcessSupervisor
from hermes_agy_acp.session_map import SessionRegistry
from hermes_agy_acp.skill_learning_agy import AgySkillLearner


class ProviderProfile(BaseModel):
    """Hermes model-provider plugin profile descriptor."""

    name: str = "antigravity-acp"
    auth_type: str = "external_process"
    marker_base_url: str = "acp://antigravity"
    supports_streaming: bool = True


def get_provider_profile() -> ProviderProfile:
    """Return the provider profile for Antigravity ACP."""
    return ProviderProfile()


_shared_supervisor: ACPProcessSupervisor | None = None
_shared_registry: SessionRegistry | None = None


def get_shared_supervisor() -> ACPProcessSupervisor:
    global _shared_supervisor
    if _shared_supervisor is None or not _shared_supervisor.is_alive():
        _shared_supervisor = ACPProcessSupervisor()
    return _shared_supervisor


def get_shared_registry() -> SessionRegistry:
    global _shared_registry
    if _shared_registry is None:
        _shared_registry = SessionRegistry()
    return _shared_registry


def create_client(**client_kwargs: Any) -> AntigravityACPClient:
    """Factory function creating an Antigravity ACP client sharing warm process pool."""
    return AntigravityACPClient(
        supervisor=get_shared_supervisor(),
        registry=get_shared_registry(),
    )


def register(registry: Any = None) -> ProviderProfile:
    """Register provider profile with Hermes plugin registry."""
    profile = get_provider_profile()
    if registry is not None and hasattr(registry, "register_provider"):
        registry.register_provider(profile)
    return profile


def diagnostics_status() -> dict[str, Any]:
    """Inspect and report status of Antigravity ACP integration."""
    settings = get_settings()
    cli = settings.get_resolved_cli_path()
    server = settings.get_resolved_server_path()
    registry = SessionRegistry()
    sessions = registry.list_sessions()
    registry.close()

    status = "healthy" if (cli and server) else "degraded"
    return {
        "status": status,
        "cli_path": str(cli) if cli else None,
        "server_path": str(server) if server else None,
        "default_model": settings.agy_default_model,
        "active_sessions": len(sessions),
    }


def diagnostics_reset(session_id: str) -> str:
    """Reset session mapping for diagnostic /reset command."""
    registry = SessionRegistry()
    success = registry.reset_session(session_id)
    registry.close()
    if success:
        return f"Successfully cleared ACP session mapping for '{session_id}'"
    return f"No active ACP session mapping found for '{session_id}'"


__all__ = [
    "ACPProcessSupervisor",
    "AgySkillLearner",
    "AntigravityACPClient",
    "ProviderProfile",
    "SessionRegistry",
    "Settings",
    "create_client",
    "diagnostics_reset",
    "diagnostics_status",
    "get_provider_profile",
    "get_settings",
    "register",
]
