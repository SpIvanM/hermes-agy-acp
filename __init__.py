"""Hermes Agent plugin entrypoint for hermes-agy-acp."""

import sys
from pathlib import Path

# Add src to sys.path so hermes_agy_acp package is importable
_src = str(Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import hermes_agy_acp
from hermes_agy_acp import *

# Attempt native Hermes provider registration if running under Hermes runtime
try:
    from providers import register_provider
    from providers.base import ProviderProfile as HermesProviderProfile

    _profile = HermesProviderProfile(
        name="antigravity-acp",
        aliases=("agy-acp", "antigravity"),
        api_mode="custom",
        auth_type="external_process",
        supports_streaming=True,
    )
    register_provider(_profile)
except Exception:
    pass
