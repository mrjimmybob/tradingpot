"""Phase 2 — Auto Mode committee feature flag.

The committee execution path coexists with the existing Standalone Adapter
path rather than replacing it: it is OFF by default, so nothing changes until
a bot or the deployment explicitly opts in. Enabling it does not remove or
alter the Standalone path.

Resolution order (first match wins):
  1. per-bot override: `bot.strategy_params["auto_committee_enabled"]`, else a
     `bot.auto_committee_enabled` attribute if present;
  2. global: the `AUTO_COMMITTEE_ENABLED` environment variable;
  3. default: disabled.
"""
from __future__ import annotations

import os
from typing import Any, Optional

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def is_committee_enabled(bot: Optional[Any] = None) -> bool:
    """Return True iff the Auto Mode committee execution path is enabled."""
    if bot is not None:
        params = getattr(bot, "strategy_params", None)
        if isinstance(params, dict) and "auto_committee_enabled" in params:
            resolved = _as_bool(params["auto_committee_enabled"])
            if resolved is not None:
                return resolved
        override = _as_bool(getattr(bot, "auto_committee_enabled", None))
        if override is not None:
            return override

    return os.getenv("AUTO_COMMITTEE_ENABLED", "").strip().lower() in _TRUTHY
