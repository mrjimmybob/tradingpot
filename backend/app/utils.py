"""Small shared utilities."""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Current UTC time as a naive datetime.

    Equivalent in value to the deprecated ``datetime.utcnow()`` but built from a
    timezone-aware instant, so it is explicit about being UTC and avoids the
    deprecation. Kept naive to stay consistent with the rest of the codebase's
    timestamp columns and comparisons.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def ms_to_utc(ms: float) -> datetime:
    """Convert epoch milliseconds to a naive UTC datetime.

    Exchange timestamps are epoch-ms. ``datetime.fromtimestamp(ms/1000)`` uses
    the LOCAL timezone, which would be inconsistent with the UTC times used
    elsewhere on any server not set to UTC. This always interprets the value as
    UTC.
    """
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None)
