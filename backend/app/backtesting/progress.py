"""Lightweight CLI progress reporting shared by the CSV loader and the
backtest replay loop.

Purely visual - nothing here reads or influences candle parsing, validation,
or backtest math (see TASK 3/5 in the progress-reporting change). Callers
pass pre-computed counters; this module never recomputes statistics of its
own, so driving it costs O(1) per update regardless of dataset size.
"""
from __future__ import annotations

import shutil
import sys

# Roughly how many progress updates a full run should produce end to end
# (Task 3: "about 50-100 updates total for a full run").
DEFAULT_UPDATE_STEPS = 80


def compute_update_every(total: int, steps: int = DEFAULT_UPDATE_STEPS) -> int:
    """How many units (rows/candles) should pass between progress updates."""
    return max(1, total // max(steps, 1))


def terminal_width(default: int = 80) -> int:
    return shutil.get_terminal_size(fallback=(default, 24)).columns


def render_bar(fraction: float, width: int) -> str:
    """Render a ``[####......] NN%`` bar sized to fit ``width`` columns,
    reserving space for the percentage suffix first."""
    fraction = max(0.0, min(1.0, fraction))
    pct = int(fraction * 100)
    suffix = f" {pct:3d}%"
    bar_width = max(5, width - len(suffix) - 2)  # -2 for the brackets
    filled = int(bar_width * fraction)
    bar = "#" * filled + "." * (bar_width - filled)
    return f"[{bar}]{suffix}"


class ProgressBar:
    """In-place, single-line progress bar. A no-op when ``quiet``.

    Only writes to stdout when the rendered text actually changed, so a
    caller can call ``update`` on every iteration without worrying about
    flooding the terminal - the check itself is O(1) string comparison.
    """

    def __init__(self, total: int, quiet: bool = False):
        self.total = max(total, 1)
        self.quiet = quiet
        self._last_rendered: str | None = None

    def update(self, current: int) -> None:
        if self.quiet:
            return
        text = render_bar(current / self.total, terminal_width())
        if text == self._last_rendered:
            return
        self._last_rendered = text
        sys.stdout.write("\r" + text)
        sys.stdout.flush()

    def finish(self) -> None:
        if self.quiet:
            return
        sys.stdout.write("\n")
        sys.stdout.flush()
