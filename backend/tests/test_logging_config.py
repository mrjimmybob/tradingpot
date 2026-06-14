"""Regression test for application log visibility.

Under uvicorn nothing configures logging for the ``app.*`` modules, so Python's
last-resort handler drops everything below WARNING — which silently hid every
``logger.info`` (including all strategy-evaluation logs) in production. The app
must configure the ``app`` logger so INFO records are emitted.
"""

import io
import logging

from app.main import _configure_logging


def _snapshot_app_logger():
    lg = logging.getLogger("app")
    return (lg.level, list(lg.handlers), lg.propagate)


def _restore_app_logger(state):
    lg = logging.getLogger("app")
    lg.setLevel(state[0])
    lg.handlers = state[1]
    lg.propagate = state[2]


def test_configure_logging_emits_app_info():
    saved = _snapshot_app_logger()
    try:
        logging.getLogger("app").handlers = []  # deterministic start (other tests leak)
        _configure_logging()
        app_logger = logging.getLogger("app")

        # INFO must not be suppressed for app.* loggers anymore.
        assert app_logger.level <= logging.INFO
        assert any(isinstance(h, logging.StreamHandler) for h in app_logger.handlers)

        # Functional: an app.* INFO record actually reaches a handler.
        buf = io.StringIO()
        probe = logging.StreamHandler(buf)
        probe.setLevel(logging.INFO)
        app_logger.addHandler(probe)
        try:
            logging.getLogger("app.services.trading_engine").info("eval-visible-marker")
        finally:
            app_logger.removeHandler(probe)
        assert "eval-visible-marker" in buf.getvalue()
    finally:
        _restore_app_logger(saved)


def test_configure_logging_is_idempotent():
    """Calling it twice must not stack duplicate handlers."""
    saved = _snapshot_app_logger()
    try:
        logging.getLogger("app").handlers = []  # deterministic start (other tests leak)
        _configure_logging()
        n1 = len([h for h in logging.getLogger("app").handlers
                  if isinstance(h, logging.StreamHandler)])
        _configure_logging()
        n2 = len([h for h in logging.getLogger("app").handlers
                  if isinstance(h, logging.StreamHandler)])
        assert n1 == n2 == 1
    finally:
        _restore_app_logger(saved)
