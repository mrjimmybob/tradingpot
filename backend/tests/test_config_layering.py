"""Tests for layered configuration management.

Covers the precedence chain (config.yaml < profile < env files < process env),
CWD-independent profile/path resolution, env-file loading + gap-fill semantics,
typed env-variable overrides, and frontend build auto-discovery.

These are configuration-only behaviours; trading/accounting/persistence are
untouched.
"""

import textwrap
from pathlib import Path

import pytest

from app.services.config import ConfigService, ConfigValidationException


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_env():
    """Isolate env between tests.

    Env-file loading writes directly to ``os.environ`` (via ``setdefault``),
    which monkeypatch does not track. Snapshot the relevant prefixes, start each
    test from a clean slate, and fully restore afterwards so nothing leaks into
    other tests (e.g. a stray TRADINGBOT_API_TOKEN turning later API calls 401).
    """
    import os

    prefixes = ("TRADINGBOT_", "MEXC_")
    saved = {k: v for k, v in os.environ.items() if k.startswith(prefixes)}
    for key in list(saved):
        del os.environ[key]
    try:
        yield
    finally:
        for key in [k for k in os.environ if k.startswith(prefixes)]:
            del os.environ[key]
        os.environ.update(saved)


def _write(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body))


@pytest.fixture
def base_config(tmp_path):
    """A minimal valid base config.yaml; returns its path."""
    cfg = tmp_path / "config.yaml"
    _write(cfg, """
        server:
          host: "127.0.0.1"
          port: 8000
        logging:
          level: "INFO"
    """)
    return cfg


# ---------------------------------------------------------------------------
# 1. Precedence / layering
# ---------------------------------------------------------------------------

class TestPrecedence:
    def test_base_only(self, base_config):
        cs = ConfigService(str(base_config))
        cs.load_and_validate()
        assert cs.get("server.port") == 8000
        assert cs.get("server.host") == "127.0.0.1"
        assert cs.get("logging.level") == "INFO"

    def test_null_section_is_treated_as_empty(self, base_config):
        """A section header with all child keys commented out parses as YAML
        null; it must validate (all properties optional), not crash startup.

        Regression: a `market_data:` section with everything commented produced
        `market_data: Expected dict, got NoneType` and `sys.exit(1)` on boot.
        """
        with open(base_config, "a") as f:
            f.write("\nmarket_data:\n")  # null section, like all-commented keys
        cs = ConfigService(str(base_config))
        cs.load_and_validate()  # must not raise
        assert cs.get("market_data.source") is None  # absent -> default downstream
        assert cs.get("server.port") == 8000          # rest of config intact

    def test_shipped_config_yaml_validates(self):
        """The actual shipped backend/config.yaml must pass validation.

        Guards against the gap that let `market_data: null` reach production: a
        config edit that boots fine in synthetic tests but crashes the REAL app
        at startup with a ConfigValidationException -> sys.exit(1) -> failed
        deploy. This validates the real file the server boots from.
        """
        real = Path(__file__).resolve().parent.parent / "config.yaml"
        cs = ConfigService(str(real))
        cs.load_and_validate()  # must not raise

    def test_profile_overrides_base(self, base_config, monkeypatch):
        _write(base_config.parent / "config.production.yaml", """
            server:
              port: 9000
            logging:
              level: "WARNING"
        """)
        monkeypatch.setenv("TRADINGBOT_PROFILE", "production")
        cs = ConfigService(str(base_config))
        cs.load_and_validate()
        # Overridden by profile...
        assert cs.get("server.port") == 9000
        assert cs.get("logging.level") == "WARNING"
        # ...while untouched base keys survive (deep merge).
        assert cs.get("server.host") == "127.0.0.1"

    def test_env_file_overrides_profile(self, base_config, tmp_path, monkeypatch):
        _write(base_config.parent / "config.production.yaml", """
            logging:
              level: "WARNING"
        """)
        env_file = tmp_path / "secrets.env"
        _write(env_file, """
            # comment line ignored
            export TRADINGBOT__LOGGING__LEVEL=ERROR
            TRADINGBOT__SERVER__HOST=0.0.0.0
        """)
        monkeypatch.setenv("TRADINGBOT_PROFILE", "production")
        monkeypatch.setenv("TRADINGBOT_ENV_FILE", str(env_file))
        cs = ConfigService(str(base_config))
        cs.load_and_validate()
        assert cs.get("logging.level") == "ERROR"   # env file beat profile
        assert cs.get("server.host") == "0.0.0.0"

    def test_process_env_beats_env_file(self, base_config, tmp_path, monkeypatch):
        env_file = tmp_path / "secrets.env"
        _write(env_file, "TRADINGBOT__LOGGING__LEVEL=ERROR\n")
        # Process env already sets the same key -> must win over the file value.
        monkeypatch.setenv("TRADINGBOT__LOGGING__LEVEL", "DEBUG")
        monkeypatch.setenv("TRADINGBOT_ENV_FILE", str(env_file))
        cs = ConfigService(str(base_config))
        cs.load_and_validate()
        assert cs.get("logging.level") == "DEBUG"


# ---------------------------------------------------------------------------
# 2. Profile selection / resolution
# ---------------------------------------------------------------------------

class TestProfile:
    def test_missing_profile_warns_and_continues(self, base_config, monkeypatch, caplog):
        monkeypatch.setenv("TRADINGBOT_PROFILE", "nope")
        cs = ConfigService(str(base_config))
        cs.load_and_validate()  # must not raise
        assert cs.get("server.port") == 8000
        assert any("profile 'nope'" in r.message for r in caplog.records)

    def test_profile_resolved_independent_of_cwd(self, base_config, monkeypatch, tmp_path):
        _write(base_config.parent / "config.staging.yaml", "server:\n  port: 1234\n")
        monkeypatch.setenv("TRADINGBOT_PROFILE", "staging")
        # Run from an unrelated working directory.
        other = tmp_path / "elsewhere"
        other.mkdir()
        monkeypatch.chdir(other)
        cs = ConfigService(str(base_config))
        cs.load_and_validate()
        assert cs.get("server.port") == 1234


# ---------------------------------------------------------------------------
# 3. Env-file loading semantics
# ---------------------------------------------------------------------------

class TestEnvFiles:
    def test_env_file_is_optional(self, base_config, monkeypatch):
        monkeypatch.setenv("TRADINGBOT_ENV_FILE", "/nonexistent/path.env")
        cs = ConfigService(str(base_config))
        cs.load_and_validate()  # must not raise
        assert cs.get("server.port") == 8000

    def test_env_file_populates_process_env_gap_fill(self, base_config, tmp_path, monkeypatch):
        import os

        env_file = tmp_path / "secrets.env"
        _write(env_file, """
            MEXC_API_KEY=from-file
            TRADINGBOT_API_TOKEN=tok-from-file
        """)
        monkeypatch.setenv("TRADINGBOT_ENV_FILE", str(env_file))
        # Pre-existing process var must NOT be overwritten by the file.
        monkeypatch.setenv("MEXC_API_KEY", "from-process")
        ConfigService(str(base_config)).load_and_validate()
        assert os.environ["MEXC_API_KEY"] == "from-process"   # gap-fill respected
        assert os.environ["TRADINGBOT_API_TOKEN"] == "tok-from-file"


# ---------------------------------------------------------------------------
# 4. Typed env-variable overrides
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    def test_int_coercion(self, base_config, monkeypatch):
        monkeypatch.setenv("TRADINGBOT__SERVER__PORT", "7000")
        cs = ConfigService(str(base_config))
        cs.load_and_validate()
        assert cs.get("server.port") == 7000
        assert isinstance(cs.get("server.port"), int)

    def test_bool_and_list_coercion(self, base_config, monkeypatch):
        monkeypatch.setenv("TRADINGBOT__SERVER__DEBUG", "true")
        monkeypatch.setenv(
            "TRADINGBOT__SERVER__CORS_ORIGINS",
            "http://a.test, http://b.test",
        )
        cs = ConfigService(str(base_config))
        cs.load_and_validate()
        assert cs.get("server.debug") is True
        assert cs.get("server.cors_origins") == ["http://a.test", "http://b.test"]

    def test_nested_single_underscore_key_preserved(self, base_config, monkeypatch):
        # frontend_dist contains a single underscore; only `__` nests.
        monkeypatch.setenv("TRADINGBOT__SERVER__FRONTEND_DIST", "/srv/ui")
        cs = ConfigService(str(base_config))
        cs.load_and_validate()
        assert cs.get("server.frontend_dist") == "/srv/ui"

    def test_bad_coercion_raises(self, base_config, monkeypatch):
        monkeypatch.setenv("TRADINGBOT__SERVER__PORT", "not-an-int")
        with pytest.raises(ConfigValidationException) as exc:
            ConfigService(str(base_config)).load_and_validate()
        assert "could not be coerced" in str(exc.value)

    def test_unknown_override_key_rejected(self, base_config, monkeypatch):
        monkeypatch.setenv("TRADINGBOT__SERVER__BOGUS", "x")
        with pytest.raises(ConfigValidationException):
            ConfigService(str(base_config)).load_and_validate()

    def test_legacy_single_underscore_vars_are_not_overrides(self, base_config, monkeypatch):
        # TRADINGBOT_API_TOKEN must not be parsed as a generic override and must
        # not crash validation (it is handled by get_api_token, not the schema).
        monkeypatch.setenv("TRADINGBOT_API_TOKEN", "secret")
        cs = ConfigService(str(base_config))
        cs.load_and_validate()  # must not raise
        assert cs.get("server.port") == 8000


# ---------------------------------------------------------------------------
# 5. Frontend auto-discovery
# ---------------------------------------------------------------------------

class TestFrontendDiscovery:
    def _make_build(self, root: Path) -> Path:
        dist = root / "frontend" / "dist"
        dist.mkdir(parents=True)
        (dist / "index.html").write_text("<!doctype html>")
        return dist

    def test_autodiscovers_default_build(self, tmp_path, monkeypatch):
        from app import main as main_mod

        dist = self._make_build(tmp_path)
        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        monkeypatch.setattr(main_mod.config_service, "get", lambda *a, **k: None)
        resolved = main_mod._resolve_frontend_dist(backend_dir=backend_dir)
        assert resolved == dist

    def test_missing_build_returns_none(self, tmp_path, monkeypatch, capsys):
        from app import main as main_mod

        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        monkeypatch.setattr(main_mod.config_service, "get", lambda *a, **k: None)
        assert main_mod._resolve_frontend_dist(backend_dir=backend_dir) is None
        assert "serving API only" in capsys.readouterr().out

    def test_configured_path_overrides_autodiscovery(self, tmp_path, monkeypatch):
        from app import main as main_mod

        custom = tmp_path / "custom-ui"
        custom.mkdir()
        (custom / "index.html").write_text("<!doctype html>")
        monkeypatch.setattr(
            main_mod.config_service, "get",
            lambda key, default=None: str(custom) if key == "server.frontend_dist" else default,
        )
        resolved = main_mod._resolve_frontend_dist(backend_dir=tmp_path / "backend")
        assert resolved == custom
