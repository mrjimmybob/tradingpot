"""Configuration management and validation service."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ConfigValidationError:
    """Represents a configuration validation error."""
    path: str
    message: str


class ConfigValidationException(Exception):
    """Raised when config validation fails."""

    def __init__(self, errors: List[ConfigValidationError]):
        self.errors = errors
        messages = [f"{e.path}: {e.message}" for e in errors]
        super().__init__("Configuration validation failed:\n" + "\n".join(messages))


# Configuration schema definition
CONFIG_SCHEMA = {
    "server": {
        "type": "dict",
        "required": False,
        "properties": {
            "host": {"type": "str", "required": False},
            "port": {"type": "int", "required": False, "min": 1, "max": 65535},
            "debug": {"type": "bool", "required": False},
            "api_token": {"type": "str", "required": False},
            "cors_origins": {"type": "list", "required": False},
            "frontend_dist": {"type": "str", "required": False},
        }
    },
    "database": {
        "type": "dict",
        "required": False,
        "properties": {
            "url": {"type": "str", "required": False},
        }
    },
    "market_data": {
        "type": "dict",
        "required": False,
        "properties": {
            # UI live-price/indicator feed source. "rest" (default) polls
            # fetch_ticker; "websocket" uses the native MEXC stream connector
            # (only where MEXC permits streams from the host IP).
            "source": {"type": "str", "required": False, "options": ["rest", "websocket"]},
            "rest_poll_interval_seconds": {"type": "float", "required": False, "min": 0.5},
        }
    },
    "trading": {
        "type": "dict",
        "required": False,
        "properties": {
            "execution_interval_seconds": {"type": "int", "required": False, "min": 1},
            "pnl_snapshot_interval_seconds": {"type": "int", "required": False, "min": 1},
            "reconciliation_interval_seconds": {"type": "int", "required": False, "min": 1},
            "default_stop_loss_percent": {"type": "float", "required": False, "min": 0, "max": 100},
            "default_daily_loss_limit": {"type": "float", "required": False, "min": 0},
            "max_consecutive_failures": {"type": "int", "required": False, "min": 1},
            "failure_backoff_max_seconds": {"type": "float", "required": False, "min": 1},
            "state_checkpoint_seconds": {"type": "float", "required": False, "min": 1},
            "backup_interval_seconds": {"type": "float", "required": False, "min": 1},
            "ticker_cache_ttl_seconds": {"type": "float", "required": False, "min": 0},
        }
    },
    "exchange": {
        "type": "dict",
        "required": False,
        "properties": {
            "api_key": {"type": "str", "required": False},
            "api_secret": {"type": "str", "required": False},
            "sandbox_mode": {"type": "bool", "required": False},
        }
    },
    "email": {
        "type": "dict",
        "required": False,
        "properties": {
            "enabled": {"type": "bool", "required": False},
            "smtp_host": {"type": "str", "required": False},
            "smtp_port": {"type": "int", "required": False, "min": 1, "max": 65535},
            "smtp_user": {"type": "str", "required": False},
            "smtp_password": {"type": "str", "required": False},
            "from_address": {"type": "str", "required": False},
            "to_address": {"type": "str", "required": False},
        }
    },
    "logging": {
        "type": "dict",
        "required": False,
        "properties": {
            "level": {"type": "str", "required": False, "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
            "format": {"type": "str", "required": False},
        }
    },
}


# --- Configuration layering: environment-variable conventions ---------------
# Single-underscore names below are *special-cased* selectors (not generic
# config overrides) and are kept distinct from the generic override prefix.
PROFILE_ENV_VAR = "TRADINGBOT_PROFILE"        # selects config.<profile>.yaml
ENV_FILE_ENV_VAR = "TRADINGBOT_ENV_FILE"      # extra env file to load (optional)

# Generic "env var -> config key" overrides. A double underscore separates
# nested keys, e.g. TRADINGBOT__SERVER__FRONTEND_DIST -> server.frontend_dist.
# Double-underscore nesting keeps single-underscore key segments intact
# (e.g. frontend_dist, execution_interval_seconds) and avoids collision with
# the single-underscore selectors above and legacy names like
# TRADINGBOT_API_TOKEN / TRADINGBOT_DATABASE_URL.
ENV_OVERRIDE_PREFIX = "TRADINGBOT__"
ENV_KEY_SEPARATOR = "__"


def _leaf_schema_for_path(path: List[str]) -> Optional[Dict[str, Any]]:
    """Return the schema node for a dotted config path, or None if unknown.

    Used to type-coerce environment-variable overrides (which are always
    strings) into the type the schema expects.
    """
    if not path:
        return None
    schema: Any = CONFIG_SCHEMA.get(path[0])
    if not isinstance(schema, dict):
        return None
    for seg in path[1:]:
        props = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(props, dict) or seg not in props:
            return None
        schema = props[seg]
    return schema if isinstance(schema, dict) else None


class ConfigService:
    """Service for loading and validating configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config service.

        Args:
            config_path: Path to config file. If None, uses default location.
        """
        if config_path is None:
            # Default config path relative to backend directory
            backend_dir = Path(__file__).parent.parent.parent
            config_path = str(backend_dir / "config.yaml")

        self.config_path = config_path
        self._config: Dict[str, Any] = {}

    @property
    def _base_dir(self) -> Path:
        """Directory holding the base config file (CWD-independent anchor)."""
        return Path(self.config_path).resolve().parent

    def load_and_validate(self) -> Dict[str, Any]:
        """Load, layer, and validate configuration.

        Precedence (lowest to highest):
          1. config.yaml (the base file)
          2. optional profile file config.<profile>.yaml (TRADINGBOT_PROFILE)
          3. environment files (loaded into the process environment)
          4. process environment variables (TRADINGBOT__SECTION__KEY overrides)

        Higher-precedence sources override lower ones. The merged result is
        validated against CONFIG_SCHEMA.

        Returns:
            Validated configuration dictionary.

        Raises:
            ConfigValidationException: If validation fails.
        """
        errors: List[ConfigValidationError] = []

        # Layer 1: base config.yaml (optional; absent -> empty base + warning).
        config = self._load_yaml_file(self.config_path, required=False, errors=errors)

        # Layer 2: optional profile, selected by env var, resolved next to the
        # base config file so it is independent of the current directory.
        profile = os.environ.get(PROFILE_ENV_VAR, "").strip()
        if profile:
            profile_path = self._profile_path(profile)
            if profile_path.is_file():
                profile_cfg = self._load_yaml_file(
                    str(profile_path), required=True, errors=errors
                )
                config = self._deep_merge(config, profile_cfg)
                logger.info(f"Loaded configuration profile '{profile}' from {profile_path}")
            else:
                logger.warning(
                    f"Configuration profile '{profile}' selected "
                    f"({PROFILE_ENV_VAR}={profile}) but no file found at "
                    f"{profile_path}; continuing without it"
                )

        # Layer 3: environment files -> process environment (gap-fill only, so
        # already-set process variables keep precedence over file values).
        self._load_env_files()

        # Layer 4: process environment variable overrides (highest precedence).
        env_cfg = self._env_overrides(errors)
        config = self._deep_merge(config, env_cfg)

        # Surface load/coercion problems before schema validation.
        if errors:
            raise ConfigValidationException(errors)

        # Validate the fully merged configuration.
        validation_errors = self._validate_dict(config, CONFIG_SCHEMA, "")
        if validation_errors:
            raise ConfigValidationException(validation_errors)

        self._config = config
        logger.info(f"Configuration loaded and validated (base={self.config_path})")
        return config

    # --- Layering helpers --------------------------------------------------

    def _load_yaml_file(
        self, path: str, required: bool, errors: List[ConfigValidationError]
    ) -> Dict[str, Any]:
        """Load one YAML file into a dict, recording problems in ``errors``.

        A missing optional file yields ``{}`` with a warning (matching the
        historical base-config behaviour). YAML/shape problems are recorded as
        validation errors rather than raised here, so all sources can be
        reported together.
        """
        if not os.path.exists(path):
            if required:
                errors.append(ConfigValidationError(
                    path="", message=f"Config file not found: {path}"
                ))
            else:
                logger.warning(f"Config file not found at {path}, using defaults")
            return {}

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            errors.append(ConfigValidationError(
                path="", message=f"Invalid YAML syntax in {path}: {e}"
            ))
            return {}

        if data is None:
            return {}
        if not isinstance(data, dict):
            errors.append(ConfigValidationError(
                path="",
                message=f"Config in {path} must be a dictionary, got {type(data).__name__}",
            ))
            return {}
        return data

    def _profile_path(self, profile: str) -> Path:
        """Resolve a profile name to config.<profile>.yaml next to the base file."""
        return self._base_dir / f"config.{profile}.yaml"

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge ``override`` onto ``base`` (override wins)."""
        result = dict(base)
        for key, value in override.items():
            existing = result.get(key)
            if isinstance(value, dict) and isinstance(existing, dict):
                result[key] = ConfigService._deep_merge(existing, value)
            else:
                result[key] = value
        return result

    def _env_file_candidates(self) -> List[Path]:
        """Env files to load, in priority order (first wins on conflicts).

        - ``TRADINGBOT_ENV_FILE`` (explicit, optional)
        - ``<repo>/deploy/tradingbot.env`` (project convention; untracked)
        """
        candidates: List[Path] = []
        explicit = os.environ.get(ENV_FILE_ENV_VAR, "").strip()
        if explicit:
            p = Path(explicit)
            candidates.append(p if p.is_absolute() else (self._base_dir / p))
        candidates.append(self._base_dir.parent / "deploy" / "tradingbot.env")
        return candidates

    def _load_env_files(self) -> None:
        """Load KEY=VALUE pairs from env files into ``os.environ`` (gap-fill).

        Optional: missing files are skipped silently. Process environment
        variables already set are never overwritten, so they take precedence
        over file-loaded values.
        """
        for path in self._env_file_candidates():
            try:
                if not path.is_file():
                    continue
                for raw in path.read_text().splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):].strip()
                    if "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    if not key:
                        continue
                    val = val.strip().strip('"').strip("'")
                    # Process env wins -> only fill if not already present.
                    os.environ.setdefault(key, val)
            except OSError as e:
                logger.warning(f"Could not read env file {path}: {e}")

    def _env_overrides(
        self, errors: List[ConfigValidationError]
    ) -> Dict[str, Any]:
        """Build a config overlay from TRADINGBOT__SECTION__KEY env variables."""
        overrides: Dict[str, Any] = {}
        for raw_key, raw_val in os.environ.items():
            if not raw_key.startswith(ENV_OVERRIDE_PREFIX):
                continue
            suffix = raw_key[len(ENV_OVERRIDE_PREFIX):]
            path = [seg.lower() for seg in suffix.split(ENV_KEY_SEPARATOR) if seg]
            if not path:
                continue
            leaf = _leaf_schema_for_path(path)
            type_name = leaf.get("type") if isinstance(leaf, dict) else None
            try:
                value = self._coerce_env_value(raw_val, type_name)
            except (ValueError, TypeError):
                errors.append(ConfigValidationError(
                    path=".".join(path),
                    message=(
                        f"Environment override {raw_key}={raw_val!r} could not "
                        f"be coerced to {type_name}"
                    ),
                ))
                continue
            self._set_nested(overrides, path, value)
        return overrides

    @staticmethod
    def _coerce_env_value(value: str, type_name: Optional[str]) -> Any:
        """Coerce a string env value to the schema type (str if unknown)."""
        if type_name == "int":
            return int(value)
        if type_name == "float":
            return float(value)
        if type_name == "bool":
            return value.strip().lower() in ("1", "true", "yes", "on")
        if type_name == "list":
            return [s.strip() for s in value.split(",") if s.strip()]
        return value

    @staticmethod
    def _set_nested(target: Dict[str, Any], path: List[str], value: Any) -> None:
        """Set ``value`` at a nested ``path`` inside ``target``, creating dicts."""
        node = target
        for seg in path[:-1]:
            nxt = node.get(seg)
            if not isinstance(nxt, dict):
                nxt = {}
                node[seg] = nxt
            node = nxt
        node[path[-1]] = value

    def _validate_dict(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
        path: str
    ) -> List[ConfigValidationError]:
        """Validate a dictionary against schema.

        Args:
            data: Data to validate
            schema: Schema to validate against
            path: Current path for error messages

        Returns:
            List of validation errors
        """
        errors = []

        # Check for unknown keys
        for key in data:
            if key not in schema:
                errors.append(ConfigValidationError(
                    path=f"{path}.{key}" if path else key,
                    message=f"Unknown configuration key '{key}'"
                ))

        # Validate each schema property
        for key, prop_schema in schema.items():
            current_path = f"{path}.{key}" if path else key

            if key not in data:
                if prop_schema.get("required", False):
                    errors.append(ConfigValidationError(
                        path=current_path,
                        message="Required field missing"
                    ))
                continue

            value = data[key]
            errors.extend(self._validate_value(value, prop_schema, current_path))

        return errors

    def _validate_value(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str
    ) -> List[ConfigValidationError]:
        """Validate a single value against schema.

        Args:
            value: Value to validate
            schema: Schema to validate against
            path: Current path for error messages

        Returns:
            List of validation errors
        """
        errors = []
        expected_type = schema.get("type")

        # Type validation
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
        }

        if expected_type == "dict":
            # A section header whose child keys are all commented out parses as
            # YAML null (e.g. `market_data:` with only comments under it). Treat
            # that as an empty section — every property is optional — rather than
            # failing startup. This is a normal config-editing pattern.
            if value is None:
                value = {}
            if not isinstance(value, dict):
                errors.append(ConfigValidationError(
                    path=path,
                    message=f"Expected dict, got {type(value).__name__}"
                ))
                return errors

            # Validate nested properties
            if "properties" in schema:
                errors.extend(self._validate_dict(value, schema["properties"], path))

        elif expected_type in type_map:
            expected = type_map[expected_type]
            if not isinstance(value, expected):
                errors.append(ConfigValidationError(
                    path=path,
                    message=f"Expected {expected_type}, got {type(value).__name__}"
                ))
                return errors

            # Numeric range validation
            if expected_type in ("int", "float"):
                if "min" in schema and value < schema["min"]:
                    errors.append(ConfigValidationError(
                        path=path,
                        message=f"Value {value} is below minimum {schema['min']}"
                    ))
                if "max" in schema and value > schema["max"]:
                    errors.append(ConfigValidationError(
                        path=path,
                        message=f"Value {value} is above maximum {schema['max']}"
                    ))

            # Options validation
            if "options" in schema and value not in schema["options"]:
                errors.append(ConfigValidationError(
                    path=path,
                    message=f"Value '{value}' not in allowed options: {schema['options']}"
                ))

        return errors

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Dot-notation key (e.g., "server.port")
            default: Default value if not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value


# Global config service instance
config_service = ConfigService()


def get_api_token() -> str:
    """Resolve the API auth token: TRADINGBOT_API_TOKEN env var first, then
    server.api_token in config.yaml. Empty string means auth is disabled
    (only permitted on loopback binding, enforced at startup)."""
    import os

    return (
        os.environ.get("TRADINGBOT_API_TOKEN", "")
        or (config_service.get("server.api_token") or "")
    )
