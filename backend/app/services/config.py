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
        }
    },
    "database": {
        "type": "dict",
        "required": False,
        "properties": {
            "url": {"type": "str", "required": False},
        }
    },
    "trading": {
        "type": "dict",
        "required": False,
        "properties": {
            "execution_interval_seconds": {"type": "int", "required": False, "min": 1},
            "pnl_snapshot_interval_seconds": {"type": "int", "required": False, "min": 1},
            "default_stop_loss_percent": {"type": "float", "required": False, "min": 0, "max": 100},
            "default_daily_loss_limit": {"type": "float", "required": False, "min": 0},
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

    def load_and_validate(self) -> Dict[str, Any]:
        """Load and validate the configuration file.

        Returns:
            Validated configuration dictionary.

        Raises:
            ConfigValidationException: If validation fails.
            FileNotFoundError: If config file not found.
        """
        errors: List[ConfigValidationError] = []

        # Check if file exists
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            self._config = {}
            return self._config

        # Load YAML
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            errors.append(ConfigValidationError(
                path="",
                message=f"Invalid YAML syntax: {str(e)}"
            ))
            raise ConfigValidationException(errors)

        if config is None:
            config = {}

        if not isinstance(config, dict):
            errors.append(ConfigValidationError(
                path="",
                message=f"Config must be a dictionary, got {type(config).__name__}"
            ))
            raise ConfigValidationException(errors)

        # Validate against schema
        errors.extend(self._validate_dict(config, CONFIG_SCHEMA, ""))

        if errors:
            raise ConfigValidationException(errors)

        self._config = config
        logger.info(f"Configuration loaded and validated from {self.config_path}")
        return config

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
