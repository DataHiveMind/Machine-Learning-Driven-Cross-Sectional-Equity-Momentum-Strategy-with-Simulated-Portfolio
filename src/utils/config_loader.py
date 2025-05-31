import os
import yaml
import json
from pathlib import Path

class ConfigLoader:
    """
    Singleton-style configuration loader for YAML/JSON config files.
    Supports environment-specific configs (e.g., DEV, PROD).
    """
    _configs = {}

    @classmethod
    def load(cls, config_path: str, env_var: str = "ENV", validate: bool = False, required_keys: list = None):
        """
        Load and cache a config file. Supports YAML and JSON.
        Optionally validates required keys.
        Args:
            config_path: Path to config file.
            env_var: Environment variable for environment-specific configs.
            validate: If True, checks for required keys.
            required_keys: List of keys that must be present.
        Returns:
            dict with config parameters.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        cache_key = str(config_file.resolve())
        if cache_key in cls._configs:
            return cls._configs[cache_key]

        # Load config
        if config_file.suffix in [".yaml", ".yml"]:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
        elif config_file.suffix == ".json":
            with open(config_file, "r") as f:
                config = json.load(f)
        else:
            raise ValueError("Unsupported config file format.")

        # Environment-specific override
        env = os.getenv(env_var, None)
        if env and isinstance(config, dict) and env in config:
            config = config[env]

        # Optional validation
        if validate and required_keys:
            missing = [k for k in required_keys if k not in config]
            if missing:
                raise ValueError(f"Missing required config keys: {missing}")

        cls._configs[cache_key] = config
        return config

    @classmethod
    def get(cls, config_path: str, **kwargs):
        """Alias for load()."""
        return cls.load(config_path, **kwargs)