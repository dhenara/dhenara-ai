"""
Settings and configuration for Dhenara-AI.
"""

import importlib
import logging
import os
from typing import Any
from dhenara.ai.config import default_settings

logger = logging.getLogger(__name__)


class Settings:
    """
    Settings class that loads default settings and user-defined settings.
    User settings can be defined in standard locations without needing environment variables.
    """

    def __init__(self):
        self._settings: dict[str, Any] = {}
        self._load_default_settings()
        self._load_user_settings()

    def _load_default_settings(self):
        for setting in dir(default_settings):
            if setting.isupper():
                self._settings[setting] = getattr(default_settings, setting)

    def _load_user_settings(self):
        """
        Search for user settings in standard locations and load them.
        Priority order:
        1. Current working directory
        2. User's home directory
        3. System-wide directory (optional)
        """
        possible_files = [
            os.path.join(os.getcwd(), "dhenara_config.py"),
            # os.path.expanduser("~/.dhenara_config.py"),
            # "/etc/dhenara_config.py",  # for system-wide settings
        ]

        for file_path in possible_files:
            if os.path.isfile(file_path):
                self._import_settings(file_path)
                logger.info(f"Loaded user settings from {file_path}")
                break  # Stop at the first found settings file

    def _import_settings(self, file_path: str):
        spec = importlib.util.spec_from_file_location("user_settings", file_path)
        if spec and spec.loader:
            user_settings = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_settings)
            for setting in dir(user_settings):
                if setting.isupper():
                    self._settings[setting] = getattr(user_settings, setting)
        else:
            logger.error(f"Unable to load settings from {file_path}")

    def __getattr__(self, name: str) -> Any:
        """Get a setting value"""
        try:
            return self._settings[name]
        except KeyError:
            raise AttributeError(f"Setting '{name}' not found")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set a setting value"""
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._settings[name] = value

    def update(self, **kwargs: Any) -> None:
        """Update settings with new values"""
        for key, value in kwargs.items():
            if key.isupper():
                self._settings[key] = value

    @property
    def configured_settings(self) -> list[str]:
        """Return list of all configured settings"""
        return sorted(self._settings.keys())


# Create a global settings object
settings = Settings()
