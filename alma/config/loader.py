"""
ALMA Configuration Loader.

Handles loading configuration from files and environment variables,
with support for Azure Key Vault secret resolution.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loads ALMA configuration from YAML files with environment variable expansion.

    Supports:
    - ${ENV_VAR} syntax for environment variables
    - ${KEYVAULT:secret-name} syntax for Azure Key Vault (when configured)
    """

    _keyvault_client = None

    @classmethod
    def load(cls, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml

        Returns:
            Parsed and expanded configuration dict
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config not found at {config_path}, using defaults")
            return cls._get_defaults()

        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)

        if raw_config is None:
            logger.warning(f"Config file {config_path} is empty, using defaults")
            return cls._get_defaults()

        # Get the 'alma' section or use whole file
        config = raw_config.get("alma", raw_config)

        # Expand environment variables and secrets
        config = cls._expand_config(config)

        return config

    @classmethod
    def _expand_config(cls, config: Any) -> Any:
        """Recursively expand environment variables and secrets in config."""
        if isinstance(config, dict):
            return {k: cls._expand_config(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [cls._expand_config(item) for item in config]
        elif isinstance(config, str):
            return cls._expand_value(config)
        return config

    @classmethod
    def _expand_value(cls, value: str) -> str:
        """
        Expand a single config value.

        Handles:
        - ${ENV_VAR} -> os.environ["ENV_VAR"]
        - ${KEYVAULT:secret-name} -> Azure Key Vault lookup
        """
        if not isinstance(value, str) or "${" not in value:
            return value

        # Handle ${VAR} patterns
        import re

        pattern = r"\$\{([^}]+)\}"

        def replace(match):
            ref = match.group(1)

            if ref.startswith("KEYVAULT:"):
                secret_name = ref[9:]  # Remove "KEYVAULT:" prefix
                return cls._get_keyvault_secret(secret_name)
            else:
                # Environment variable
                env_value = os.environ.get(ref)
                if env_value is None:
                    logger.warning(f"Environment variable {ref} not set")
                    return match.group(0)  # Keep original if not found
                return env_value

        return re.sub(pattern, replace, value)

    @classmethod
    def _get_keyvault_secret(cls, secret_name: str) -> str:
        """
        Retrieve secret from Azure Key Vault.

        Requires AZURE_KEYVAULT_URL environment variable.
        """
        if cls._keyvault_client is None:
            vault_url = os.environ.get("AZURE_KEYVAULT_URL")
            if not vault_url:
                logger.error("AZURE_KEYVAULT_URL not set, cannot retrieve secrets")
                return f"${{KEYVAULT:{secret_name}}}"

            try:
                from azure.identity import DefaultAzureCredential
                from azure.keyvault.secrets import SecretClient

                credential = DefaultAzureCredential()
                cls._keyvault_client = SecretClient(
                    vault_url=vault_url,
                    credential=credential,
                )
            except ImportError:
                logger.error(
                    "azure-identity and azure-keyvault-secrets packages required "
                    "for Key Vault integration"
                )
                return f"${{KEYVAULT:{secret_name}}}"

        try:
            secret = cls._keyvault_client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            return f"${{KEYVAULT:{secret_name}}}"

    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "project_id": "default",
            "storage": "file",
            "embedding_provider": "local",
            "agents": {},
        }

    @classmethod
    def save(cls, config: Dict[str, Any], config_path: str):
        """
        Save configuration to YAML file.

        Note: Does NOT save secrets - those should remain as ${} references.
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump({"alma": config}, f, default_flow_style=False)
