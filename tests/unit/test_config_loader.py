"""
Unit tests for ALMA Configuration Loader.

Tests ConfigLoader: YAML loading, env var expansion, Key Vault lookup,
defaults, and save/load roundtrip.
"""

from unittest.mock import MagicMock

import yaml

from alma.config.loader import ConfigLoader


class TestConfigLoaderDefaults:
    """Tests for default configuration."""

    def test_defaults_returned_for_missing_file(self, tmp_path):
        missing = str(tmp_path / "nonexistent.yaml")
        config = ConfigLoader.load(missing)
        assert config == ConfigLoader._get_defaults()

    def test_defaults_have_required_keys(self):
        defaults = ConfigLoader._get_defaults()
        assert "project_id" in defaults
        assert "storage" in defaults
        assert "embedding_provider" in defaults
        assert "agents" in defaults

    def test_default_storage_is_file(self):
        assert ConfigLoader._get_defaults()["storage"] == "file"


class TestConfigLoaderYAML:
    """Tests for YAML file loading."""

    def test_load_simple_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"alma": {"project_id": "test", "storage": "sqlite"}})
        )
        config = ConfigLoader.load(str(config_file))
        assert config["project_id"] == "test"
        assert config["storage"] == "sqlite"

    def test_load_without_alma_section(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"project_id": "direct", "storage": "file"}))
        config = ConfigLoader.load(str(config_file))
        assert config["project_id"] == "direct"

    def test_load_nested_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        data = {
            "alma": {
                "project_id": "nested",
                "agents": {"helena": {"can_learn": ["testing"]}},
            }
        }
        config_file.write_text(yaml.dump(data))
        config = ConfigLoader.load(str(config_file))
        assert config["agents"]["helena"]["can_learn"] == ["testing"]

    def test_load_empty_file_returns_defaults(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        config = ConfigLoader.load(str(config_file))
        assert config == ConfigLoader._get_defaults()

    def test_load_list_values(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        data = {"alma": {"backends": ["sqlite", "postgres", "file"]}}
        config_file.write_text(yaml.dump(data))
        config = ConfigLoader.load(str(config_file))
        assert config["backends"] == ["sqlite", "postgres", "file"]


class TestConfigLoaderEnvExpansion:
    """Tests for environment variable expansion."""

    def test_expand_single_env_var(self, monkeypatch):
        monkeypatch.setenv("ALMA_TEST_VAR", "expanded_value")
        result = ConfigLoader._expand_value("${ALMA_TEST_VAR}")
        assert result == "expanded_value"

    def test_expand_env_var_in_string(self, monkeypatch):
        monkeypatch.setenv("DB_HOST", "localhost")
        result = ConfigLoader._expand_value("postgresql://${DB_HOST}:5432/alma")
        assert result == "postgresql://localhost:5432/alma"

    def test_expand_multiple_env_vars(self, monkeypatch):
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        result = ConfigLoader._expand_value("${DB_HOST}:${DB_PORT}")
        assert result == "localhost:5432"

    def test_missing_env_var_keeps_original(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR_12345", raising=False)
        result = ConfigLoader._expand_value("${NONEXISTENT_VAR_12345}")
        assert result == "${NONEXISTENT_VAR_12345}"

    def test_no_expansion_without_dollar_brace(self):
        result = ConfigLoader._expand_value("plain_string")
        assert result == "plain_string"

    def test_expand_config_dict(self, monkeypatch):
        monkeypatch.setenv("MY_PROJECT", "test-proj")
        config = {"project_id": "${MY_PROJECT}", "storage": "file"}
        expanded = ConfigLoader._expand_config(config)
        assert expanded["project_id"] == "test-proj"
        assert expanded["storage"] == "file"

    def test_expand_config_nested_dict(self, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", "abc123")
        config = {"storage": {"connection": {"key": "${SECRET_KEY}"}}}
        expanded = ConfigLoader._expand_config(config)
        assert expanded["storage"]["connection"]["key"] == "abc123"

    def test_expand_config_list(self, monkeypatch):
        monkeypatch.setenv("AGENT1", "helena")
        config = ["${AGENT1}", "victor"]
        expanded = ConfigLoader._expand_config(config)
        assert expanded == ["helena", "victor"]

    def test_expand_non_string_passthrough(self):
        assert ConfigLoader._expand_config(42) == 42
        assert ConfigLoader._expand_config(3.14) == 3.14
        assert ConfigLoader._expand_config(True) is True
        assert ConfigLoader._expand_config(None) is None

    def test_full_load_with_env_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALMA_PROJ", "my-project")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"alma": {"project_id": "${ALMA_PROJ}", "storage": "sqlite"}})
        )
        config = ConfigLoader.load(str(config_file))
        assert config["project_id"] == "my-project"


class TestConfigLoaderKeyVault:
    """Tests for Azure Key Vault integration."""

    def test_keyvault_ref_detected(self):
        result = ConfigLoader._expand_value("${KEYVAULT:my-secret}")
        # Without AZURE_KEYVAULT_URL, should return original
        assert "KEYVAULT:my-secret" in result

    def test_keyvault_no_vault_url(self, monkeypatch):
        monkeypatch.delenv("AZURE_KEYVAULT_URL", raising=False)
        ConfigLoader._keyvault_client = None  # Reset cached client
        result = ConfigLoader._get_keyvault_secret("test-secret")
        assert result == "${KEYVAULT:test-secret}"

    def test_keyvault_missing_packages(self, monkeypatch):
        monkeypatch.setenv("AZURE_KEYVAULT_URL", "https://myvault.vault.azure.net")
        ConfigLoader._keyvault_client = None
        # azure packages not installed in test env, so ImportError expected
        result = ConfigLoader._get_keyvault_secret("test-secret")
        assert "KEYVAULT:test-secret" in result

    def test_keyvault_client_cached(self, monkeypatch):
        mock_client = MagicMock()
        mock_secret = MagicMock()
        mock_secret.value = "secret_value"
        mock_client.get_secret.return_value = mock_secret

        ConfigLoader._keyvault_client = mock_client
        result = ConfigLoader._get_keyvault_secret("my-secret")
        assert result == "secret_value"
        mock_client.get_secret.assert_called_once_with("my-secret")

        # Cleanup
        ConfigLoader._keyvault_client = None

    def test_keyvault_get_secret_failure(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.get_secret.side_effect = Exception("Vault unavailable")

        ConfigLoader._keyvault_client = mock_client
        result = ConfigLoader._get_keyvault_secret("failing-secret")
        assert result == "${KEYVAULT:failing-secret}"

        # Cleanup
        ConfigLoader._keyvault_client = None


class TestConfigLoaderSave:
    """Tests for saving configuration."""

    def test_save_creates_file(self, tmp_path):
        config_file = tmp_path / "output" / "config.yaml"
        config = {"project_id": "saved", "storage": "sqlite"}
        ConfigLoader.save(config, str(config_file))
        assert config_file.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        config_file = tmp_path / "deep" / "nested" / "config.yaml"
        ConfigLoader.save({"project_id": "deep"}, str(config_file))
        assert config_file.exists()

    def test_save_load_roundtrip(self, tmp_path):
        config_file = tmp_path / "roundtrip.yaml"
        original = {"project_id": "roundtrip", "storage": "file", "agents": {}}
        ConfigLoader.save(original, str(config_file))
        loaded = ConfigLoader.load(str(config_file))
        assert loaded == original

    def test_save_wraps_in_alma_section(self, tmp_path):
        config_file = tmp_path / "wrapped.yaml"
        ConfigLoader.save({"project_id": "test"}, str(config_file))
        with open(config_file) as f:
            raw = yaml.safe_load(f)
        assert "alma" in raw
        assert raw["alma"]["project_id"] == "test"
