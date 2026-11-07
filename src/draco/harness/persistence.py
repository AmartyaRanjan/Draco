"""Build the configured state store from `DracoSettings`: Postgres (`state_url`), SQLite (`state_path`)
or in-memory, optionally encrypting payloads with the key named by `state_encryption_key`."""
from __future__ import annotations

from typing import Any

from draco.core.config import DracoSettings
from draco.core.errors import ConfigurationError
from draco.security.secrets import EnvSecretProvider, SecretProvider
from draco.state.memory import InMemoryStore
from draco.state.stores import PayloadCipher


def cipher_from_settings(settings: DracoSettings, secrets: SecretProvider | None = None) -> PayloadCipher | None:
    if not settings.state_encryption_key:
        return None
    from draco.security.crypto import FernetCipher
    return FernetCipher.from_provider(secrets or EnvSecretProvider(), settings.state_encryption_key)


def open_store(settings: DracoSettings, secrets: SecretProvider | None = None) -> Any:
    """`secrets` resolves `state_encryption_key` (default: environment variables)."""
    cipher = cipher_from_settings(settings, secrets)
    if settings.state_url:
        if not settings.state_url.startswith(("postgresql://", "postgres://")):
            raise ConfigurationError("state_url must be a postgresql:// URL")
        from draco.state.postgres import PostgresStore
        return PostgresStore(settings.state_url, cipher=cipher)
    if settings.state_path:
        from draco.state.sqlite import SqliteStore
        return SqliteStore(settings.state_path, cipher=cipher)
    if cipher is not None:
        raise ConfigurationError("state_encryption_key is set but no persistent store (state_path/state_url) is")
    return InMemoryStore()
