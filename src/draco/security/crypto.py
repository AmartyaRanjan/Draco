"""Encryption at rest for persisted payloads (`pip install draco[crypto]`).

`FernetCipher` implements `draco.state.PayloadCipher` with `cryptography`'s MultiFernet (AES-128-CBC +
HMAC-SHA256, authenticated). Keys come from a `SecretProvider` as one secret holding comma-separated
Fernet keys, newest first: the first key encrypts, every key decrypts. To rotate, prepend a new key,
run `store.rotate_encryption()`, then drop the old key."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from draco.core.errors import ConfigurationError
from draco.security.secrets import Secret, SecretProvider


def _fernet() -> Any:
    try:
        from cryptography import fernet
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConfigurationError("encryption at rest needs cryptography: pip install 'draco[crypto]'") from exc
    return fernet


class FernetCipher:
    def __init__(self, keys: Sequence[Secret]):
        if not keys:
            raise ConfigurationError("FernetCipher needs at least one key")
        f = _fernet()
        try:
            self._f = f.MultiFernet([f.Fernet(k.reveal().encode()) for k in keys])
        except (ValueError, TypeError):  # never echo key material
            raise ConfigurationError("invalid Fernet key (expected 32 url-safe base64-encoded bytes)") from None
        self.key_count = len(keys)

    @classmethod
    def from_secret(cls, secret: Secret) -> FernetCipher:
        """One secret holding comma-separated keys, newest (primary) first."""
        return cls([Secret(k.strip()) for k in secret.reveal().split(",") if k.strip()])

    @classmethod
    def from_provider(cls, provider: SecretProvider, name: str) -> FernetCipher:
        return cls.from_secret(provider.get(name))

    @staticmethod
    def generate_key() -> str:
        return str(_fernet().Fernet.generate_key().decode())

    def encrypt(self, plaintext: str) -> str:
        return str(self._f.encrypt(plaintext.encode()).decode())

    def decrypt(self, token: str) -> str:
        return str(self._f.decrypt(token.encode()).decode())

    def rotate(self, token: str) -> str:
        return str(self._f.rotate(token.encode()).decode())

    def __repr__(self) -> str:
        return f"FernetCipher(keys=<{self.key_count} redacted>)"
