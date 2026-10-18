"""Session persistence: reuse the app's `SqliteConversationStore` / `ConversationStore` protocol
unchanged. Draco's own stores hold execution history, not conversational memory."""
from __future__ import annotations

from typing import Any


def sqlite_conversation_store(path: str) -> Any:
    from app.inference.memory import SqliteConversationStore
    return SqliteConversationStore(path)
