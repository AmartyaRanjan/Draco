"""Explicitly assign ownerless documents during a verified single-tenant migration."""
import argparse
from pathlib import Path

from app.db import registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-id", required=True, help="authenticated principal that owns all legacy documents")
    parser.add_argument("--expected-count", required=True, type=int,
                        help="exact number of ownerless documents confirmed by the operator")
    parser.add_argument("--database", type=Path, default=registry.DB_PATH, help="SQLite registry path")
    args = parser.parse_args()

    registry.DB_PATH = args.database
    registry.init_db()
    migrated = registry.assign_unowned_documents(args.owner_id, args.expected_count)
    print(f"Assigned ownership to {migrated} legacy documents.")


if __name__ == "__main__":
    main()
