"""Re-ingest completed documents into the existing Qdrant/OpenSearch indexes.

Use after a change that alters how chunks are indexed but not the uploaded files, e.g. turning on
contextual chunk headers (INGESTION_CONTEXT_HEADERS, default true): chunks written before that
change have no header in their embedding or BM25 text until they are re-ingested.

Each document is re-converted from its stored upload and written under a fresh version; the old
chunks are deleted only after the new version is fully written, so a document stays searchable and
`completed` throughout, and a failure leaves its old chunks in place. Runs in-process (needs the
worker environment: Docling for PDF/DOCX/HTML, the embedding model, both stores). Nothing is
written without --yes; without it the selection is only listed.

    PYTHONPATH=. python scripts/reindex.py --all                  # dry run: list what would be re-ingested
    PYTHONPATH=. python scripts/reindex.py --all --yes
    PYTHONPATH=. python scripts/reindex.py --owner alice --yes
    PYTHONPATH=. python scripts/reindex.py --document-id <id> --document-id <id> --yes
"""
import argparse
import sys
import time
from contextlib import closing
from pathlib import Path

from app.db import registry


def select_documents(document_ids: list[str] | None, owner: str | None) -> list[dict]:
    query = "SELECT document_id, filename, file_path, owner_id, content_hash FROM documents WHERE status = 'completed'"
    params: list = []
    if owner:
        query += " AND owner_id = ?"
        params.append(owner)
    if document_ids:
        query += f" AND document_id IN ({','.join('?' * len(document_ids))})"
        params.extend(document_ids)
    query += " ORDER BY created_at, document_id"
    with closing(registry.get_connection()) as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all", action="store_true", help="every completed document")
    scope.add_argument("--document-id", action="append", help="one document (repeatable)")
    scope.add_argument("--owner", help="every completed document of this owner")
    parser.add_argument("--database", type=Path, default=registry.DB_PATH, help="SQLite registry path")
    parser.add_argument("--yes", action="store_true", help="actually re-ingest (default: dry run)")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)

    registry.DB_PATH = args.database
    registry.init_db()
    docs = select_documents(args.document_id, args.owner)
    if not docs:
        print("No completed documents match.")
        return 0
    for d in docs:
        print(f"{d['document_id']}  {Path(d['file_path']).suffix.lower() or '?':6} {d['filename']}")
    if not args.yes:
        print(f"Dry run: {len(docs)} document(s) would be re-ingested. Re-run with --yes.")
        return 0

    from app.ingestion.worker import reindex_document  # heavy imports happen lazily inside

    failed = 0
    for d in docs:
        t0 = time.perf_counter()
        try:
            out = reindex_document(d["document_id"])
            print(f"ok    {d['document_id']}  {out['chunk_count']} chunks  {time.perf_counter() - t0:.1f}s")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {d['document_id']}  {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.stop_on_error:
                break
    print(f"Re-ingested {len(docs) - failed} of {len(docs)} document(s); {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
