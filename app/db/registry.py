import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "documents.db"

#connect to the SQLite database and return the connection object conn.
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    # closing() is required: sqlite3.Connection as a context manager only
    # commits/rolls back, it never closes. Without an explicit close the
    # file handle survives until cyclic GC, which holds an exclusive lock
    # on Windows and leaks descriptors in long-lived API/worker processes.
    with closing(get_connection()) as conn:
        # WAL: the API and the ingestion worker both write this file; readers must not block writers (persists in the file).
        conn.execute("PRAGMA journal_mode=WAL")
        with conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL,
                job_id TEXT,
                error TEXT,
                content_hash TEXT,
                owner_id TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

            # Databases created before duplicate detection and document ownership lack these columns.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
            if "content_hash" not in columns:
                conn.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")
            if "owner_id" not in columns:
                conn.execute("ALTER TABLE documents ADD COLUMN owner_id TEXT")

            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_content_hash
            ON documents (content_hash)
        """)
            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_owner_id
            ON documents (owner_id)
        """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS document_pages (
                document_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                page_type TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                image_count INTEGER NOT NULL,
                large_image_count INTEGER NOT NULL,
                ocr_used INTEGER NOT NULL,
                reason TEXT NOT NULL,
                extracted_chars INTEGER,
                low_yield INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (document_id, page_no)
            )
        """)

            # Databases created before low-yield flagging lack these columns.
            page_columns = {row["name"] for row in conn.execute("PRAGMA table_info(document_pages)")}
            if "extracted_chars" not in page_columns:
                conn.execute("ALTER TABLE document_pages ADD COLUMN extracted_chars INTEGER")
            if "low_yield" not in page_columns:
                conn.execute(
                    "ALTER TABLE document_pages ADD COLUMN low_yield INTEGER NOT NULL DEFAULT 0"
                )

#insert a new document into the database.
def create_document(document_id, filename, file_path, owner_id=None):
    with closing(get_connection()) as conn:
        with conn:
            conn.execute("""
            INSERT INTO documents
                (document_id, filename, file_path, status, owner_id)
            VALUES (?, ?, ?, 'uploaded', ?)
        """, (document_id, filename, file_path, owner_id))

# A failed document does not block re-uploading the same file.
FAILED_STATUS = "failed"


def register_upload(document_id, filename, file_path, content_hash, owner_id=None):
    """Insert the document unless identical content is already registered.

    Returns None if the document was inserted, otherwise the existing document
    (as a dict). The check and insert share one write transaction, so two
    concurrent uploads of the same file cannot both be accepted.
    """
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            """
            SELECT * FROM documents
            WHERE content_hash = ? AND status != ? AND owner_id IS ?
            ORDER BY created_at
            LIMIT 1
            """,
            (content_hash, FAILED_STATUS, owner_id),
        ).fetchone()

        if existing is not None:
            conn.rollback()
            return dict(existing)

        conn.execute(
            """
            INSERT INTO documents
                (document_id, filename, file_path, status, content_hash, owner_id)
            VALUES (?, ?, ?, 'uploaded', ?, ?)
            """,
            (document_id, filename, file_path, content_hash, owner_id),
        )
        conn.commit()
        return None
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_pages(document_id, pages):
    """Replace the stored page-level data for a document (safe to re-run on retry).

    pages: iterable of dicts with the document_pages columns except document_id.
    """
    with closing(get_connection()) as conn:
        with conn:
            conn.execute("DELETE FROM document_pages WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
            INSERT INTO document_pages
                (document_id, page_no, page_type, char_count, image_count,
                 large_image_count, ocr_used, reason, extracted_chars, low_yield)
            VALUES (:document_id, :page_no, :page_type, :char_count, :image_count,
                    :large_image_count, :ocr_used, :reason, :extracted_chars, :low_yield)
            """,
                [
                    {"extracted_chars": None, "low_yield": 0, **page, "document_id": document_id}
                    for page in pages
                ],
            )


def get_pages(document_id):
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT * FROM document_pages WHERE document_id = ? ORDER BY page_no",
            (document_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_document_for_owner(document_id, owner_id):
    if not isinstance(owner_id, str) or not owner_id.strip():
        return None
    with closing(get_connection()) as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ? AND owner_id = ?",
            (document_id, owner_id),
        ).fetchone()
        return dict(row) if row else None


def get_pages_for_owner(document_id, owner_id):
    if not isinstance(owner_id, str) or not owner_id.strip():
        return []
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT pages.* FROM document_pages AS pages "
            "JOIN documents ON documents.document_id = pages.document_id "
            "WHERE pages.document_id = ? AND documents.owner_id = ? ORDER BY pages.page_no",
            (document_id, owner_id),
        ).fetchall()
        return [dict(row) for row in rows]


def list_document_ids_for_owner(owner_id, requested_document_ids=None):
    """Return only documents owned by the authenticated principal.

    Client filters may narrow this set, never widen it. Ownerless legacy rows are excluded.
    """
    if not isinstance(owner_id, str) or not owner_id.strip():
        return []
    with closing(get_connection()) as conn:
        if requested_document_ids is None:
            rows = conn.execute(
                "SELECT document_id FROM documents WHERE owner_id = ? ORDER BY document_id",
                (owner_id,),
            ).fetchall()
            return [row["document_id"] for row in rows]
        requested = list(dict.fromkeys(requested_document_ids))
        if not requested:
            return []
        placeholders = ",".join("?" for _ in requested)
        rows = conn.execute(
            f"SELECT document_id FROM documents WHERE owner_id = ? AND document_id IN ({placeholders})",
            (owner_id, *requested),
        ).fetchall()
        allowed = {row["document_id"] for row in rows}
        return [document_id for document_id in requested if document_id in allowed]


def list_documents_for_owner(owner_id, limit=50):
    if not isinstance(owner_id, str) or not owner_id.strip():
        return []
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT document_id, filename, status FROM documents WHERE owner_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (owner_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]


def assign_unowned_documents(owner_id, expected_count):
    """Explicit operator migration for legacy single-tenant data; never called automatically."""
    if not isinstance(owner_id, str) or not owner_id.strip():
        raise ValueError("owner_id must be a non-empty authenticated principal")
    if not isinstance(expected_count, int) or expected_count < 0:
        raise ValueError("expected_count must be a non-negative integer")
    with closing(get_connection()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        count = conn.execute("SELECT COUNT(*) FROM documents WHERE owner_id IS NULL OR owner_id = ''").fetchone()[0]
        if count != expected_count:
            conn.rollback()
            raise ValueError(f"expected {expected_count} unowned documents, found {count}")
        conn.execute(
            "UPDATE documents SET owner_id = ? WHERE owner_id IS NULL OR owner_id = ''",
            (owner_id,),
        )
        conn.commit()
        return count


#status of the document can be 'uploaded', 'processing', 'processed', 'error'
def update_status(document_id, status, job_id=None, error=None):
    with closing(get_connection()) as conn:
        with conn:
            conn.execute("""
            UPDATE documents
            SET status = ?,
                job_id = COALESCE(?, job_id),
                error = ?
            WHERE document_id = ?
        """, (status, job_id, error, document_id))

#retrieve document by id
def get_document(document_id):
    with closing(get_connection()) as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return dict(row) if row else None

#for saving the RQ id of the job associated with a document in the database. This allows us to track which RQ job is processing which document.
def set_job_id(document_id, job_id):
    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                "UPDATE documents SET job_id = ? WHERE document_id = ?",
                (job_id, document_id),
            )