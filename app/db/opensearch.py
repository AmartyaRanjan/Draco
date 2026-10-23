
import os

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk


INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "rag_chunks")

_client = None


def get_opensearch_client() -> OpenSearch:
    """Return one reusable OpenSearch client per process."""
    global _client

    if _client is None:
        url = os.getenv(
            "OPENSEARCH_URL",
            "http://localhost:9200",
        )
        username = os.getenv("OPENSEARCH_USERNAME") or None
        password = os.getenv("OPENSEARCH_PASSWORD") or None
        use_ssl = url.startswith("https://")
        verify_setting = os.getenv("OPENSEARCH_VERIFY_CERTS")
        # HTTPS must verify by default. An explicit opt-out is retained only for
        # local development; production refuses to construct an unsafe client.
        verify_certs = use_ssl if verify_setting is None else verify_setting.lower() == "true"
        if use_ssl and not verify_certs and os.getenv("APP_ENV", "development").lower() == "production":
            raise ValueError("OPENSEARCH_VERIFY_CERTS=true is required for HTTPS in production.")

        auth = (
            (username, password)
            if username and password
            else None
        )

        _client = OpenSearch(
            hosts=[url],
            http_auth=auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_assert_hostname=verify_certs,
            ssl_show_warn=verify_certs,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )

    return _client


def ensure_index() -> None:
    """Create the BM25 index with explicit field mappings."""
    client = get_opensearch_client()

    if client.indices.exists(index=INDEX_NAME):
        return

    client.indices.create(
        index=INDEX_NAME,
        body={
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "document_version": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "page_start": {"type": "integer"},
                    "page_end": {"type": "integer"},
                    "text": {"type": "text"},
                    "section_path": {"type": "text"},
                }
            }
        },
    )


def index_chunks(chunks: list[dict]) -> None:
    """Index one chunk batch for lexical/BM25 retrieval."""
    if not chunks:
        return

    ensure_index()

    actions = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            raise ValueError("Every chunk must have a chunk_id.")

        actions.append(
            {
                "_op_type": "index",
                "_index": INDEX_NAME,
                "_id": chunk_id,
                "_source": chunk,
            }
        )

    # Raise if any bulk item fails; do not silently mark ingestion complete.
    success_count, errors = bulk(
        get_opensearch_client(),
        actions,
        raise_on_error=False,
        raise_on_exception=False,
    )

    if errors:
        raise RuntimeError(
            f"OpenSearch bulk indexing failed for "
            f"{len(errors)} items; {success_count} succeeded."
        )


def delete_stale_versions(document_id: str, keep_version: str) -> None:
    """Remove chunks of this document written under any other version."""
    get_opensearch_client().delete_by_query(
        index=INDEX_NAME,
        body={
            "query": {
                "bool": {
                    "filter": [{"term": {"document_id": document_id}}],
                    "must_not": [
                        {"term": {"document_version": keep_version}}
                    ],
                }
            }
        },
        refresh=True,
        conflicts="proceed",
    )


def refresh_index() -> None:
    """Make recent writes searchable immediately, useful in tests."""
    get_opensearch_client().indices.refresh(index=INDEX_NAME)
