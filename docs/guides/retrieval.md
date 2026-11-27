# Wrap a retrieval backend

Retrieval is where multi-tenant systems most often leak. A shared vector index holds every customer's documents, and a search can return a chunk the caller must never see, even if you asked for only their documents. Draco's retrieval adapter wraps *your* backend and authorizes access **before the search and again after it**.

Draco does not ship a vector store or a retriever. You bring the backend; the adapter adds the checks and the events.

## The contract

A backend implements `Retriever`: one async method that takes the context, a query and an optional list of document ids, and returns a `RetrievalResult` of `Evidence`:

```python
from draco.adapters.retrieval import Evidence, RetrievalResult

class MyRetriever:
    async def retrieve(self, ctx, query, document_ids):
        hits = await my_index.search(query, restrict_to=document_ids)
        return RetrievalResult(query, [
            Evidence(document_id=h.doc, text=h.text, chunk_id=h.id, score=h.score) for h in hits
        ], degraded=[])      # list names of backends that failed, if any
```

`Evidence` holds the `document_id`, the `text`, and optionally a `chunk_id`, a `score`, a `source` and a `locator` (page range, offsets). `degraded` lets you report a partial answer, for example `["bm25"]` when one of two search backends was down.

## Turn it into a workload

`retrieval_workload(retriever, document_resource=...)` returns a workload you register with the harness. Its input is a query string, or `{"query": ..., "document_ids": [...]}`, and its output is `{"query", "evidence", "degraded"}`:

```python
from draco.adapters.retrieval import retrieval_workload

h.register("search", retrieval_workload(MyRetriever(), document_resource=owned))
```

## Say who owns each document

`document_resource` is a function from a document id to a `Resource` that carries the **owning tenant**:

```python
from draco.security import Resource

def owned(doc_id):
    return Resource("document", doc_id, tenant_id=owner_of(doc_id))
```

With it, the adapter authorizes `document.read` for every document the request names (before the search) and for every document the search **returned** (after it), dropping and auditing anything the caller may not read. The second check is the important one: shared indexes over-return, so a result list is never trusted to be pre-filtered.

!!! warning "Without `document_resource`, an identified caller is denied"
    If a run has an identity but you gave no `document_resource`, access to documents is **denied** with "document access cannot be verified", not allowed. The adapter fails closed, so forgetting this configuration cannot silently expose data.

## Example: one shared index, two tenants

<!-- doctest -->
```python
import asyncio
from draco import Harness, Identity
from draco.adapters.retrieval import Evidence, RetrievalResult, retrieval_workload
from draco.security import Resource

OWNERS = {"d1": "acme", "d2": "globex"}

class SharedIndex:
    """One index for every tenant. It over-returns: it ignores the document filter."""
    async def retrieve(self, ctx, query, document_ids):
        return RetrievalResult(query, [Evidence("d1", "Acme pricing sheet", chunk_id="d1#0", score=0.9),
                                       Evidence("d2", "Globex salary bands", chunk_id="d2#0", score=0.8)])

def owned(doc_id):
    return Resource("document", doc_id, tenant_id=OWNERS[doc_id])

async def main():
    alice = Identity("alice", "acme")
    h = Harness()
    h.register("search", retrieval_workload(SharedIndex(), document_resource=owned))
    h.register("search_unverified", retrieval_workload(SharedIndex()))

    r = await h.run("search", "pricing", identity=alice)
    print(r.status, [e["document_id"] for e in r.output["evidence"]])
    after = next(e for e in h.tracer.trace(r.execution_id) if e["event"] == "retrieval.after")
    print("selected:", after["data"]["selected"], "| withheld:", after["data"]["withheld"])

    r = await h.run("search", {"query": "salaries", "document_ids": ["d2"]}, identity=alice)
    print(r.status, "|", r.error.message)

    r = await h.run("search_unverified", "pricing", identity=alice)
    print(r.status, "|", r.error.message)

asyncio.run(main())
```

<!-- output -->
```text
completed ['d1']
selected: 1 | withheld: 1
failed | 'document.read' on document 'd2' denied: cross-tenant access denied
failed | document access cannot be verified: no document_resource configured
```

Line by line:

1. Alice searched with no document filter. The index returned a chunk from each tenant, and the adapter kept only Acme's.
2. The `retrieval.after` event says one chunk was selected and **one was withheld**, so you can alert on unexpected withholding.
3. Alice named Globex's document explicitly. The request was denied **before** the search ran.
4. With no `document_resource`, the identified caller is refused outright.

## Ingestion

`ingest_workload(callable, resource_for=...)` is the same idea for writes. Its input is `{"document_id", "file_path"}`, and `document.ingest` is authorized on the target document before your callable runs. The callable may be sync (run in a thread) or async, and re-running is only safe if it is idempotent, for example with deterministic chunk ids.

```python
from draco.adapters.ingestion import ingest_workload

h.register("ingest", ingest_workload(my_ingest, resource_for=owned))
```

## See also

- [Security model](../concepts/security.md): the full list of authorizer actions.
- [draco-rag](https://github.com/AmartyaRanjan/draco-rag) builds its hybrid retriever on these same before-and-after checks, in a complete service.
