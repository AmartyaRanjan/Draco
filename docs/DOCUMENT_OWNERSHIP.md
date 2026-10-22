# Document ownership and legacy migration

Document access is scoped to the authenticated principal bound to the API key. New direct uploads
and approved-source ingestion record that principal as the document owner. Retrieval always
intersects requested document IDs with that owner's registry rows; omitted filters mean all of the
owner's documents. Status, page, and agent document-access paths use the same ownership boundary.

The schema migration adds a nullable `owner_id` column. Existing rows are intentionally left
unassigned and inaccessible; startup never guesses their owner. For a verified single-tenant
deployment only, an operator may assign all remaining ownerless rows after confirming that they
belong to one principal:

```powershell
python -m scripts.assign_legacy_document_owner --owner-id <authenticated-principal> --expected-count <verified-count>
```

The command aborts unless the current ownerless row count exactly matches `--expected-count`. Use
`--database <path>` for a non-default registry location. Do not run this migration on a shared or
multi-user database. Back up the database first and determine ownership from trusted records; there
is no automatic fallback for ambiguous legacy documents.
