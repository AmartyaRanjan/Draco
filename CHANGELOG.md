# Changelog: PDF detection and Docling conversion stage

> Framework (`src/draco`) changes are tracked in [docs/DRACO_CHANGELOG.md](docs/DRACO_CHANGELOG.md).

Newest entry first. Iterations 4 and 5 touch the ingestion pipeline, the stores, the registry and their tests. Iterations 1 and 2 were limited to `app/ingestion/pdf_detector.py`,
`app/ingestion/convert.py` and `tests/`; Iteration 3 also touches the registry, upload route and worker.

---

## Iteration 5: OCR routing default, small chunks and low-yield pages (2026-10-11)

Follows up the "Found, not changed" list from Iteration 4. Measured on the real PDFs, ingested through the worker path against throwaway stores.

### 1. Page routing is now the default, with two guards

**Problem:** with document routing, one page that is not clearly digital turned OCR on for every page. The detector itself was right (534 of 548 pages of one book have a text layer), but the 548-page book spent 317 s converting and the 673-page book 381 s, almost all of it OCR.

**Changed:** `PDF_ROUTING` now defaults to `page` (`DEFAULT_ROUTING` in `convert.py`). On the 548-page book, conversion fell from 317 s to 120 s with the same 1,012 chunks.

**Found when testing the 673-page book:** page routing made it slower, not faster. Its 116 scattered OCR pages split the PDF into 171 separate Docling runs, and each run has a fixed cost: 520 s of conversion against 381 s for document routing. Two guards fix this:
- **Gap bridging** (`_bridge_short_gaps`, `PDF_ROUTING_MAX_BRIDGED_PAGES`, default 5): a stretch of 5 or fewer digital pages between two OCR runs is OCR'd as well, so the runs merge. This cut the 673-page book from 171 runs to 57 and from 520 s to 419 s of conversion, but that was still about 10% slower than document routing.
- **OCR fraction limit** (`PDF_ROUTING_MAX_OCR_FRACTION`, default 0.25): if more than a quarter of the pages need OCR after bridging, the whole document is converted with OCR as before. The 673-page book (36%) now takes that path. **The 25% threshold is a judgement.** It lies between the two measurements (4% was 2.6x faster, 36% was about 10% slower), and nothing in between was measured.
- A 5-page gap is an estimate from the two books' timings, not a tuned value.

The 673-page book was not re-run with the final fraction limit. It now takes the same document-routing path as the Iteration 4 run (423 s).

Recorded `ocr_used` per page now reflects bridged pages, so it matches what was actually OCR'd.

### 2. Tiny chunks are merged into the previous chunk

**Problem:** slide-style PDFs produced fragments such as `Side by side boxplot` / `X` (25 characters) and chunks holding only `<!-- formula-not-decoded -->`. They are noise for retrieval.

**Changed:** `merge_small_chunks()` in `chunk.py` folds any chunk with fewer than 50 characters of real text (`INGESTION_MIN_CHUNK_CHARS`) into the previous chunk, extending its page range.
- Docling's `formula-not-decoded` placeholders are removed from the text that is added, so a chunk of only placeholders adds nothing but its page number.
- A small chunk at the very start joins the next chunk instead.
- A merge that would exceed 512 tokens is skipped and the small chunk stays on its own, because the embedder truncates longer text and the tail would silently become unsearchable.
- Chunk IDs are computed after merging, so they stay stable for the same content.

The slide PDF went from 14 chunks to 10, none under 50 characters. A chunk with a placeholder alongside real text keeps it; only added text is cleaned.

### 3. Low-yield pages are flagged

**Problem:** nothing showed which pages contributed nothing to the index, so a document could finish as `completed` with several pages missing.

**Added**
- `document_pages.extracted_chars` and `low_yield` (in-place migration for existing databases). A page is low-yield when fewer than 50 characters (`PAGE_LOW_YIELD_CHARS`) of indexable text were extracted. They appear in `GET /documents/{id}/pages`, and the worker logs a warning listing them.
- Only text that can reach the index is counted. Docling nests text found inside a chart or handwriting under the picture, and the chunker skips it. My first version counted it and flagged almost nothing: pages 8 and 10 of the slide PDF looked fine because of chart axis ticks, yet produced no chunks. Counting only non-picture text flags pages 1, 3, 7, 8, 10, 12, 16 and 17 of that PDF.
- `extracted_chars` differs from `char_count`, which is the PDF's own text layer measured before conversion.

Limits: a blank page is also flagged, and a page with a few real lines plus a large chart may not be. The flag says text is missing; it does not recover it. Reading charts and handwriting would need a vision model, which is not added.

### Tests

77 to 79 tests pass. New: `tests/test_chunk.py` (merge rules, token limit), gap bridging, the OCR-fraction fallback, per-page extracted text including picture-nested text, low-yield records, and the registry migration. The existing routing tests that described document routing now select it explicitly.

---

## Iteration 4: Ingestion pipeline test run and fixes (2026-10-11)

The chunking, embedding, Qdrant and OpenSearch stages written since Iteration 3 were run end to end on real PDFs: 6 PDFs from 1 to 673 pages, through the real `process_document` path, against throwaway Qdrant and OpenSearch names and a temporary registry database (the real `rag_chunks` data and `documents.db` were not used). Every PDF finished with the same chunk count in Qdrant and OpenSearch. Conversion was 85% or more of the time on the large books: 548 pages took 367 s (317 s converting), 673 pages took 423 s (381 s converting); embedding and both store writes together were under 30 s.

### 1. The pipeline could not be imported

**Problem:** `pipeline.py`, `chunk.py` and `embed.py` imported `app.ingestion.schemas`, but the module is `app/ingestion/schema.py`. Every ingestion job would have failed with `ModuleNotFoundError` as soon as the worker reached the pipeline.

**Fixed:** the three imports now use `app.ingestion.schema`.

### 2. Re-ingesting a new version left the old chunks behind

**Problem:** the chunk ID includes the document version, so a new version never overwrites the old one. Ingesting version 2 of a document that already had version 1 left both in both stores (13 chunks became 26), and search would return duplicates.

**Added**
- `delete_stale_versions(document_id, keep_version)` in `app/db/qdrant.py` (filter delete) and `app/db/opensearch.py` (`delete_by_query`, refreshed, `conflicts="proceed"`).
- `ingest_converted_document()` calls both after every batch of the new version has been written.
  **Why after, not before:** if a write fails part-way, the previous version stays searchable and the retry can still overwrite the new version's IDs. A test confirms nothing is deleted when a write fails, or when the document produces no chunks.
- Re-ingesting the same version is unchanged: it writes the same IDs, so nothing is duplicated and the new cleanup deletes nothing. Checked against the real stores: versions 1, 1, 2 give 13, 13, 13 chunks.

### 3. The tokenizer was reloaded for every document

**Problem:** `iter_chunks()` called `build_chunker()` each time, which reloads the bge-m3 tokenizer from disk. It cost about 4.5 s per document, which was most of the time for small files (a 2-page PDF took 5.3 s, of which 0.2 s was embedding).

**Changed:** `build_chunker()` is wrapped in `lru_cache(maxsize=1)`, so each worker process builds it once. The worker is single-threaded and the chunker holds no per-document state.

### 4. Tests and dependencies

**Fixed:** three tests in `tests/test_worker.py` were already failing, because the worker now checks that the file exists, rejects non-PDF files with `NotImplementedError`, and runs the ingestion pipeline. They now use real temporary files and a stubbed pipeline, and the non-PDF test expects a failed status instead of "completed". Added tests for a missing file and an ingestion failure.

**Added:** `tests/test_pipeline.py` with Docling, the embedding model and both stores mocked. It covers batching, the delete-after-all-batches order, a failed write keeping the old version, an empty document, an embedding count mismatch, and the chunker being built once.

**Added to `requirements.txt`:** `docling`, `fastapi`, `python-multipart`, `opensearch-py`, `qdrant-client`, `sentence-transformers`, `torch`, `pypdf` and `python-dotenv`, pinned to the versions the test run used. None were listed before, and Docling was not installed in `.venv`.

### Found, not changed

- **Slow OCR on large books:** all pages of both large books were routed to OCR, which is where almost all the time went. The detector thresholds in `pdf_detector.py` or `PDF_ROUTING=page` are worth testing.
- **Lost content on slide-style PDFs:** "Analysis with Statistics" (17 pages) gave 14 chunks. Six pages produced none, one chunk is only `<!-- formula-not-decoded -->`, and several are under 50 characters. Image-only content such as plots is not captured.
- **Redundant round trips:** `ensure_collection()` and `ensure_index()` run on every batch. The embedding batch size is 2 against a pipeline batch of 16. Neither is a bottleneck at the measured speeds.

---

## Iteration 3: Duplicates, page-level data and page-level routing (2026-10-10)

The "no changes outside this stage" restriction was lifted for this iteration, so it touches the registry, the upload route and the worker as well as the stage itself. Chunking, embedding and Qdrant are still not implemented (those files are empty stubs).

### 1. Duplicate uploads

**Problem:** uploading the same file twice stored it twice, queued two ingestion jobs and, once chunking and embedding exist, would have written every chunk to the vector store twice.

**Added**
- `app/api/routes.py`: a SHA-256 of the file is computed while the upload streams to disk, at no extra read cost.
- `app/db/registry.py`: a `content_hash` column, with an in-place migration for existing databases (`ALTER TABLE`) and an index.
- `register_upload()` in `registry.py` checks for identical content and inserts the new row in one `BEGIN IMMEDIATE` transaction.
  **Why:** a separate check then insert would let two simultaneous uploads both pass the check. A test with 8 threads confirms exactly one is accepted, and the same test fails when the lock is removed.
- A duplicate upload now returns HTTP 200 with the existing `document_id`, `job_id` and `status` and `"duplicate": true`. The extra copy is deleted and no new job is queued. New uploads return `"duplicate": false`.
- `FAILED_STATUS`: a document with status `failed` does not count as a duplicate, so a failed ingestion can be retried by uploading the file again.

**Fixed (latent bug exposed by this change)**
- If enqueueing failed, the row was left as `queued` forever and the file was deleted. With duplicate detection, every retry would then have been told it duplicated a document that never ran. That row is now marked `failed`.

**Replaced**
- `create_document()` is no longer used by the upload route (`register_upload()` replaces it). The function itself is still in `registry.py`.

### 2. Page-level data

**Problem:** the per-page classification was computed and then thrown away. Downstream stages and operators could not tell which pages were OCR'd or why.

**Added**
- `document_pages` table with page number, type, character count, image counts, whether OCR was used and the reason. `save_pages()` replaces a document's rows, so a retried job does not create duplicates. `get_pages()` reads them back.
- `GET /documents/{id}/pages` returns the rows for a document.
- `PdfConversion` in `convert.py` (document, routing used, per-page OCR decision, inspection) and `PdfConversion.page_records()`.
- `convert_pdf()` returns a `PdfConversion`. `convert_document()` still returns just the Docling document, so existing callers are unaffected.
- `app/ingestion/worker.py` now runs `convert_pdf` for `.pdf` files and saves the page data. Other file types are skipped as before. Docling is imported inside the function so the API process, which imports the worker, does not need Docling installed.

### 3. Page-level OCR routing

**Problem:** with document-level routing, one scanned page in a 300-page digital PDF turned OCR on for all 300 pages.

**Added**
- `routing="page"` (or the `PDF_ROUTING=page` environment variable). Consecutive pages with the same OCR decision are converted together using Docling's `page_range` and the cached converter for that OCR mode, then merged with `DoclingDocument.concatenate`. This needs no PDF splitting and no temporary files.
- Facts checked against docling-core 2.101.1, not assumed:
  - `concatenate` rebases page numbers, so contiguous in-order runs covering the whole PDF come out numbered 1..N.
  - It drops `origin`, so `origin` and `name` are copied back from the first run.
- Safeguards: if a run does not finish with `SUCCESS`, returns the wrong number of pages, or the merged page numbers differ from the PDF's pages, the whole document is converted again with OCR on. That fallback is logged and recorded in `PdfConversion.routing`.
- PDFs where every page gets the same decision skip the merge and are converted whole.

**Default is still `document`.** Page routing is opt-in because it has not been run against real Docling models (see limitations).

### 4. Other changes

**Changed**
- `DetectorConfig.max_pages` now defaults to `None` (inspect every page). Page routing and the stored page data need a verdict for every page. Measured cost is about 4 ms per page, so a 400-page PDF takes roughly 1.6 s, small next to conversion.
- The `ConversionStatus` and `DoclingDocument` imports in `convert.py` are new. The `docling_core` package was already a dependency of Docling.

**Removed**
- `select_ocr()` from `convert.py`. Its logic moved into `convert_pdf()`, which needs the per-page decisions, not just a boolean. Nothing else imported it.

### Tests
- 52 tests in total (was 25): 14 for the detector, 20 for conversion, 10 for the registry, 5 for the upload route and 3 for the worker, including the mutation checks above.
- Project venv (Docling stubbed): 52 run, OK, 2 skipped (they need real Docling).
- Scratch venv with real `docling-slim==2.137.0`: 52 run, all passed. This includes a test that merges real `DoclingDocument` objects built as page runs and checks page numbers, text order and `origin`.
- `import app.main` still works without Docling installed.

### Limitations and open items
- **Page routing is unvalidated against real conversions.** The merge is verified on real `DoclingDocument` objects, but nobody has run real PDFs through the layout and OCR models here. Compare page-mode output against document-mode output on a few real mixed PDFs before enabling it. Content that spans a page boundary between two runs (for example a table split across pages) may differ from a single-pass conversion.
- **Duplicates are exact only.** The same document re-exported or re-scanned has different bytes and is not detected. Duplicate pages inside one document are not detected.
- **Old rows are not hashed.** Documents uploaded before this change have no `content_hash`, so re-uploading them is not caught. I did not backfill because it means reading every stored file at startup.
- **A duplicate is still fully uploaded before it is recognised**, since the hash needs the whole file. Only the extra copy is discarded. A client-supplied hash would avoid that.
- **The status `failed` is the only retry state.** A document stuck in `processing` after a crashed worker still blocks re-uploads.
- **Unchanged:** the detector heuristic limits listed under Iteration 1, the RQ fork caveat from Iteration 2, `requirements.txt` missing `docling` and `pypdf`, and the worker marking documents `completed` although chunking, embedding and storage are not implemented.

---

## Iteration 2: Optimization pass (2026-10-10)

Measure first, change only what the measurements justify.

### Measurements

| What | Result | Conclusion |
|---|---|---|
| Detector, all 400 pages (synthetic) | 1934 ms, about 4.8 ms/page | Cheap in absolute terms |
| Detector, default 50-page sample | 281 ms | Negligible next to Docling, which takes seconds to minutes per document |
| Docling `DocumentConverter` source (2.137.0) | Loaded pipelines and models are cached on the converter instance, keyed by (pipeline class, options hash) | Creating a new converter per call discards that cache |

The detector figures use synthetic text pages, which are lighter than real documents. Treat them as an order of magnitude, not a benchmark.

### Added
- `_get_converter(do_ocr)` in `convert.py`, wrapped in `functools.lru_cache(maxsize=2)`. It returns one `DocumentConverter` per OCR mode (at most two).
  **Why:** `convert_document` built a new `DocumentConverter` on every call, so Docling's layout and table models were reloaded for each document. Reusing the converter keeps them loaded. `maxsize=2` is exactly the number of distinct configurations (OCR on, OCR off).
- Test `test_converter_is_reused_per_ocr_mode` in `tests/test_convert.py`. It checks one construction per OCR mode across repeated calls.

### Changed
- `convert_document` now calls `_get_converter(do_ocr)` instead of building a converter inline. Behavior, return value, routing and logging are unchanged.
- `ConvertRoutingTests.setUp` clears the converter cache before and after each test. The cache is module-level state, so without this one test's mock would leak into the next.

### Removed
- Nothing.

### Considered and not done
- **Detector early exit** (stop at the first non-digital page). Saves at most about 0.3 s per document, which is noise next to OCR time. It would also remove the full per-page counts that the logging and the `PdfInspection` interface provide. Not worth the loss.
- **Faster table mode or other Docling tuning** (for example the FAST table-structure mode). It trades extraction quality for speed, and the stated priority is completeness. Not changed without an explicit decision.
- **Accelerator or device settings.** These depend on the deployment hardware, which isn't known.

### Caveat: the cache may not help under default RQ
RQ's default `Worker` forks a child process per job. The cache lives in the child and is discarded when the job ends, so each job would still pay model loading. It helps when the process stays alive: `SimpleWorker`, scripts, or a worker that preloads the converter before forking. The worker is out of scope here and was not changed. Decide this when the worker is wired up.

### Verification
- Project venv (Docling not installed, stubbed): 25 tests, OK, 1 skipped (needs real Docling).
- Scratch venv with real `docling-slim==2.137.0`: 25 tests, OK.
- Not verified: actual Docling conversion or model-load time. No model weights or torch were installed, so the saving from the cache is based on the source, not a measured number.

---

## Iteration 1: Page-level detection and OCR routing (2026-10-10)

### `app/ingestion/pdf_detector.py`: rewritten

**Removed**
- `detect_pdf_type(pdf_path, sample_pages=10, min_chars_per_page=40, min_text_page_ratio=0.5) -> str`.
  **Why:** it returned one string for the whole document, so the converter had no per-page data. Its `"mixed"` label also merged "mixed content" with "not enough evidence". No other code imported it.
- First-10-pages sampling.
  **Why:** a scanned section later in a long PDF was never seen.
- The `ratio` and `avg_chars` document-level decision.
  **Why:** averages hide mixed content. Pages are now judged individually.

**Added**
- `inspect_pdf(path, DetectorConfig) -> PdfInspection`. It classifies each page as `digital`, `scanned` or `ambiguous`.
- `DetectorConfig` holds all thresholds: `min_digital_chars=100`, `max_scanned_chars=10`, `large_image_pixels=250_000`, `max_pages=50`.
  **Why:** the heuristics should be explainable and tunable without editing code.
- `PageResult` records the type, character count, image counts and a human-readable reason per page.
- `PdfInspection` provides `count()`, `digital_ratio`, `all_digital` and `sampled`.
- Image evidence: image XObjects are counted from each page's resource dictionary without decoding them. A large image next to a text layer makes the page `ambiguous`.
  **Why:** the task notes that a page can have native text and a scanned table. Text alone can't show that. The check costs a dictionary lookup, with no rendering.
- A no-text page counts as `scanned` only if it has images; with no images it is `ambiguous`.
  **Why:** a blank page and an image-only page are different evidence.
- Even sampling across the document when `max_pages` is exceeded.
  **Why:** this replaces "first 10 pages" and covers the whole document at bounded cost.
- Text counting ignores whitespace, control characters and `U+FFFD`.
  **Why:** replacement characters aren't usable text.
- `PdfDetectionError` for empty, malformed, missing and password-protected files. An empty-password file is decrypted and read.
  **Why:** previously callers got a bare `ValueError` or raw pypdf exceptions.
- A page whose extraction throws becomes `ambiguous` instead of aborting the whole inspection.
- Logging of inspected/total pages and the digital/scanned/ambiguous counts. Document text is never logged.

**Known limits**
- The detector doesn't read pixels, so small images and images in form XObjects are missed.
- A bad text layer that still yields printable characters counts as digital.
- Sampled documents can miss a scanned page between sampled pages.

### `app/ingestion/convert.py`: updated

**Added**
- `select_ocr(pdf_path) -> bool`: returns False only when every inspected page is digital. If detection raises `PdfDetectionError`, it logs a warning and returns True.
  **Why:** a detection failure must never be treated as digital, and OCR-on is the safe fallback.
- `build_pdf_pipeline_options(do_ocr)`, built once and tested against the real Docling class.
- Logging of the chosen OCR mode and of conversion failures. Failures are re-raised unchanged.

**Changed**
- OCR routing now uses the page-level inspection instead of `pdf_type != "digital"`.
  **Why:** ambiguous pages, previously only `"mixed"`, now force OCR.

**Unchanged on purpose**
- Docling's standard `PdfFormatOption` pipeline, `do_table_structure=True`, `generate_page_images=False`, and `convert_document` returning `result.document`.
  **Why:** these already matched the requirements. Layout analysis stays on whether OCR is on or off. Checked against Docling 2.137.0 source.

### `tests/`: added
- `pdf_factory.py` builds test PDFs with pypdf's low-level API. `reportlab` isn't installed and adding it would be a new dependency.
- `test_pdf_detector.py` has 14 tests: digital, scanned, mixed, empty, malformed, missing, password-protected, digital text plus large image, digital text plus small image, blank page, short text, configurable thresholds, sampling, and empty-password.
- `test_convert.py` has 10 tests: routing for each case with a mocked Docling, the detector-failure fallback, error logging and re-raise, the return value, and one skippable test against the real `PdfPipelineOptions`.
- `unittest` was used because it needs no new dependency. Test runs: 24 passed in a venv with real `docling-slim`; 23 passed and 1 skipped in the project venv, where Docling is stubbed.
- The synthetic "image containing text" test only checks that a large image makes a page ambiguous. It doesn't check that the detector understands the image.

### Open items
- `requirements.txt` lists neither `docling` nor `pypdf`. It was left alone because dependency changes were out of scope.
- No real-world PDFs were available, so detector accuracy hasn't been measured.
- The worker doesn't call this stage yet.
