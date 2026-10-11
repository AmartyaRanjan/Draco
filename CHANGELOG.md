# Changelog: PDF detection and Docling conversion stage

Newest entry first. Iterations 1 and 2 were limited to `app/ingestion/pdf_detector.py`,
`app/ingestion/convert.py` and `tests/`; Iteration 3 also touches the registry, upload route and worker.

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
