# Medical Records Reviewer — first working slice

Local PDF ingestion, OCR-first page images, SQLite FTS5 and semantic search over OCR, LM Studio evidence extraction, quotation validation, review decisions and resumable extraction jobs.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\start.ps1
```

Open http://127.0.0.1:8765. Enter a full local PDF path. Password-protected PDFs accept an opening password which is not saved. Importing the same file resumes failed page extraction and otherwise deduplicates by SHA-256.

For model extraction, start the LM Studio local server on port 1234, load a downloaded text model with at least 8192 context tokens, and click Find models or enter its exact model identifier. No models are downloaded by this app. Only the loopback LM Studio endpoint at http://127.0.0.1:1234 is used; HTTP proxies are disabled. Other model servers are not wired up yet.

OCR versions are stored separately from existing PDF text. The latest successful OCR transcription is automatically used for search and evidence extraction; evidence runs retain their OCR version ID. The SQLite FTS5 index is rebuilt from the latest successful OCR when the app starts, so old or failed OCR versions do not remain searchable. Embedded PDF text is preserved only as an unverified comparison reference.

The original is copied byte-for-byte to `data/originals`; page images and the SQLite database remain in `data`. This contains sensitive information and is not encrypted by the app. Use OS disk encryption and appropriate filesystem access controls. Bind only to loopback. No telemetry, external scripts, fonts or cloud APIs are used.

## Search

Search offers keyword, semantic and hybrid modes. Keyword search uses SQLite FTS5 over the latest successful OCR. Semantic search uses a local embedding model through LM Studio's `/v1/embeddings` endpoint; choose a downloaded embedding model, then click **Build / update semantic index**. OCR is split into overlapping chunks, and each vector is tied to its source page and OCR version. Re-OCRed pages are excluded from semantic results until re-indexed. Hybrid search combines lexical and semantic ranks. All modes support document, evidence review-status and extractor filters and open results at the source page. Embedding requests stay on the fixed loopback LM Studio endpoint; no remote service or embedding package is used.

## Local reasoning-model tools

The app exposes read-only JSON tools under `/api/tools/` for a local reasoning model or a local agent client to call. The model does not query SQLite directly: the app retrieves bounded results and provenance for each request. `POST /api/tools/search_evidence` accepts `{"query":"...","filters":{"document_id":"...","evidence_status":"confirmed","source_stage":"evidence"},"mode":"hybrid","limit":20}`; `filters` is optional and may contain `document_id`, `evidence_status` and `source_stage`. Semantic and hybrid modes require a built semantic index. `GET /api/tools/document_pages/{document_id}` returns the mapping from one-based source page numbers to stable page IDs. `POST /api/tools/get_source` accepts a document ID and one-based page numbers and returns the latest successful OCR text, OCR model/version, page ID, and whether the source page image is available. `POST /api/tools/get_timeline` returns extracted candidate events sorted chronologically, retaining their dates as written, source quotes, review status, model and page provenance; uncertain and non-ISO dates remain unnormalized. `POST /api/tools/get_assessment_criteria` searches locally configured criteria by topic and includes source/version metadata. Since no verified regulatory rules are configured yet, it explicitly returns an empty result; it never fetches or invents rules. These routes are intended for loopback use alongside LM Studio and do not send records to a cloud service. The model should cite returned document/page references and treat extracted events as candidates, not established facts.

For OCR, the app requests a 16,384-token context with an 8,192-token output cap. NuExtract and MedGemma requests have a 4,096-token output cap; their input allocation is the loaded LM Studio context length minus that cap. The interface shows each stage's per-request allocation and each job's highest observed input/output token counts so far. Job peaks reset when that stage starts or is retried. LM Studio must expose the loaded model context in its local `/api/v1/models` response for the exact allocation to be shown.

## Review workflow

### Dataset management and manual stages

Importing a file or folder automatically prepares and caches its page images. The automatic-processing checkbox additionally runs OCR and evidence extraction; when unchecked, the file is prepared for later manual stages. In **Dataset and workflow**, each document has controls to run OCR, start evidence runs, and review candidates. OCR must successfully transcribe every page before evidence extraction; it is the default source for search and evidence. For encrypted single files, enter the opening password in the masked password field during import. Folder imports report files that could not be prepared and continue with the rest.

**Batch folder run** opens a local Windows folder picker and processes supported files without per-file prompts. It accepts PDFs and PNG, JPEG, TIFF or BMP images; each image becomes a one-page document while the original bytes are preserved. Include subfolders is optional. It prepares every file first, then runs OCR across the whole batch, then primary evidence extraction across all OCR-ready files, followed by the optional MedGemma pass across those files. Grouping work by model avoids repeatedly switching models for each file. The report records batch start, finish and elapsed time for benchmarking. Failures are shown per file and do not stop later files; encrypted PDFs without a supplied password are reported as errors. Batch progress and the latest report remain available after a browser refresh. Start LM Studio's local server and enter the model identifiers first. When the models do not all fit in memory, LM Studio's Just in Time Model Loading and auto-unload settings can switch models between these batch-wide phases.

MedGemma is an optional second extraction pass. Enter its LM Studio model ID in the separate MedGemma field for a folder batch, or use **Run MedGemma as additional pass** on the selected document. It runs on OCR text through the text extraction API; it does not receive page images or an OCR reasoning setting. Keep LightOnOCR selected in the OCR model field. A MedGemma text model cannot be submitted as the OCR model. MedGemma creates separate evidence candidates with its model and run provenance. Candidates are intentionally not de-duplicated across extractors; possible replications appear as separate cards for review.

**Delete from dataset** asks for confirmation and removes the application-owned PDF copy, page images, text, FTS entries, evidence, review decisions, jobs, runs, and failed-attempt history. The external source file is not deleted. Other documents are unaffected. Deletion is blocked while processing is active. If filesystem removal fails (for example a file is locked), database records remain so deletion can be retried; some artifacts may already have been removed. This is logical deletion, not forensic secure erasure of SQLite free pages or backups.

If LM Studio has API authentication enabled, `start.ps1` first checks the current process environment, then the saved Windows user environment variable `LM_API_TOKEN`, and finally prompts with hidden input if neither is set. This also works when the PowerShell window predates setting the user variable. The token is held in the reviewer process environment and sent only to the fixed local LM Studio endpoint. Load the desired model in LM Studio before extraction. Old Ollama jobs cannot be resumed through LM Studio; start a new job, preserving earlier evidence.

1. Import a PDF and prepare its pages, which creates source images and preserves any embedded PDF text for comparison.
2. Run OCR on every page using an installed local OCR model. The latest successful OCR transcription is automatically indexed and used as evidence source text.
3. Run candidate extraction using an installed local LM Studio model.
4. Select a page, compare its image and OCR transcription with candidate quotations, then save review decisions.
5. Retry interrupted/failed jobs; completed pages are skipped. A new extraction job retains earlier results as separate evidence runs.

Candidate quotations are compared character-for-character with the latest successful OCR text. A mismatch is kept with the candidate and flagged for checking against the source image; it no longer discards the page's other candidates. Exact OCR matches are marked in the review card. Model dates/descriptions remain unverified candidates. A completed OCR job means transcription requests completed, not that their contents are accurate. A completed evidence job does not prove all consequential evidence was found. Very long pages are explicitly failed rather than silently truncated.

## Current limits

This milestone supports individual PDFs, OCR-first text, a fixed LM Studio adapter, local semantic indexing, and page-level review. Directory ingestion, Lemonade adapter, cross-page reconciliation, chronology, exports, bounding-box highlights, model version fingerprints, and extraction-version migrations remain future work. The worker is single-process and handles one processing task at a time; do not run multiple server workers. Import is synchronous, extraction runs in a background thread. Failed model attempts are retained in an append-only failure history; retries update the current run status while successful runs and review history are preserved.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests create synthetic PDFs only and use mock models. They verify source preservation, deduplication, scan flags, lexical and semantic search, quote-mismatch warnings, independent MedGemma candidates and resumability. Actual MedGemma and embedding quality requires separately installed models and local benchmarks.




## OCR with LightOnOCR in LM Studio

1. Restart the reviewer after updating and refresh the browser.
2. Prepare the document's pages. Existing page images can be reused.
3. Load `lightonocr-3-4b` in LM Studio, with its vision projector available. Use enough context for the image and up to 8192 output tokens (start with 16384 context).
4. Click **Find OCR models**, or enter the exact model identifier in the separate OCR field.
5. Click **Run OCR on all pages** for the document. This reads image pixels rather than the PDF text layer. OCR jobs run sequentially, show progress and can retry failed pages without repeating completed pages.
6. Open a page in Review and click **Refresh OCR results**. Compare the original image, existing PDF text and OCR versions. The newest successful OCR version is used automatically.
7. After OCR succeeds on every page, start evidence extraction with the separate evidence model. No per-page text selection is needed.

Each fresh OCR job creates new versions without deleting prior output; the latest successful version becomes the automatic working source. Empty or truncated responses are failures. OCR sends the page image alone, explicitly disables reasoning through LM Studio’s native chat API, and stores only message output as Markdown. Reasoning output is excluded from transcription. It cannot guarantee handwriting accuracy. Images currently use the 140-DPI rendering from page preparation. Automatic discrepancy detection, higher-resolution rerender controls and region crops are future work.
