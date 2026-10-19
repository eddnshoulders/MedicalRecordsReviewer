# Medical Records Reviewer

A local-first tool for preparing medical-record PDFs and images, running OCR and candidate evidence extraction through LM Studio, reviewing page-level results, and searching the resulting dataset. The app serves its UI on `http://127.0.0.1:8765` and uses a local SQLite database under `data/` by default.

## Start the reviewer

From PowerShell in this folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\start.ps1
```

`start.ps1` reads `LM_API_TOKEN` from the current process or saved Windows user environment, and prompts for it with hidden input if LM Studio authentication is enabled and no token is available. The token is passed only to the local LM Studio server at `http://127.0.0.1:1234`. Open `http://127.0.0.1:8765` in a browser. Stop the server with Ctrl+C in its PowerShell window.

The app binds to loopback only. It does not send records to a cloud service. The database, imported originals, page images and OCR are stored locally under `data/` (or the directory selected by `REVIEWER_DATA`). These files can contain sensitive information and are not encrypted by the app; use disk encryption and suitable filesystem permissions.

## Import files

The **Files** page has a saved top-level import folder. Choose it with **Choose top-level folder…** or enter an absolute path; the path is stored in SQLite and restored on startup. **Import File** opens a file picker in that folder. If a PDF requires a password, the app prompts for it and does not save it. **Import Folder** opens a folder picker and imports supported PDFs and images from that folder; **Include subfolders** controls recursion.

Importing prepares and caches page images; it does not start OCR or evidence extraction. Supported image types are PNG, JPEG, TIFF and BMP. The original file bytes are preserved in the local dataset. Duplicate imports are identified by content hash.

## Settings and models

Expand **Settings** on the Files page to choose the available LM Studio models for OCR, NuExtract, MedGemma and embeddings. Context lengths can be set beside each model and are saved in SQLite. The selected context is used when the workflow loads a model. The setting to stop a stage when a page reaches the model output-token limit applies to manual and batch processing.

Default model identifiers are `lightonocr-3-4b`, `nuextract3` and `medgemma-1.5-4b-it`; the corresponding models must be installed in LM Studio. Model availability is checked before processing. This app does not download models.

## Page workflow

Select one or more files in the dataset table, then run OCR, NuExtract or MedGemma across the selection. Each stage skips completed pages and can resume incomplete or interrupted work. A completed stage must be explicitly cleared before it can be run again, which protects existing output from accidental replacement. **Stop all** requests a stop after the current LM Studio request; completed page results are retained.

Open a file by clicking its name to inspect its pages. Page buttons show the latest stage outcome; filters can hide selected states. The page view includes page-level run and delete controls and colour-coded OCR, NuExtract and MedGemma status. Page actions refresh the file’s progress values. Processing messages and issues are available in an expandable, vertically resizable section.

OCR is run for every included page. After OCR, HTML tables are converted to a compact readable representation; this derived text is shown alongside the OCR and supplied to both NuExtract and MedGemma. The original OCR transcription remains available and editable as a new version. The latest successful OCR is used for search and extraction. A page’s **NuExtract image** or **MedGemma image** control adds that page image to its next run alongside OCR text; it does not replace or clear the document’s existing results. Use **Ignore page** for a blank or unusable page. Ignoring a page removes its saved processing and review data; including it again allows it to be processed.

Candidates from NuExtract and MedGemma are stored with page and model provenance for review. Quotation mismatches are flagged rather than silently discarded. A completed job means requests completed; it does not guarantee OCR accuracy or that every relevant event was extracted.

## Search and semantic indexing

The **Search** page provides keyword, semantic and hybrid search. Keyword search uses SQLite FTS5 over the latest successful OCR. Semantic search uses a locally available LM Studio embedding model; use **Build / update semantic index** after selecting the model. OCR text is split into overlapping chunks, with each vector tied to its source page and OCR version. Re-OCRed pages need re-indexing before their new text appears in semantic results. Search results retain page provenance and can be opened for review.

## Logs and job progress

The **Log** page contains application messages, workflow jobs and folder activity. The workflow and activity panels can be expanded and resized vertically. Each file’s main-table stage indicators include completion and runtime. The app refreshes detailed dataset and page views while jobs are active and checks job state less often while idle.

## Reasoning-model tools

Read-only JSON endpoints under `/api/tools/` let a local reasoning model or local agent retrieve bounded evidence and source provenance without querying SQLite directly:

- `POST /api/tools/search_evidence` searches OCR and evidence with optional filters.
- `GET /api/tools/document_pages/{document_id}` maps source page numbers to stable page IDs.
- `POST /api/tools/get_source` returns latest OCR text and page provenance for requested pages.
- `POST /api/tools/get_timeline` returns candidate events in chronological order with their source references.
- `POST /api/tools/get_assessment_criteria` searches locally configured criteria. No regulatory criteria are preloaded; the endpoint does not fetch or invent rules.

These endpoints are intended for local use with the reviewer. A reasoning model should cite the returned document/page references and treat extracted events as candidates, not established facts.

## Data management

**Delete File and Data** removes the application-owned original copy and associated page images, OCR, search entries, evidence, reviews, jobs and runs after confirmation. It does not delete the external source file. Deleting a stage removes that stage’s results for the selected file or page. Deletion is not forensic secure erasure of SQLite free pages, filesystem snapshots or backups.

## Tests

Run the test suite from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use synthetic documents and mock model responses; actual extraction and embedding quality must be assessed with locally installed models.
