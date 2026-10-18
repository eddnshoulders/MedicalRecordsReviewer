# Local AI Medical-Records Review Tool

## Purpose

Build a privacy-preserving local tool for reviewing large collections of medical and legal records.

The system should **not attempt to replace professional judgement**. Its primary purpose is high-recall evidence discovery:

> Surface potentially consequential evidence, preserve exact provenance, organise it into a useful chronology, identify contradictions/gaps, and make human verification fast.

The immediate use case is an Infected Blood Compensation Authority (IBCA) claim, but the architecture should be sufficiently generic to reuse for other medical-record-heavy legal work.

---

# Core problem

Large medical-record bundles can contain thousands of pages across:

- native PDFs
- scanned PDFs
- image-only pages
- correspondence
- clinic letters
- laboratory results
- handwritten notes
- historical records
- duplicate documents
- documents with incorrect or incomplete metadata

A human reviewer can easily miss consequential evidence.

Examples already missed in a professional legal review include:

- first recorded Factor VIII treatment
- interferon treatment dates
- FibroScan / fibrosis evidence
- later liver imaging

The tool should make those sorts of omissions much less likely.

---

# Design principle

Optimise for:

## High recall first

It is preferable to show a reviewer 30 plausible candidate pages and have 20 be irrelevant than to miss the one page that materially changes a claim.

Precision can be improved later.

## Human verification

AI output should be treated as:

> candidate evidence

rather than:

> established fact

Every extracted claim must link back to the original source.

## Provenance everywhere

Never produce an important factual assertion without retaining:

- source document
- page number
- source text
- extraction method
- confidence / uncertainty
- relevant dates

Ideally also retain page/image hashes so results remain auditable.

---

# Intended architecture

```text
Raw files
    ↓
Document ingestion
    ↓
PDF security / normalisation
    ↓
Page extraction
    ↓
Native text extraction
    ↓
OCR / VLM fallback
    ↓
Canonical page store
    ↓
Search/indexing
    ↓
Candidate evidence extraction
    ↓
Structured event database
    ↓
Cross-document reconciliation
    ↓
IBCA-specific rules/checklists
    ↓
Human review interface / report
```

---

# Module 1 — File ingestion

Input:

- directory tree containing medical/legal records
- PDFs
- images
- possibly DOC/DOCX later

For every file record:

- original filename
- path
- SHA-256 hash
- file size
- page count
- creation/modification metadata if useful
- whether encrypted
- whether native text exists

Do not modify originals.

Create a working copy / derived-data directory.

---

# Module 2 — PDF normalisation

Need to handle:

- encrypted PDFs where the user possesses the opening password
- malformed PDFs
- rotated pages
- image-only PDFs
- mixed native-text/image PDFs

Possible tools:

- qpdf
- pypdf / pymupdf
- Ghostscript if required

Important:

The system should not destroy the original file.

Produce a normalised derivative and maintain a mapping back to the original.

---

# Module 3 — Page-level canonical dataset

Treat **page** as the fundamental unit.

Example structure:

```json
{
  "document_id": "...",
  "page_id": "...",
  "source_file": "...",
  "page_number": 42,
  "native_text": "...",
  "ocr_text": "...",
  "best_text": "...",
  "text_source": "native|ocr|vlm",
  "image_path": "...",
  "sha256": "...",
  "rotation": 0,
  "quality_score": 0.91
}
```

This enables all downstream results to cite an exact page.

---

# Module 4 — Text extraction

Use a staged approach.

## Stage A — Native PDF text

Extract text directly whenever possible.

This is:

- fastest
- cheapest
- generally most accurate

## Stage B — OCR

For image-only or poor-quality pages use local OCR.

Possible engines:

- OCRmyPDF
- Tesseract
- PaddleOCR
- another high-quality local OCR engine

OCR should be done page-by-page and cached.

## Stage C — Vision-language model fallback

Only use a VLM where necessary, for example:

- unusual page layouts
- handwriting
- badly degraded scans
- tables OCR cannot understand
- annotations
- forms with spatial relationships

Do not run expensive vision inference across every page unnecessarily.

---

# Module 5 — Search/index

Create both:

## Lexical search

Examples:

- SQLite FTS5
- BM25
- Elasticsearch/OpenSearch later if needed

This is valuable for exact concepts such as:

- interferon
- ribavirin
- Factorate
- Factor VIII
- FibroScan
- fibrosis
- cirrhosis
- ultrasound
- HCV
- hepatitis
- transaminitis
- ELF

## Semantic search

Create embeddings for pages/chunks to find concepts even where terminology differs.

Examples:

> evidence of liver scarring

might retrieve:

> increased hepatic stiffness

or:

> concern regarding fibrosis

Use local embeddings where possible.

Hybrid lexical + semantic search is preferable.

---

# Module 6 — Medical event extraction

Extract candidate events into structured form.

Example:

```json
{
  "event_type": "treatment",
  "date_start": "2002-12",
  "date_end": "2003-12",
  "date_precision": "month",
  "treatment": "Pegylated interferon + ribavirin",
  "condition": "Hepatitis C",
  "source_page_id": "...",
  "source_quote": "...",
  "confidence": 0.93,
  "review_status": "unreviewed"
}
```

Candidate event categories:

- diagnosis
- infection
- blood-product treatment
- antiviral treatment
- surgery
- investigation
- scan
- laboratory result
- medication
- adverse treatment effect
- clinician assessment
- functional impact
- education impact
- employment impact
- care requirement
- mental/cognitive symptoms
- severity change

---

# Module 7 — Date extraction and chronology

Dates are especially important.

Extract:

- exact dates
- month/year
- approximate dates
- relative dates

Examples:

> "five months ago"

should preferably be retained both as:

```text
source statement: five months
document date: 4 June 2003
inferred approximate start: January 2003
```

Do **not silently convert inference into fact**.

Chronology should distinguish:

- explicit dates
- inferred dates
- conflicting dates

---

# Module 8 — Contradiction detection

Identify competing evidence.

Example:

```text
Source A:
December 2002 – December 2003

Source B:
January 2003 start

Source C:
June 2003 letter says treatment had been underway for five months
```

Output:

```text
Possible discrepancy in third interferon course start date.

Evidence supports approximately Dec 2002–Jan 2003.
Requires human review.
```

This is more valuable than forcing a single answer.

---

# Module 9 — IBCA evidence checklist

Add a domain-specific layer for the infected-blood compensation scheme.

The tool should automatically search for evidence relevant to:

## Infection

- earliest blood-product treatment
- first Factor VIII / Factorate / cryoprecipitate exposure
- HCV tests
- diagnosis date
- possible infection windows
- treatment-centre records

## Hepatitis severity

- fibrosis
- cirrhosis
- FibroScan
- ELF
- biopsy
- ultrasound
- CT/MRI
- portal hypertension
- liver function tests
- transaminitis
- hepatoma surveillance
- AFP

## Interferon / Level 2B

Find every course of:

- interferon
- pegylated interferon
- ribavirin

Extract:

- treatment start
- treatment end
- duration
- treatment type
- stopping reason
- adverse effects
- recovery period

Especially flag:

- courses before age 16
- overlapping statutory treatment years
- courses crossing calendar-year boundaries

## Long-term treatment effects

Search for:

- fatigue
- exhaustion
- cognitive impairment
- brain fog
- memory
- concentration
- motivation
- depression
- mood changes
- sleep disturbance
- inability to sustain activity
- prolonged recovery

## Financial-loss / employment evidence

Extract:

- employment
- salary/rate
- job title
- contract dates
- gaps in employment
- reduced hours
- burnout/recovery periods
- career progression evidence

## Education

Extract:

- GCSE/A-level dates
- university
- examination disruption
- special consideration
- medical letters supporting exam impact
- delayed progression
- treatment overlapping education

## Care

Search for:

- family assistance
- care during treatment
- inability to perform normal activities
- unpaid support
- paid care

## Inquiry evidence

Identify references to:

- Inquiry witness statements
- exhibits
- medical evidence
- previous sworn/official accounts

---

# Module 10 — Evidence cards

The main human-review object should be an **evidence card**.

Example:

---

### Interferon treatment — June 2003 letter

**Candidate relevance:** Level 2B / treatment effects / education

**Document:** WITN1040013  
**Page:** 1  
**Document date:** 4 June 2003

**Extract:**

> "He has been on interferon and ribavirin for five months..."

**Potential significance:**

- supports treatment beginning approximately January 2003
- confirms significant treatment side effects
- explicitly links treatment to examination performance

**Confidence:** High

**Status:** ☐ Confirmed ☐ Rejected ☐ Needs investigation

---

The reviewer should always be able to open the original page.

---

# Module 11 — Claim chronology

Automatically generate something like:

| Date | Event | Evidence | Confidence |
|------|-------|----------|------------|
| 11 Oct 1982 | First recorded Factorate treatment | Medical record | High |
| 9 Sep 1996 | First interferon course started | Clinic record | High |
| ~Dec 1996 | First interferon course ended | estimated duration | Medium |
| Feb 2001 | Interferon + ribavirin started | medical record | High |
| Dec 2002 | Peg-IFN + ribavirin started | medical record | High |
| 4 Jun 2003 | Significant treatment side effects documented | consultant letter | High |
| Dec 2003 | final treatment completed | medical record | High |
| 2010 | FibroScan 5.5 kPa / ELF ~8.4 | hepatology letter | High |
| 2 Sep 2013 | liver ultrasound normal | ultrasound report | High |

Every row should link to evidence.

---

# Module 12 — Missing-evidence detection

Useful questions for the tool to ask:

- Is a treatment mentioned but no start date found?
- Is a start date present but no end date?
- Is a scan recommended but no result present?
- Is six-monthly surveillance recommended but records are absent?
- Are laboratory results referenced but not included?
- Does a clinician refer to another letter/document that is missing?
- Is there a large unexplained chronological gap?

Output something like:

```text
Possible missing evidence:
2010 hepatology letter recommends repeat FibroScan/ELF after one year.
No repeat result found in current corpus.
Consider requesting additional records.
```

---

# Module 13 — Question answering

Later add a local RAG interface.

Example queries:

> When did each interferon course start and end?

> Find every reference to fibrosis.

> What evidence exists that treatment affected education?

> What evidence predating the compensation scheme supports persistent fatigue?

> What is the earliest evidence of Factor VIII treatment?

Answers must provide page-level citations and preferably short verbatim excerpts.

No uncited answers.

---

# Module 14 — Report generation

Possible outputs:

## Evidence report

Organised by:

- infection
- severity
- treatment
- care
- financial loss
- education
- long-term effects

## Chronology

Fully cited timeline.

## Discrepancy report

Conflicting dates / diagnoses / treatment durations.

## Missing-record report

Potential missing documents or investigations.

## IBCA checklist

For each potentially relevant requirement:

```text
Requirement: First blood-product treatment
Evidence found: YES
Best evidence: [link]

Requirement: Interferon treatment
Evidence found: YES
Courses detected: 3

Requirement: Fibrosis evidence
Evidence found: YES
Best evidence: 2010 hepatology letter
```

---

# Model strategy

Avoid using one huge model for everything.

Use a pipeline.

## Cheap deterministic processing

Use for:

- PDF parsing
- page detection
- regex/date extraction
- keyword search
- deduplication
- OCR

## Small/medium local models

Use for:

- document classification
- relevance scoring
- named entities
- medical event extraction
- summarisation
- chronology construction

## Larger model

Use selectively for:

- ambiguous evidence
- cross-document reasoning
- contradiction analysis
- difficult medical language
- generating final summaries

This makes local inference practical.

---

# Privacy

This system handles highly sensitive medical data.

Default requirement:

> No source medical data should leave the local machine.

Prefer:

- local OCR
- local embeddings
- local LLM/VLM
- encrypted storage
- local database
- no telemetry
- no cloud logging

Development assistance from Codex/Claude/ChatGPT can be used to write the software without sending the medical corpus itself.

Any optional cloud inference must be explicitly enabled rather than default behaviour.

---

# Auditability

Maintain a processing log containing:

- file hash
- processing date
- software/model versions
- extraction method
- prompts/templates if applicable
- model outputs
- human review decisions

AI-generated material should always be distinguishable from source evidence.

---

# Benchmark / demonstration

A major purpose of the prototype is to measure whether inexpensive AI-assisted review can reduce consequential omissions.

Create a labelled benchmark based on already-known facts.

Examples of **must-find evidence**:

1. first Factorate treatment — 11 Oct 1982
2. first interferon course — 9 Sep 1996
3. second interferon course — Feb 2001
4. final peg-interferon/ribavirin course — Dec 2002–Dec 2003
5. significant side effects during final treatment
6. examination impact
7. 2010 FibroScan 5.5 kPa
8. ELF result
9. mild transaminitis
10. recommendation for liver surveillance
11. 2013 normal liver ultrasound
12. persistent fatigue/cognitive effects
13. employment/burnout/recovery evidence

Measure:

- recall of known consequential evidence
- precision
- pages requiring human review
- processing time
- reviewer time
- hardware used
- development time

The principal metric should initially be:

> **Did the system surface every known consequential item?**

---

# Useful experiment

Run the pipeline **blind** against the complete record bundle.

Do not give the extraction system the page locations of known evidence.

Then compare:

```text
Known consequential evidence: 13 items
Automatically surfaced: 13
Missed: 0

Corpus size: X pages
Candidate pages requiring review: Y
Reduction in human reading burden: Z%
```

This creates an empirical demonstration rather than merely an argument about AI.

---

# MVP

The first version does not need a sophisticated UI.

A one-day MVP could consist of:

1. recursively ingest PDFs
2. extract native text
3. render/OCR pages lacking useful text
4. store page-level text in SQLite
5. create FTS search index
6. run predefined IBCA keyword/concept searches
7. use a local or selected LLM to classify candidate pages
8. extract structured events
9. output:
   - HTML/Markdown chronology
   - evidence cards
   - source/page references
   - IBCA checklist
10. manually verify results

The MVP should optimise for **proof of capability**, not polish.

---

# Suggested project layout

```text
records-ai/
│
├── data/
│   ├── originals/
│   ├── normalized/
│   ├── pages/
│   ├── ocr/
│   └── derived/
│
├── src/
│   ├── ingest/
│   ├── pdf/
│   ├── ocr/
│   ├── indexing/
│   ├── search/
│   ├── extraction/
│   ├── models/
│   ├── reconciliation/
│   ├── domains/
│   │   └── ibca/
│   └── reporting/
│
├── db/
│   └── records.sqlite
│
├── config/
│   ├── ibca_terms.yaml
│   ├── event_types.yaml
│   └── models.yaml
│
├── tests/
│   ├── benchmark/
│   └── unit/
│
└── reports/
```

---

# Important implementation principle

Separate the generic engine from IBCA-specific knowledge.

For example:

```text
Generic:
    ingest
    OCR
    page store
    indexing
    event extraction
    chronology
    provenance
    contradiction detection

Domain plugin:
    IBCA terminology
    eligibility concepts
    evidence checklist
    Level 2B concepts
    severity evidence
    financial-loss concepts
```

That makes the tool reusable for other legal/medical record-review tasks.

---

# Longer-term possibilities

Once the basic system works:

- local web UI
- clickable PDF/page viewer
- vector search
- graph of people/events/documents
- medical-code recognition
- automatic duplicate detection
- handwriting VLM
- multi-model consensus
- active learning from reviewer decisions
- document clustering
- automatic SAR gap analysis
- legal-rule engine
- comparison against claim declarations
- automatic identification of facts missing from a submitted claim
- support for multiple claimants
- role-based access controls
- encrypted network deployment
- GPU inference server

---

# Central hypothesis

The project is testing:

> **Can an inexpensive, privacy-preserving and auditable AI-assisted screening system substantially reduce the probability that consequential evidence is missed during large-scale medical-record review, while leaving substantive judgement with humans?**

The system does not need perfect AI.

It only needs to demonstrate that:

> **machine-assisted high-recall screening + targeted human verification**

is materially safer and more efficient than relying on humans to manually discover every important item within thousands of pages.

---

# Architectural note

**Page-level provenance should be treated as a non-negotiable primitive from the start**, rather than something added later.

If every extraction, event and inference ultimately resolves back to:

```text
document_id + page_id + exact source text
```

then most of the difficult auditability problem becomes much easier.
