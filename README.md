# Parsing Evaluator (PDF -> Markdown + JSON)

Parse PDFs into a page-by-page Markdown file and a structured JSON payload.

This repo includes:

- A parsing pipeline built on PyMuPDF, with optional LLM cleanup and OCR fallbacks.
- A FastAPI service that exposes the pipeline as a single upload endpoint.

## Quickstart

```bash
. .venv/bin/activate
uv sync
uvicorn src.api.main:app --reload --port 8000
```

Health check:

```bash
curl -sS http://127.0.0.1:8000/health
```

## API

### `POST /parse`

Upload a PDF and receive a zip containing:

- `<pdf_name>.md`: Markdown with one section per page
- `<pdf_name>.json`: JSON with `parsed_pages` and `metrics`

Multipart fields:

- `pdf` (file, required): the PDF to parse

Query parameters:

- `llm_cleanup_enabled` (bool, default `false`): enable Ollama/Groq cleanup fallback
- `ocr_enabled` (bool, default `false`): enable OCR fallback (requires OCR deps)
- `max_workers` (int, default `4`): page parsing concurrency
- `max_pages` (int, optional): only parse the first N pages (recommended for very large PDFs)

Example:

```bash
curl -sS -X POST "http://127.0.0.1:8000/parse?max_workers=4" \
  -F "pdf=@pdfs/FAR_06.pdf" \
  -o FAR_06.zip
```

Inspect:

```bash
unzip -l FAR_06.zip
```

Large PDFs (first 10 pages only):

```bash
curl -sS -X POST "http://127.0.0.1:8000/parse?max_pages=10&max_workers=4" \
  -F "pdf=@pdfs/EM%20385-1-1_15%20March%202024.pdf" \
  -o EM_385_first_10.zip
```

### `GET /health`

Returns:

```json
{"status":"ok"}
```

## Pipeline (CLI)

The pipeline can also be run directly (writes results to `parsed_output/`, `ground_truth/`, and `evaluation/` by default):

```bash
. .venv/bin/activate
python pdf_paring.py pdfs/FAR_06.pdf
```

Notes:

- The CLI can generate ground-truth pages (via Ollama/Groq) and compute metrics.
- The API endpoint disables ground-truth generation by default and returns only parsing outputs.

## Output Format

### Markdown (`<name>.md`)

- Title header: `# <pdf_name>`
- One section per page: `## Page N`
- Source and quality stats per page
- Extracted text (plus a “Links” section when present)

### JSON (`<name>.json`)

Contains:

- `pdf_name`
- `parsed_pages`: list of `{page, text, source, quality, links, issues}`
- `metrics`: evaluation structure (when ground-truth is present, otherwise it will contain an “No ground truth...” issue)

## LLM / OCR Options

By default, the API runs with `llm_cleanup_enabled=false` and `ocr_enabled=false`.

LLM cleanup / ground-truth:

- Ollama: uses `OLLAMA_HOST`, `OLLAMA_GT_MODEL`, `OLLAMA_TIMEOUT_SECONDS`
- Groq fallback (optional): `GROQ_API_KEY`, `GROQ_GT_MODEL`, `GROQ_BASE_URL`

OCR fallback requires additional system/python dependencies (not installed by default in this repo). If you enable `ocr_enabled=true` without OCR deps present, the pipeline will record an issue and continue.

## Mermaid (Flow)

```mermaid
flowchart TD
  A[Client] -->|POST /parse (pdf)| B[FastAPI /parse]
  B --> C[Stream upload to temp file]
  C --> D{max_pages?}
  D -->|no| E[Parse full PDF]
  D -->|yes| F[Trim to first N pages]
  E --> G[LangGraphPDFEvaluationPipeline]
  F --> G
  G --> H[Per-page parse via PyMuPDF]
  H -->|optional| I[LLM cleanup fallback]
  H -->|optional| J[OCR fallback]
  G --> K[Build Markdown]
  G --> L[Build JSON]
  K --> M[Zip: .md + .json]
  L --> M
  M -->|application/zip| A
```

## Repo Layout

- `src/api/main.py`: FastAPI app (`/health`, `/parse`)
- `pdf_parsing_pipeline.py`: pipeline implementation
- `pdf_paring.py`: CLI runner (batch)
- `pdfs/`: sample PDFs for local testing
- `api_outputs/`: optional local artifacts from calling the API (ignored by default)
