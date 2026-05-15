from __future__ import annotations

import io
import json
import re
import tempfile
import zipfile
from pathlib import Path

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from src.pdf_parsing_pipeline import (
    LangGraphPDFEvaluationPipeline,
    PDFPipelineConfig,
    compute_page_metrics,
)


app = FastAPI(title="PDF Parsing Evaluator API", version="0.1.0")


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem.strip() if filename else "document"
    return stem or "document"

def _parse_parsed_md_pages(md_text: str) -> dict[int, str]:
    """
    Parse this project's generated Markdown format into {page_number: page_text}.

    Expected structure:
    - "# <pdf_name>"
    - Repeated blocks:
        "## Page N"
        "_Source: ..._"
        optional "Validation" section
        page text...
    """
    pages: dict[int, str] = {}
    current_page: int | None = None
    buffer: list[str] = []

    for raw_line in md_text.splitlines():
        line = raw_line.rstrip("\n")
        match = re.match(r"^##\s+Page\s+(\d+)\s*$", line)
        if match:
            if current_page is not None:
                pages[current_page] = "\n".join(buffer).strip()
            current_page = int(match.group(1))
            buffer = []
            continue

        if current_page is None:
            continue

        # Skip our metadata / validation scaffolding.
        if re.match(r"^_Source:\s+.*_$", line):
            continue
        if line.strip() == "Validation":
            continue
        if re.match(r"^- Page \d+ .+", line):
            continue

        buffer.append(line)

    if current_page is not None:
        pages[current_page] = "\n".join(buffer).strip()

    return pages


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/parse")
async def parse_pdf(
    pdf: UploadFile = File(...),
    llm_cleanup_enabled: bool = False,
    ocr_enabled: bool = False,
    max_workers: int = 4,
    max_pages: int | None = None,
) -> Response:
    """Parse an uploaded PDF and return a zip containing `<name>.md` and `<name>.json`."""

    if not pdf.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    pdf_name = _safe_stem(pdf.filename)

    with tempfile.TemporaryDirectory(prefix="pdf-parse-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        pdf_path = tmp_root / f"{pdf_name}.pdf"
        bytes_written = 0
        with pdf_path.open("wb") as f:
            while True:
                chunk = await pdf.read(1024 * 1024)  # 1MiB
                if not chunk:
                    break
                bytes_written += len(chunk)
                f.write(chunk)
        if bytes_written == 0:
            raise HTTPException(status_code=400, detail="Empty upload")

        parse_pdf_path = pdf_path
        if max_pages is not None:
            try:
                max_pages_int = int(max_pages)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="max_pages must be an integer")
            if max_pages_int < 1:
                raise HTTPException(status_code=400, detail="max_pages must be >= 1")

            with fitz.open(pdf_path) as doc:
                if max_pages_int < len(doc):
                    trimmed_path = tmp_root / f"{pdf_name}__first_{max_pages_int}.pdf"
                    trimmed = fitz.open()
                    trimmed.insert_pdf(doc, from_page=0, to_page=max_pages_int - 1)
                    trimmed.save(trimmed_path)
                    trimmed.close()
                    parse_pdf_path = trimmed_path

        config = PDFPipelineConfig(
            parsed_output_dir=tmp_root / "parsed_output",
            ground_truth_dir=tmp_root / "ground_truth",
            evaluation_dir=tmp_root / "evaluation",
            cache_dir=tmp_root / "cache",
            llm_cleanup_enabled=llm_cleanup_enabled,
            ocr_enabled=ocr_enabled,
            generate_ground_truth=False,
            max_workers=max(1, int(max_workers)),
        )

        pipeline = LangGraphPDFEvaluationPipeline(config=config)
        try:
            result = pipeline.run(parse_pdf_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to parse PDF: {exc}") from exc

        md_path = Path(result["parsed_output_path"])
        md_text = md_path.read_text(encoding="utf-8")

        json_payload = {
            "pdf_name": result["pdf_name"],
            "parsed_pages": result["parsed_pages"],
            "metrics": result["metrics"],
            "issues": result["metrics"].get("issues", []),
        }
        json_text = json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n"

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{pdf_name}.md", md_text)
            zf.writestr(f"{pdf_name}.json", json_text)
        zip_bytes = zip_buffer.getvalue()

    headers = {"Content-Disposition": f'attachment; filename="{pdf_name}.zip"'}
    return Response(content=zip_bytes, media_type="application/zip", headers=headers)


@app.post("/evaluate")
async def evaluate_parsed_md_accuracy(
    pdf: UploadFile = File(...),
    md: UploadFile = File(...),
    llm_cleanup_enabled: bool = False,
    ocr_enabled: bool = False,
    max_workers: int = 4,
    max_pages: int | None = None,
) -> dict[str, object]:
    """
    Upload a PDF + an already-parsed `.md` file, then score how closely the `.md`
    matches the parsing produced by this service.

    Returns page-wise and aggregate metrics (WER/CER/line-F1/token coverage).
    """

    if not pdf.filename:
        raise HTTPException(status_code=400, detail="Missing pdf filename")
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported for pdf")

    if not md.filename:
        raise HTTPException(status_code=400, detail="Missing md filename")
    if not md.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are supported for md")

    pdf_name = _safe_stem(pdf.filename)
    md_bytes = await md.read()
    if not md_bytes:
        raise HTTPException(status_code=400, detail="Empty md upload")

    try:
        md_text = md_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"md must be utf-8 text: {exc}") from exc

    provided_by_page = _parse_parsed_md_pages(md_text)
    if not provided_by_page:
        raise HTTPException(
            status_code=400,
            detail="Unable to find any '## Page N' sections in the provided md file",
        )

    with tempfile.TemporaryDirectory(prefix="pdf-eval-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        pdf_path = tmp_root / f"{pdf_name}.pdf"
        bytes_written = 0
        with pdf_path.open("wb") as f:
            while True:
                chunk = await pdf.read(1024 * 1024)  # 1MiB
                if not chunk:
                    break
                bytes_written += len(chunk)
                f.write(chunk)
        if bytes_written == 0:
            raise HTTPException(status_code=400, detail="Empty pdf upload")

        parse_pdf_path = pdf_path
        if max_pages is not None:
            try:
                max_pages_int = int(max_pages)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="max_pages must be an integer")
            if max_pages_int < 1:
                raise HTTPException(status_code=400, detail="max_pages must be >= 1")

            with fitz.open(pdf_path) as doc:
                if max_pages_int < len(doc):
                    trimmed_path = tmp_root / f"{pdf_name}__first_{max_pages_int}.pdf"
                    trimmed = fitz.open()
                    trimmed.insert_pdf(doc, from_page=0, to_page=max_pages_int - 1)
                    trimmed.save(trimmed_path)
                    trimmed.close()
                    parse_pdf_path = trimmed_path

        config = PDFPipelineConfig(
            parsed_output_dir=tmp_root / "parsed_output",
            ground_truth_dir=tmp_root / "ground_truth",
            evaluation_dir=tmp_root / "evaluation",
            cache_dir=tmp_root / "cache",
            llm_cleanup_enabled=llm_cleanup_enabled,
            ocr_enabled=ocr_enabled,
            generate_ground_truth=False,
            max_workers=max(1, int(max_workers)),
        )

        pipeline = LangGraphPDFEvaluationPipeline(config=config)
        try:
            result = pipeline.run(parse_pdf_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to parse PDF: {exc}") from exc

    parsed_by_page = {item["page"]: item["text"] for item in result["parsed_pages"]}
    overlap_pages = sorted(set(parsed_by_page) & set(provided_by_page))
    if not overlap_pages:
        raise HTTPException(
            status_code=400,
            detail="No overlapping page numbers between parsed PDF and provided md",
        )

    totals = {
        "char_edits": 0,
        "char_total": 0,
        "word_edits": 0,
        "word_total": 0,
        "matched_lines": 0,
        "pred_line_total": 0,
        "gt_line_total": 0,
        "token_coverage": 0.0,
    }
    page_wise: list[dict[str, object]] = []
    issues: list[str] = []

    for page_number in overlap_pages:
        page_metrics = compute_page_metrics(parsed_by_page[page_number], provided_by_page[page_number])
        for key in (
            "char_edits",
            "char_total",
            "word_edits",
            "word_total",
            "matched_lines",
            "pred_line_total",
            "gt_line_total",
        ):
            totals[key] += int(page_metrics[key])
        totals["token_coverage"] += float(page_metrics["token_coverage"])

        page_wise.append(
            {
                "page": page_number,
                "wer": round(float(page_metrics["wer"]), 4),
                "cer": round(float(page_metrics["cer"]), 4),
                "line_precision": round(float(page_metrics["line_precision"]), 4),
                "line_recall": round(float(page_metrics["line_recall"]), 4),
                "line_f1": round(float(page_metrics["line_f1"]), 4),
                "token_coverage": round(float(page_metrics["token_coverage"]), 4),
            }
        )

    if len(overlap_pages) != len(parsed_by_page):
        issues.append(
            "Some parsed pages were not found in the provided md (page numbers may be missing)"
        )
    if len(overlap_pages) != len(provided_by_page):
        issues.append(
            "Some md pages were not found in the parsed output (md may contain extra/mismatched pages)"
        )

    # Aggregate.
    line_precision = (
        totals["matched_lines"] / totals["pred_line_total"] if totals["pred_line_total"] else 0.0
    )
    line_recall = totals["matched_lines"] / totals["gt_line_total"] if totals["gt_line_total"] else 0.0
    line_f1 = (
        2 * line_precision * line_recall / (line_precision + line_recall)
        if (line_precision + line_recall)
        else 0.0
    )

    aggregate = {
        "wer": round(totals["word_edits"] / totals["word_total"], 4) if totals["word_total"] else 0.0,
        "cer": round(totals["char_edits"] / totals["char_total"], 4) if totals["char_total"] else 0.0,
        "line_precision": round(line_precision, 4),
        "line_recall": round(line_recall, 4),
        "line_f1": round(line_f1, 4),
        "token_coverage": round(totals["token_coverage"] / len(overlap_pages), 4),
    }

    return {
        "pdf_name": result["pdf_name"],
        "evaluated_pages": len(overlap_pages),
        "metrics": aggregate,
        "page_wise": page_wise,
        "issues": issues,
    }
