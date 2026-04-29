from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from src.pdf_parsing_pipeline import LangGraphPDFEvaluationPipeline, PDFPipelineConfig


app = FastAPI(title="PDF Parsing Evaluator API", version="0.1.0")


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem.strip() if filename else "document"
    return stem or "document"


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
        result = pipeline.run(parse_pdf_path)

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
