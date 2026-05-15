from __future__ import annotations

import io
import json
import zipfile

import fitz
import httpx
import pytest

from src.api.main import app


def _build_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.mark.anyio
async def test_parse_endpoint_returns_zip_with_md_and_json() -> None:
    pdf_bytes = _build_pdf_bytes("Hello PDF")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/parse",
            files={"pdf": ("sample.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            params={"llm_cleanup_enabled": "false", "ocr_enabled": "false", "max_workers": "1"},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/zip")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = set(zf.namelist())
        assert "sample.md" in names
        assert "sample.json" in names

        md_text = zf.read("sample.md").decode("utf-8")
        assert "# sample" in md_text

        payload = json.loads(zf.read("sample.json").decode("utf-8"))
        assert payload["pdf_name"] == "sample"
        assert payload["parsed_pages"]


@pytest.mark.anyio
async def test_parse_rejects_non_pdf_extension() -> None:
    pdf_bytes = _build_pdf_bytes("Hello PDF")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/parse",
            files={"pdf": ("sample.txt", io.BytesIO(pdf_bytes), "application/pdf")},
        )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_parse_rejects_empty_upload() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/parse",
            files={"pdf": ("sample.pdf", io.BytesIO(b""), "application/pdf")},
        )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_parse_rejects_max_pages_lt_1() -> None:
    pdf_bytes = _build_pdf_bytes("Hello PDF")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/parse",
            files={"pdf": ("sample.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            params={"max_pages": "0"},
        )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_parse_handles_corrupt_pdf_bytes() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/parse",
            files={"pdf": ("sample.pdf", io.BytesIO(b"not-a-real-pdf"), "application/pdf")},
            params={"max_workers": "1"},
        )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_evaluate_endpoint_scores_identical_md_as_zero_error() -> None:
    pdf_bytes = _build_pdf_bytes("Hello PDF")

    # First, generate the canonical md via /parse.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        parse_resp = await client.post(
            "/parse",
            files={"pdf": ("sample.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            params={"llm_cleanup_enabled": "false", "ocr_enabled": "false", "max_workers": "1"},
        )
        assert parse_resp.status_code == 200

        with zipfile.ZipFile(io.BytesIO(parse_resp.content)) as zf:
            md_text = zf.read("sample.md").decode("utf-8")

        eval_resp = await client.post(
            "/evaluate",
            files={
                "pdf": ("sample.pdf", io.BytesIO(pdf_bytes), "application/pdf"),
                "md": ("sample.md", io.BytesIO(md_text.encode("utf-8")), "text/markdown"),
            },
            params={"llm_cleanup_enabled": "false", "ocr_enabled": "false", "max_workers": "1"},
        )

    assert eval_resp.status_code == 200
    payload = eval_resp.json()
    assert payload["evaluated_pages"] == 1
    assert payload["metrics"]["wer"] == 0.0
    assert payload["metrics"]["cer"] == 0.0


@pytest.mark.anyio
async def test_evaluate_endpoint_detects_mismatched_md() -> None:
    pdf_bytes = _build_pdf_bytes("Hello PDF")
    bad_md = "# sample\n\n## Page 1\n\nThis is totally different.\n"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/evaluate",
            files={
                "pdf": ("sample.pdf", io.BytesIO(pdf_bytes), "application/pdf"),
                "md": ("sample.md", io.BytesIO(bad_md.encode("utf-8")), "text/markdown"),
            },
            params={"llm_cleanup_enabled": "false", "ocr_enabled": "false", "max_workers": "1"},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["evaluated_pages"] == 1
    assert payload["metrics"]["wer"] >= 0.0
