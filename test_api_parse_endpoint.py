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
