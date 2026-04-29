from __future__ import annotations

from pathlib import Path

import fitz

from src.pdf_parsing_pipeline import (
    LangGraphPDFEvaluationPipeline,
    PDFPipelineConfig,
)


class FakeOllamaClient:
    def cleanup_page_text(self, pdf_name: str, page_number: int, page_text: str) -> str:
        return "Cleaned fallback text for evaluation."

    def generate_ground_truth(self, pdf_name: str, page_number: int, page_text: str) -> str:
        return page_text


class FailingOllamaClient:
    def cleanup_page_text(self, pdf_name: str, page_number: int, page_text: str) -> str:
        raise RuntimeError("Ollama unavailable")

    def generate_ground_truth(self, pdf_name: str, page_number: int, page_text: str) -> str:
        raise RuntimeError("Ollama unavailable")


class FakeGroqClient:
    def cleanup_page_text(self, pdf_name: str, page_number: int, page_text: str) -> str:
        return "Groq cleaned text."

    def generate_ground_truth(self, pdf_name: str, page_number: int, page_text: str) -> str:
        return "Groq ground truth."


def build_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_pipeline_generates_outputs_and_zero_error_metrics(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(
        pdf_path,
        [
            "Section 1. This is a clean first page for parsing.",
            "Section 2. This is a clean second page for parsing.",
        ],
    )

    config = PDFPipelineConfig(
        parsed_output_dir=tmp_path / "parsed_output",
        ground_truth_dir=tmp_path / "ground_truth",
        evaluation_dir=tmp_path / "evaluation",
        cache_dir=tmp_path / "cache",
        parser_min_chars=10,
        parser_min_words=3,
        llm_cleanup_enabled=False,
        ocr_enabled=False,
        generate_ground_truth=True,
        max_workers=1,
    )
    pipeline = LangGraphPDFEvaluationPipeline(config=config, ollama_client=FakeOllamaClient())

    result = pipeline.run(pdf_path)

    assert Path(result["parsed_output_path"]).exists()
    assert Path(result["ground_truth_dir"]).exists()
    assert Path(result["evaluation_path"]).exists()
    assert result["metrics"]["evaluated_pages"] == 2
    assert result["metrics"]["metrics"]["wer"] == 0.0
    assert result["metrics"]["metrics"]["cer"] == 0.0


def test_pipeline_uses_llm_fallback_for_low_quality_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "short.pdf"
    build_pdf(pdf_path, ["Tiny"])

    config = PDFPipelineConfig(
        parsed_output_dir=tmp_path / "parsed_output",
        ground_truth_dir=tmp_path / "ground_truth",
        evaluation_dir=tmp_path / "evaluation",
        cache_dir=tmp_path / "cache",
        parser_min_chars=30,
        parser_min_words=4,
        llm_cleanup_enabled=True,
        ocr_enabled=False,
        generate_ground_truth=False,
        max_workers=1,
    )
    pipeline = LangGraphPDFEvaluationPipeline(config=config, ollama_client=FakeOllamaClient())

    result = pipeline.run(pdf_path)
    page = result["parsed_pages"][0]

    assert page["source"] == "pymupdf_llm"
    assert "Cleaned fallback text for evaluation." in page["text"]
    assert page["quality"]["is_low_quality"] is False


def test_pipeline_uses_groq_when_ollama_fails(tmp_path: Path) -> None:
    pdf_path = tmp_path / "groq.pdf"
    build_pdf(pdf_path, ["Tiny"])

    config = PDFPipelineConfig(
        parsed_output_dir=tmp_path / "parsed_output",
        ground_truth_dir=tmp_path / "ground_truth",
        evaluation_dir=tmp_path / "evaluation",
        cache_dir=tmp_path / "cache",
        parser_min_chars=30,
        parser_min_words=4,
        llm_cleanup_enabled=True,
        ocr_enabled=False,
        generate_ground_truth=True,
        max_workers=1,
        groq_api_key="test-key",
    )
    pipeline = LangGraphPDFEvaluationPipeline(config=config, ollama_client=FailingOllamaClient())
    pipeline.groq_client = FakeGroqClient()

    result = pipeline.run(pdf_path)
    page = result["parsed_pages"][0]
    gt_file = Path(result["ground_truth_dir"]) / "page_1.txt"

    assert page["source"] == "pymupdf_groq"
    assert "Groq cleaned text." in page["text"]
    assert gt_file.exists()
    assert gt_file.read_text(encoding="utf-8").strip() == "Groq ground truth."
