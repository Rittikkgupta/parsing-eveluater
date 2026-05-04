from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

import fitz
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI


load_dotenv()


class PageState(TypedDict, total=False):
    page_number: int
    page: fitz.Page
    pdf_name: str
    parser_min_chars: int
    parser_min_words: int
    enable_llm_fallback: bool
    enable_ocr_fallback: bool
    raw_text: str
    text: str
    source: str
    quality: dict[str, Any]
    links: list[dict[str, Any]]
    issues: list[str]


class DocumentState(TypedDict, total=False):
    pdf_path: str
    pdf_name: str
    doc: fitz.Document
    page_count: int
    parsed_pages: list[dict[str, Any]]
    parsed_output_path: str
    ground_truth_dir: str
    evaluation_path: str
    metrics: dict[str, Any]
    issues: list[str]


@dataclass(slots=True)
class PDFPipelineConfig:
    parsed_output_dir: Path = Path("parsed_output")
    ground_truth_dir: Path = Path("ground_truth")
    evaluation_dir: Path = Path("evaluation")
    cache_dir: Path = Path(".cache/pdf_pipeline")
    parser_min_chars: int = 80
    parser_min_words: int = 20
    llm_cleanup_enabled: bool = True
    ocr_enabled: bool = True
    generate_ground_truth: bool = True
    ground_truth_pages: int = 5
    max_workers: int = 4
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_GT_MODEL", "qwen3.5")
    )
    ollama_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))
    )
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_GT_MODEL", "llama-3.1-8b-instant")
    )
    groq_base_url: str = field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )

    def ensure_directories(self) -> None:
        for path in (
            self.parsed_output_dir,
            self.ground_truth_dir,
            self.evaluation_dir,
            self.cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class OllamaClient:
    host: str
    model: str
    timeout_seconds: int
    cache_dir: Path
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _request(self, prompt: str, cache_key: str) -> str:
        cache_path = self.cache_dir / f"{cache_key}.txt"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8").strip()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = str(body.get("response") or "").strip()
        if text:
            with self._lock:
                cache_path.write_text(text + "\n", encoding="utf-8")
        return text

    def cleanup_page_text(self, pdf_name: str, page_number: int, page_text: str) -> str:
        prompt = (
            "Clean the extracted PDF text without summarizing.\n"
            "Preserve headings, numbering, links, and legal wording.\n"
            "Fix only obvious line-wrap and hyphenation artifacts.\n"
            "Return plain text only.\n\n"
            f"PDF: {pdf_name}\n"
            f"Page: {page_number}\n\n"
            "Content:\n"
            f"{page_text}"
        )
        cache_key = f"cleanup_{slugify(pdf_name)}_{page_number}"
        return self._request(prompt, cache_key)

    def generate_ground_truth(self, pdf_name: str, page_number: int, page_text: str) -> str:
        prompt = (
            "Extract the exact text from the given PDF page.\n\n"
            "Instructions:\n"
            "- Preserve headings and numbering\n"
            "- Maintain proper sentence structure\n"
            "- Fix broken words\n"
            "- Preserve links if present\n"
            "- Do NOT summarize\n"
            "- Output only clean text\n\n"
            "Content:\n"
            f"{page_text}"
        )
        cache_key = f"gt_{slugify(pdf_name)}_{page_number}"
        return self._request(prompt, cache_key)


@dataclass(slots=True)
class GroqClient:
    api_key: str
    model: str
    cache_dir: Path
    base_url: str = "https://api.groq.com/openai/v1"
    _client: OpenAI | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _complete(self, prompt: str, cache_key: str) -> str:
        cache_path = self.cache_dir / f"{cache_key}.txt"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8").strip()

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract and clean text faithfully. "
                        "Do not summarize. Return plain text only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            with self._lock:
                cache_path.write_text(text + "\n", encoding="utf-8")
        return text

    def cleanup_page_text(self, pdf_name: str, page_number: int, page_text: str) -> str:
        prompt = (
            "Clean the extracted PDF text without summarizing.\n"
            "Preserve headings, numbering, links, and legal wording.\n"
            "Fix only obvious line-wrap and hyphenation artifacts.\n"
            "Return plain text only.\n\n"
            f"PDF: {pdf_name}\n"
            f"Page: {page_number}\n\n"
            "Content:\n"
            f"{page_text}"
        )
        cache_key = f"groq_cleanup_{slugify(pdf_name)}_{page_number}"
        return self._complete(prompt, cache_key)

    def generate_ground_truth(self, pdf_name: str, page_number: int, page_text: str) -> str:
        prompt = (
            "Extract the exact text from the given PDF page.\n\n"
            "Instructions:\n"
            "- Preserve headings and numbering\n"
            "- Maintain proper sentence structure\n"
            "- Fix broken words\n"
            "- Preserve links if present\n"
            "- Do NOT summarize\n"
            "- Output only clean text\n\n"
            "Content:\n"
            f"{page_text}"
        )
        cache_key = f"groq_gt_{slugify(pdf_name)}_{page_number}"
        return self._complete(prompt, cache_key)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "document"


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def should_merge_lines(current_line: str, next_line: str) -> bool:
    if not current_line or not next_line:
        return False
    if current_line.endswith("-"):
        return True
    if re.search(r"[.!?:;)\]]$", current_line):
        return False
    if re.match(r"^([A-Z][a-z]+|\d+(\.\d+)*|Part|Subpart|\([a-z0-9]+\))\b", next_line):
        return False
    return True


def merge_block_lines(block_lines: list[str]) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for line in block_lines:
        line = normalize_whitespace(line)
        if not line:
            continue
        if not buffer:
            buffer = line
            continue
        if should_merge_lines(buffer, line):
            buffer = buffer[:-1] + line if buffer.endswith("-") else f"{buffer} {line}"
        else:
            merged.append(buffer)
            buffer = line
    if buffer:
        merged.append(buffer)
    return merged


def extract_link_text(page: fitz.Page, rect: fitz.Rect | None) -> str:
    if rect is None:
        return ""
    selected: list[str] = []
    for word in page.get_text("words"):
        x0, y0, x1, y1, text, *_ = word
        if rect.intersects(fitz.Rect(x0, y0, x1, y1)):
            selected.append(text)
    return normalize_whitespace(" ".join(selected))


def extract_links(page: fitz.Page) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for link in page.get_links():
        rect = link.get("from")
        uri = link.get("uri")
        target = link.get("file")
        links.append(
            {
                "text": extract_link_text(page, rect),
                "uri": uri,
                "target": target,
                "kind": link.get("kind"),
            }
        )
    return links


def extract_text_with_pymupdf(page: fitz.Page) -> str:
    text_dict = page.get_text("dict", sort=True)
    blocks: list[str] = []
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_lines: list[str] = []
        for line in block.get("lines", []):
            spans = [span.get("text", "") for span in line.get("spans", [])]
            line_text = normalize_whitespace("".join(spans))
            if line_text:
                block_lines.append(line_text)
        if block_lines:
            blocks.append("\n".join(merge_block_lines(block_lines)))
    return normalize_whitespace("\n\n".join(blocks))


def assess_text_quality(text: str, min_chars: int, min_words: int) -> dict[str, Any]:
    words = re.findall(r"\b\w+\b", text)
    broken_word_hits = len(re.findall(r"\b\w+-\s*\n?\w+\b", text))
    line_count = len([line for line in text.splitlines() if line.strip()])
    sentences = re.split(r"(?<=[.!?])\s+", text.strip()) if text.strip() else []
    has_sentence_end = bool(re.search(r"[.!?]$", text.strip()))
    is_low_quality = (
        len(text.strip()) < min_chars
        or len(words) < min_words
        or (len(words) > 0 and line_count > len(words) * 0.8)
        or (len(words) >= min_words and not has_sentence_end and len(sentences) <= 1)
    )
    return {
        "char_count": len(text.strip()),
        "word_count": len(words),
        "line_count": line_count,
        "broken_word_hits": broken_word_hits,
        "has_sentence_end": has_sentence_end,
        "is_low_quality": is_low_quality,
    }


def run_ocr(page: fitz.Page) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OCR dependencies are unavailable") from exc

    pix = page.get_pixmap()
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return normalize_whitespace(pytesseract.image_to_string(image))


def post_process_text(text: str, links: list[dict[str, Any]]) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"([a-z])\n([a-z])", r"\1 \2", text)
    text = re.sub(r"\n([0-9]+-[0-9]+)\n", r"\n\1\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", "\n", text)

    link_lines: list[str] = []
    for link in links:
        label = link.get("text") or "link"
        target = link.get("uri") or link.get("target")
        if target:
            link_lines.append(f"- [{label}]({target})")
    if link_lines:
        text = f"{text}\n\nLinks\n" + "\n".join(link_lines)
    return text.strip()


def validate_page(page_item: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    page_number = page_item["page"]
    text = page_item["text"]
    quality = page_item["quality"]

    if not text.strip():
        issues.append(f"Page {page_number} is empty")
    if quality["word_count"] == 0:
        issues.append(f"Page {page_number} has zero words")
    if quality["is_low_quality"]:
        issues.append(f"Page {page_number} remained low quality after fallbacks")
    if re.search(r"\b[a-z]{1,2}\s+[A-Z]{2,}\b", text):
        issues.append(f"Page {page_number} may contain broken sentences or casing")
    if page_item["links"]:
        missing_visible_text = [link for link in page_item["links"] if not link.get("text")]
        if missing_visible_text:
            issues.append(f"Page {page_number} has links with missing visible text")
    return issues


def load_ground_truth(ground_truth_dir: Path) -> dict[int, str]:
    if not ground_truth_dir.exists():
        return {}
    pages: dict[int, str] = {}
    for path in sorted(ground_truth_dir.iterdir()):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        match = re.search(r"(\d+)", path.stem)
        if not match:
            continue
        pages[int(match.group(1))] = path.read_text(encoding="utf-8")
    return pages


def levenshtein_distance(seq_a: list[str] | str, seq_b: list[str] | str) -> int:
    if seq_a == seq_b:
        return 0
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    prev_row = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, start=1):
        curr_row = [i]
        for j, b in enumerate(seq_b, start=1):
            insert_cost = curr_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (0 if a == b else 1)
            curr_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = curr_row
    return prev_row[-1]


def normalize_for_comparison(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_line(line: str) -> str:
    return " ".join(line.split()).casefold()


def compute_page_metrics(pred_text: str, gt_text: str) -> dict[str, float | int]:
    pred_text = normalize_for_comparison(pred_text)
    gt_text = normalize_for_comparison(gt_text)

    pred_words = pred_text.split()
    gt_words = gt_text.split()
    pred_lines = [normalize_line(line) for line in pred_text.splitlines() if line.strip()]
    gt_lines = [normalize_line(line) for line in gt_text.splitlines() if line.strip()]

    char_edits = levenshtein_distance(pred_text, gt_text)
    word_edits = levenshtein_distance(pred_words, gt_words)
    matched_lines = sum((Counter(pred_lines) & Counter(gt_lines)).values())

    precision = matched_lines / len(pred_lines) if pred_lines else 0.0
    recall = matched_lines / len(gt_lines) if gt_lines else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    coverage = len(set(pred_words) & set(gt_words)) / len(set(gt_words)) if gt_words else 0.0

    return {
        "cer": char_edits / len(gt_text) if gt_text else 0.0,
        "wer": word_edits / len(gt_words) if gt_words else 0.0,
        "line_precision": precision,
        "line_recall": recall,
        "line_f1": f1,
        "token_coverage": coverage,
        "char_edits": char_edits,
        "char_total": len(gt_text),
        "word_edits": word_edits,
        "word_total": len(gt_words),
        "matched_lines": matched_lines,
        "pred_line_total": len(pred_lines),
        "gt_line_total": len(gt_lines),
    }


class LangGraphPDFEvaluationPipeline:
    def __init__(
        self,
        config: PDFPipelineConfig | None = None,
        ollama_client: OllamaClient | None = None,
    ) -> None:
        self.config = config or PDFPipelineConfig()
        self.config.ensure_directories()
        self.ollama_client = ollama_client or OllamaClient(
            host=self.config.ollama_host,
            model=self.config.ollama_model,
            timeout_seconds=self.config.ollama_timeout_seconds,
            cache_dir=self.config.cache_dir,
        )
        self.groq_client = (
            GroqClient(
                api_key=self.config.groq_api_key,
                model=self.config.groq_model,
                cache_dir=self.config.cache_dir,
                base_url=self.config.groq_base_url,
            )
            if self.config.groq_api_key
            else None
        )
        self.page_graph = self._build_page_graph().compile()
        self.document_graph = self._build_document_graph().compile()

    def _build_page_graph(self) -> StateGraph:
        workflow = StateGraph(PageState)
        workflow.add_node("primary_parser", self._page_primary_parser)
        workflow.add_node("llm_fallback", self._page_llm_fallback)
        workflow.add_node("ocr_fallback", self._page_ocr_fallback)
        workflow.add_node("post_process", self._page_post_process)

        workflow.add_edge(START, "primary_parser")
        workflow.add_conditional_edges(
            "primary_parser",
            self._route_after_primary,
            {"llm_fallback": "llm_fallback", "post_process": "post_process"},
        )
        workflow.add_conditional_edges(
            "llm_fallback",
            self._route_after_llm,
            {"ocr_fallback": "ocr_fallback", "post_process": "post_process"},
        )
        workflow.add_edge("ocr_fallback", "post_process")
        workflow.add_edge("post_process", END)
        return workflow

    def _build_document_graph(self) -> StateGraph:
        workflow = StateGraph(DocumentState)
        workflow.add_node("load_pdf", self._load_pdf)
        workflow.add_node("parse_pages", self._parse_pages)
        workflow.add_node("save_output", self._save_output)
        workflow.add_node("generate_ground_truth", self._generate_ground_truth)
        workflow.add_node("evaluate", self._evaluate)
        workflow.add_node("save_metrics", self._save_metrics)

        workflow.add_edge(START, "load_pdf")
        workflow.add_edge("load_pdf", "parse_pages")
        workflow.add_edge("parse_pages", "save_output")
        workflow.add_edge("save_output", "generate_ground_truth")
        workflow.add_edge("generate_ground_truth", "evaluate")
        workflow.add_edge("evaluate", "save_metrics")
        workflow.add_edge("save_metrics", END)
        return workflow

    def _page_primary_parser(self, state: PageState) -> PageState:
        text = extract_text_with_pymupdf(state["page"])
        return {
            "raw_text": text,
            "text": text,
            "source": "pymupdf",
            "links": extract_links(state["page"]),
            "quality": assess_text_quality(
                text,
                min_chars=state["parser_min_chars"],
                min_words=state["parser_min_words"],
            ),
            "issues": [],
        }

    def _route_after_primary(self, state: PageState) -> Literal["llm_fallback", "post_process"]:
        if state["quality"]["is_low_quality"] and state["enable_llm_fallback"]:
            return "llm_fallback"
        return "post_process"

    def _page_llm_fallback(self, state: PageState) -> PageState:
        issues = list(state.get("issues", []))
        text = state["text"]
        source = "pymupdf_llm"

        # Preference order:
        # - If Groq is configured, use it first (fast, hosted).
        # - If Groq isn't available (no API key) or fails, fall back to local Ollama.
        if self.groq_client is not None:
            try:
                candidate = self.groq_client.cleanup_page_text(
                    state["pdf_name"], state["page_number"], text
                )
                if candidate:
                    text = candidate
                    source = "pymupdf_groq"
                    issues.append(f"Groq cleanup used on page {state['page_number']}")
            except Exception as exc:
                issues.append(f"Groq cleanup unavailable on page {state['page_number']}: {exc}")

        if source != "pymupdf_groq":
            try:
                candidate = self.ollama_client.cleanup_page_text(
                    state["pdf_name"], state["page_number"], text
                )
                if candidate:
                    text = candidate
                    source = "pymupdf_ollama"
                    issues.append(f"Ollama cleanup used on page {state['page_number']}")
            except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
                issues.append(f"Ollama cleanup unavailable on page {state['page_number']}: {exc}")

        return {
            "text": text,
            "source": source,
            "quality": assess_text_quality(
                text,
                min_chars=state["parser_min_chars"],
                min_words=state["parser_min_words"],
            ),
            "issues": issues,
        }

    def _route_after_llm(self, state: PageState) -> Literal["ocr_fallback", "post_process"]:
        if state["quality"]["is_low_quality"] and state["enable_ocr_fallback"]:
            return "ocr_fallback"
        return "post_process"

    def _page_ocr_fallback(self, state: PageState) -> PageState:
        issues = list(state.get("issues", []))
        text = state["text"]
        try:
            candidate = run_ocr(state["page"])
            if candidate:
                text = candidate
        except RuntimeError as exc:
            issues.append(f"OCR fallback unavailable on page {state['page_number']}: {exc}")

        return {
            "text": text,
            "source": "ocr" if text != state["text"] else state["source"],
            "quality": assess_text_quality(
                text,
                min_chars=state["parser_min_chars"],
                min_words=state["parser_min_words"],
            ),
            "issues": issues,
        }

    def _page_post_process(self, state: PageState) -> PageState:
        text = post_process_text(state["text"], state.get("links", []))
        quality = assess_text_quality(
            text,
            min_chars=state["parser_min_chars"],
            min_words=state["parser_min_words"],
        )
        return {"text": text, "quality": quality}

    def _load_pdf(self, state: DocumentState) -> DocumentState:
        pdf_path = Path(state["pdf_path"]).resolve()
        doc = fitz.open(pdf_path)
        return {
            "pdf_name": pdf_path.stem,
            "doc": doc,
            "page_count": len(doc),
            "issues": [],
        }

    def _invoke_page_graph(self, page: fitz.Page, pdf_name: str, page_number: int) -> dict[str, Any]:
        result = self.page_graph.invoke(
            {
                "page": page,
                "page_number": page_number,
                "pdf_name": pdf_name,
                "parser_min_chars": self.config.parser_min_chars,
                "parser_min_words": self.config.parser_min_words,
                "enable_llm_fallback": self.config.llm_cleanup_enabled,
                "enable_ocr_fallback": self.config.ocr_enabled,
            }
        )
        page_item = {
            "page": page_number,
            "text": result["text"],
            "source": result["source"],
            "quality": result["quality"],
            "links": result.get("links", []),
            "issues": result.get("issues", []),
        }
        page_item["issues"] = page_item["issues"] + validate_page(page_item)
        return page_item

    def _invoke_page_graph_for_pdf(
        self, pdf_path: str, pdf_name: str, page_number: int
    ) -> dict[str, Any]:
        with fitz.open(pdf_path) as doc:
            return self._invoke_page_graph(doc[page_number - 1], pdf_name, page_number)

    def _parse_pages(self, state: DocumentState) -> DocumentState:
        doc = state["doc"]
        page_numbers = list(range(1, state["page_count"] + 1))

        if self.config.max_workers > 1 and len(page_numbers) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                parsed_pages = list(
                    executor.map(
                        lambda n: self._invoke_page_graph_for_pdf(
                            state["pdf_path"], state["pdf_name"], n
                        ),
                        page_numbers,
                    )
                )
        else:
            parsed_pages = [
                self._invoke_page_graph(doc[page_number - 1], state["pdf_name"], page_number)
                for page_number in page_numbers
            ]

        issues: list[str] = []
        if len(parsed_pages) != state["page_count"]:
            issues.append("Missing pages detected during parsing")
        for item in parsed_pages:
            issues.extend(item["issues"])
        return {"parsed_pages": parsed_pages, "issues": issues}

    def _save_output(self, state: DocumentState) -> DocumentState:
        output_path = self.config.parsed_output_dir / f"{state['pdf_name']}.md"
        lines = [f"# {state['pdf_name']}", ""]
        for item in state["parsed_pages"]:
            lines.append(f"## Page {item['page']}")
            lines.append("")
            lines.append(
                f"_Source: {item['source']} | Words: {item['quality']['word_count']} | Low quality: {item['quality']['is_low_quality']}_"
            )
            lines.append("")
            if item["issues"]:
                lines.append("Validation")
                lines.extend(f"- {issue}" for issue in item["issues"])
                lines.append("")
            lines.append(item["text"])
            lines.append("")
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return {"parsed_output_path": str(output_path)}

    def _generate_ground_truth(self, state: DocumentState) -> DocumentState:
        gt_dir = self.config.ground_truth_dir / state["pdf_name"]
        gt_dir.mkdir(parents=True, exist_ok=True)
        issues = list(state.get("issues", []))

        if not self.config.generate_ground_truth:
            return {"ground_truth_dir": str(gt_dir)}

        for item in state["parsed_pages"][: self.config.ground_truth_pages]:
            gt_path = gt_dir / f"page_{item['page']}.txt"
            if gt_path.exists():
                continue
            gt_text = ""
            if self.groq_client is not None:
                try:
                    gt_text = self.groq_client.generate_ground_truth(
                        state["pdf_name"], item["page"], item["text"]
                    )
                    if gt_text.strip():
                        issues.append(f"Groq generated ground truth for page {item['page']}")
                except Exception as groq_exc:
                    issues.append(
                        f"Groq ground truth generation failed for page {item['page']}: {groq_exc}"
                    )

            if not gt_text.strip():
                try:
                    gt_text = self.ollama_client.generate_ground_truth(
                        state["pdf_name"], item["page"], item["text"]
                    )
                    if gt_text.strip():
                        issues.append(f"Ollama generated ground truth for page {item['page']}")
                except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
                    issues.append(
                        f"Ollama ground truth generation failed for page {item['page']}: {exc}"
                    )
            if gt_text.strip():
                gt_path.write_text(gt_text.strip() + "\n", encoding="utf-8")
        return {"ground_truth_dir": str(gt_dir), "issues": issues}

    def _evaluate(self, state: DocumentState) -> DocumentState:
        ground_truth_by_page = load_ground_truth(Path(state["ground_truth_dir"]))
        parsed_by_page = {item["page"]: item["text"] for item in state["parsed_pages"]}
        evaluated_pages = sorted(set(parsed_by_page) & set(ground_truth_by_page))

        if not evaluated_pages:
            metrics = {
                "pdf_name": state["pdf_name"],
                "evaluated_pages": 0,
                "metrics": {
                    "wer": None,
                    "cer": None,
                    "line_precision": None,
                    "line_recall": None,
                    "line_f1": None,
                    "token_coverage": None,
                },
                "page_wise": [],
                "issues": state["issues"] + ["No ground truth pages available for evaluation"],
            }
            return {"metrics": metrics}

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
        page_wise: list[dict[str, Any]] = []

        for page_number in evaluated_pages:
            page_metrics = compute_page_metrics(
                parsed_by_page[page_number], ground_truth_by_page[page_number]
            )
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

        line_precision = (
            totals["matched_lines"] / totals["pred_line_total"]
            if totals["pred_line_total"]
            else 0.0
        )
        line_recall = (
            totals["matched_lines"] / totals["gt_line_total"] if totals["gt_line_total"] else 0.0
        )
        line_f1 = (
            2 * line_precision * line_recall / (line_precision + line_recall)
            if (line_precision + line_recall)
            else 0.0
        )
        metrics = {
            "pdf_name": state["pdf_name"],
            "evaluated_pages": len(evaluated_pages),
            "metrics": {
                "wer": round(totals["word_edits"] / totals["word_total"], 4)
                if totals["word_total"]
                else 0.0,
                "cer": round(totals["char_edits"] / totals["char_total"], 4)
                if totals["char_total"]
                else 0.0,
                "line_precision": round(line_precision, 4),
                "line_recall": round(line_recall, 4),
                "line_f1": round(line_f1, 4),
                "token_coverage": round(totals["token_coverage"] / len(evaluated_pages), 4),
            },
            "page_wise": page_wise,
            "issues": state["issues"],
        }
        return {"metrics": metrics}

    def _save_metrics(self, state: DocumentState) -> DocumentState:
        output_path = self.config.evaluation_dir / f"{state['pdf_name']}.json"
        output_path.write_text(json.dumps(state["metrics"], indent=2), encoding="utf-8")
        state["doc"].close()
        return {"evaluation_path": str(output_path)}

    def run(self, pdf_path: str | Path) -> dict[str, Any]:
        result = self.document_graph.invoke({"pdf_path": str(pdf_path)})
        return {
            "pdf_name": result["pdf_name"],
            "parsed_output_path": result["parsed_output_path"],
            "ground_truth_dir": result["ground_truth_dir"],
            "evaluation_path": result["evaluation_path"],
            "metrics": result["metrics"],
            "parsed_pages": result["parsed_pages"],
        }

    def run_batch(self, pdf_paths: list[str | Path]) -> list[dict[str, Any]]:
        return [self.run(pdf_path) for pdf_path in pdf_paths]


def build_default_config() -> PDFPipelineConfig:
    return PDFPipelineConfig(
        generate_ground_truth=os.getenv("GENERATE_GT_WITH_OLLAMA", "true").lower()
        in {"1", "true", "yes"},
        llm_cleanup_enabled=os.getenv("ENABLE_LLM_CLEANUP", "true").lower()
        in {"1", "true", "yes"},
        ocr_enabled=os.getenv("ENABLE_OCR_FALLBACK", "true").lower() in {"1", "true", "yes"},
    )


def run_pipeline(pdf_paths: list[str | Path], config: PDFPipelineConfig | None = None) -> list[dict[str, Any]]:
    pipeline = LangGraphPDFEvaluationPipeline(config=config or build_default_config())
    return pipeline.run_batch(pdf_paths)
