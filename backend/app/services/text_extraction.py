import io
import logging
import shutil
import statistics
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pymupdf
from PIL import Image, ImageOps

from app.config import Settings

log = logging.getLogger(__name__)

PageSource = Literal["pdf_text", "ocr"]


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float  # 0-100
    page: int

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)


@dataclass
class Line:
    page: int
    words: list[Word]
    source: PageSource
    cells: list[list[Word]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "  ".join(" ".join(w.text for w in c) for c in self.cells)

    @property
    def cell_texts(self) -> list[str]:
        return [" ".join(w.text for w in c) for c in self.cells]

    @property
    def mean_conf(self) -> float:
        return statistics.fmean(w.conf for w in self.words) if self.words else 0.0


@dataclass
class PageText:
    number: int  # 1-based
    source: PageSource
    lines: list[Line]


@dataclass
class DocumentText:
    pages: list[PageText]
    warnings: list[str] = field(default_factory=list)

    @property
    def lines(self) -> list[Line]:
        return [ln for p in self.pages for ln in p.lines]

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    @property
    def ocr_pages(self) -> list[int]:
        return [p.number for p in self.pages if p.source == "ocr"]


class OCRUnavailableError(RuntimeError):
    pass


def tesseract_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# Line grouping


def group_lines(words: list[Word], page: int, source: PageSource) -> list[Line]:
    words = [w for w in words if w.text.strip()]
    if not words:
        return []
    words.sort(key=lambda w: (w.yc, w.x0))
    rows: list[list[Word]] = []
    for w in words:
        if rows:
            row = rows[-1]
            row_yc = statistics.fmean(x.yc for x in row)
            row_h = statistics.median(x.height for x in row)
            if abs(w.yc - row_yc) <= 0.5 * max(row_h, w.height):
                row.append(w)
                continue
        rows.append([w])

    lines = []
    for row in rows:
        row.sort(key=lambda w: w.x0)
        lines.append(Line(page=page, words=row, source=source, cells=split_cells(row)))
    return lines


def split_cells(row: list[Word]) -> list[list[Word]]:
    h = statistics.median(w.height for w in row)
    gap_limit = 0.9 * h
    cells: list[list[Word]] = [[row[0]]]
    for prev, w in zip(row, row[1:]):
        if w.x0 - prev.x1 > gap_limit:
            cells.append([w])
        else:
            cells[-1].append(w)
    return cells


# PDF text


def _pdf_page_words(page: pymupdf.Page, number: int) -> list[Word]:
    out = []
    for x0, y0, x1, y1, text, *_ in page.get_text("words", sort=True):
        out.append(Word(text, x0, y0, x1, y1, 95.0, number))
    return out


# OCR


def remove_rules(gray: Image.Image, ratio: float = 0.35) -> Image.Image:
    arr = np.array(gray)
    dark = arr < 140
    rows = np.where(dark.mean(axis=1) > ratio)[0]
    cols = np.where(dark.mean(axis=0) > ratio)[0]
    arr[rows, :] = 255  # table rules
    arr[:, cols] = 255
    return Image.fromarray(arr)


def estimate_skew(gray: Image.Image, max_angle: float = 3.0, step: float = 0.1) -> float:
    small = gray.copy()
    small.thumbnail((900, 900))
    inv = ImageOps.invert(small)
    best, best_score = 0.0, -1.0
    n = int(max_angle / step)
    for i in range(-n, n + 1):
        angle = i * step
        rot = np.asarray(inv.rotate(angle, resample=Image.BILINEAR, fillcolor=0), dtype=np.float32)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best, best_score = angle, score
    return best


def deskew(gray: Image.Image) -> Image.Image:
    angle = estimate_skew(gray)
    if abs(angle) < 0.05:
        return gray
    return gray.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)


def preprocess(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    # Upscale small
    if gray.width < 1500:
        scale = 1500 / gray.width
        gray = gray.resize((int(gray.width * scale), int(gray.height * scale)), Image.LANCZOS)
    return remove_rules(deskew(gray))


def ocr_image_words(img: Image.Image, number: int, lang: str) -> list[Word]:
    if not tesseract_available():
        raise OCRUnavailableError("Tesseract OCR is not installed or not on PATH.")
    import pytesseract

    data = pytesseract.image_to_data(
        preprocess(img), lang=lang, config="--psm 6", output_type=pytesseract.Output.DICT
    )
    out = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append(Word(text, x, y, x + w, y + h, conf, number))
    return out


# Public API


def extract_pdf(data: bytes, settings: Settings) -> DocumentText:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    warnings: list[str] = []
    if doc.needs_pass:
        raise ValueError("PDF is password protected.")
    if doc.page_count == 0:
        raise ValueError("PDF has no pages.")
    if doc.page_count > settings.max_pages:
        raise ValueError(f"PDF has {doc.page_count} pages; limit is {settings.max_pages}.")

    pages: list[PageText] = []
    ocr_ok = tesseract_available()
    for idx, page in enumerate(doc, start=1):
        words = _pdf_page_words(page, idx)
        char_count = sum(len(w.text) for w in words)
        if char_count >= settings.min_text_chars:
            pages.append(PageText(idx, "pdf_text", group_lines(words, idx, "pdf_text")))
            continue
        if not ocr_ok:
            warnings.append(
                f"Page {idx} has little text and OCR is unavailable; page skipped."
            )
            pages.append(PageText(idx, "pdf_text", group_lines(words, idx, "pdf_text")))
            continue
        pix = page.get_pixmap(dpi=settings.ocr_dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_words = ocr_image_words(img, idx, settings.ocr_lang)
        pages.append(PageText(idx, "ocr", group_lines(ocr_words, idx, "ocr")))
    doc.close()
    return DocumentText(pages, warnings)


def extract_image(data: bytes, settings: Settings) -> DocumentText:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise ValueError(f"Could not open image: {exc}") from exc
    if img.width * img.height > 60_000_000:
        raise ValueError("Image dimensions are too large.")
    words = ocr_image_words(img, 1, settings.ocr_lang)
    return DocumentText([PageText(1, "ocr", group_lines(words, 1, "ocr"))])
