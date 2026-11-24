"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass, replace
from rank_bm25 import BM25Okapi

# --- Task 1 constants -------------------------------------------------------------------------
MAX_LINE = 400          # MAX_LINE_CHARS: a longer line is dropped by the scorer, so we split it
CHUNK_LIMIT = 1450      # below MAX_CHUNK_CHARS (1500) to leave room for the context header
PIECE_LIMIT = 1250      # a single text piece is split before it reaches the chunk limit
SECTION_FILL = 350      # start a new chunk at a section boundary once the current one holds this much
OVERLAP = 100           # characters repeated when a long prose block has to be split

HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
SKIP_TAGS = {"nav", "style", "script", "head", "title", "meta", "link", "noscript"}
BLOCK_TAGS = ("p", "ul", "ol", "table", "div", "section", "h1", "h2", "h3", "h4", "header",
              "footer", "main", "article", "li", "dl", "dt", "dd", "figure", "aside")


def _collapse(text: Any) -> str:
    """One line of text: whitespace collapsed, no leading or trailing blanks."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _soup(fragment: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(fragment, "lxml")
    except Exception:
        return BeautifulSoup(fragment, "html.parser")


# --- Task 1.1: tables -------------------------------------------------------------------------

def _rows_to_grid(rows: list[Tag]) -> list[list[str]]:
    """The rows of a table as lists of cell texts, with colspan/rowspan expanded."""
    grid: list[list[str]] = []
    carry: dict[int, list] = {}  # column -> [rows it still fills, text]
    for tr in rows:
        row: list[str] = []
        col = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while carry.get(col, [0])[0] > 0:  # a rowspan from a previous row covers this column
                row.append(carry[col][1])
                carry[col][0] -= 1
                col += 1
            text = _collapse(cell.get_text(" ", strip=True))
            colspan = max(1, int(cell.get("colspan") or 1))
            rowspan = max(1, int(cell.get("rowspan") or 1))
            for k in range(colspan):
                row.append(text)
                if rowspan > 1:
                    carry[col + k] = [rowspan - 1, text]
            col += colspan
        while carry.get(col, [0])[0] > 0:  # a rowspan that runs past the last cell of this row
            row.append(carry[col][1])
            carry[col][0] -= 1
            col += 1
        grid.append(row)
    return grid


def _column_headers(header_grid: list[list[str]], width: int) -> list[str]:
    """One header per column: every level of a multi-row header, top to bottom."""
    headers = []
    for col in range(width):
        parts: list[str] = []
        for row in header_grid:
            value = row[col] if col < len(row) else ""
            if value and value not in parts:
                parts.append(value)
        headers.append(" ".join(parts))
    return headers


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    soup = _soup(table_html)
    table = soup.find("table")
    if table is None:
        return _collapse(soup.get_text(" ", strip=True))

    caption = ""
    caption_tag = table.find("caption")
    if caption_tag is not None:
        caption = _collapse(caption_tag.get_text(" ", strip=True))

    # header rows: <thead>, or a leading all-<th> row when the table has no <thead>
    head_rows: list[Tag] = []
    thead = table.find("thead")
    if thead is not None:
        head_rows = thead.find_all("tr")

    tbody = table.find("tbody")
    candidates = tbody.find_all("tr") if tbody is not None else table.find_all("tr")
    body_rows = [tr for tr in candidates if tr.find_parent("thead") is None]

    if not head_rows and body_rows:
        first_cells = body_rows[0].find_all(["td", "th"], recursive=False)
        if first_cells and all(cell.name == "th" for cell in first_cells):
            head_rows, body_rows = [body_rows[0]], body_rows[1:]

    body = _rows_to_grid(body_rows)
    if not body:
        return caption
    width = max(len(row) for row in body)
    headers = _column_headers(_rows_to_grid(head_rows), width)
    headers += [""] * (width - len(headers))

    lines: list[str] = [caption] if caption else []
    for row in body:
        if not any(value for value in row):
            continue
        label = row[0] if row else ""
        cells = [(headers[i], row[i]) for i in range(1, width) if i < len(row) and row[i]]
        if not cells:
            continue
        line = " | ".join(f"{header}: {value}" if header else value for header, value in cells)
        line = f"{label} | {line}" if label else line
        if len(line) <= MAX_LINE:
            lines.append(line)
        else:
            # keep every cell on its own (short) line rather than losing the whole row
            lines.extend(f"{label} | {header}: {value}" if header else f"{label} | {value}"
                         for header, value in cells)
    return "\n".join(lines)


# --- Task 1.2: chunks -------------------------------------------------------------------------

def _document_type(title: str, body_text: str) -> str:
    """annual_report / earnings_call / research_report, read off the title first."""
    low = title.lower()
    if "results review" in low or "equity research" in low:
        return "research_report"
    if "earnings call" in low or "earnings conference call" in low:
        return "earnings_call"
    if "annual report" in low:
        return "annual_report"
    low_body = body_text.lower()
    if "results review" in low_body or "equity research" in low_body:
        return "research_report"
    if "earnings conference call" in low_body or "earnings call" in low_body:
        return "earnings_call"
    return "annual_report"


def _fiscal_year(title: str, body_text: str, doc_type: str) -> int:
    """The fiscal year the document is about (not the year of its publication)."""
    patterns = [r"annual report\s+((?:19|20)\d{2})", r"full year\s+((?:19|20)\d{2})",
                r"fy\s?((?:19|20)\d{2})", r"fiscal year ended\s+\d{1,2}\s+\w+\s+((?:19|20)\d{2})"]
    for text in (title, body_text):
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
    for pattern in patterns:
        match = re.search(pattern, body_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 2025 if doc_type in ("earnings_call", "research_report") else 0


def _document_company(soup: BeautifulSoup) -> str:
    """The company the document is about: the <h1> without its exchange/ticker bracket or legal suffix."""
    heading = soup.find("h1")
    name = _collapse(heading.get_text(" ", strip=True)) if heading is not None else ""
    name = re.sub(r"\s*\([^()]*\)\s*$", "", name).strip()
    stripped = re.sub(r"\s+(?:inc|corp|corporation|ltd|limited|plc|llc|co)\.?$", "", name, flags=re.IGNORECASE)
    return stripped or name


def _document_broker(soup: BeautifulSoup, title: str) -> str:
    """The broker that wrote a research report (never the company under review)."""
    line = ""
    broker_div = soup.find("div", class_="broker")
    if broker_div is not None:
        line = _collapse(broker_div.get_text(" ", strip=True))
    if not line and title:
        line = title
    return line.split("|")[0].strip()


def _section_label(section: str, subsection: str) -> str:
    if section and subsection:
        return f"{section} > {subsection}"
    return section or subsection


def _own_text(node: Tag) -> str:
    """The inline text of an element, ignoring the block children handled separately."""
    pieces = [str(child) for child in node.children
              if not isinstance(child, Tag) or child.name not in BLOCK_TAGS]
    return _collapse(" ".join(pieces))


def _iter_units(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """(section label, text) for every heading, paragraph, list item and table, in reading order."""
    body = soup.body or soup
    units: list[tuple[str, str]] = []
    state = {"section": "", "subsection": ""}

    def walk(node: Tag) -> None:
        for child in node.children:
            if not isinstance(child, Tag) or child.name in SKIP_TAGS:
                continue
            name = child.name
            if name in HEADING_TAGS:
                text = _collapse(child.get_text(" ", strip=True))
                if not text:
                    continue
                if HEADING_TAGS[name] <= 2:
                    state["section"], state["subsection"] = text, ""
                else:
                    state["subsection"] = text
                units.append((_section_label(state["section"], state["subsection"]), text))
            elif name == "table":
                text = table_to_text(str(child))
                if text:
                    units.append((_section_label(state["section"], state["subsection"]), text))
            elif child.find(BLOCK_TAGS) is not None:
                own = _own_text(child)
                if own:
                    units.append((_section_label(state["section"], state["subsection"]), own))
                walk(child)
            else:
                text = _collapse(child.get_text(" ", strip=True))
                if text:
                    units.append((_section_label(state["section"], state["subsection"]), text))

    walk(body)
    return units


def _split_long(text: str, limit: int, repeat_first: bool = False) -> list[str]:
    """Split a text that is longer than `limit` characters, keeping every character."""
    if len(text) <= limit:
        return [text]
    if "\n" in text:
        lines = text.split("\n")
        head = ""
        if repeat_first and len(lines) > 1:
            head, lines = lines[0], lines[1:]
        reserve = len(head) + 1 if head else 0
        parts: list[str] = []
        current: list[str] = []
        size = reserve
        for line in lines:
            while len(line) + reserve > limit:  # a single line longer than a whole piece
                cut = line.rfind(" ", 0, limit - reserve)
                cut = cut if cut > 0 else limit - reserve
                if current:
                    parts.append("\n".join([head, *current]) if head else "\n".join(current))
                    current, size = [], reserve
                parts.append("\n".join([head, line[:cut]]) if head else line[:cut])
                line = line[cut:].lstrip()
            if current and size + len(line) + 1 > limit:
                parts.append("\n".join([head, *current]) if head else "\n".join(current))
                current, size = [], reserve
            current.append(line)
            size += len(line) + 1
        if current:
            parts.append("\n".join([head, *current]) if head else "\n".join(current))
        return [part.strip() for part in parts if part.strip()]

    parts, start, length = [], 0, len(text)
    while start < length:
        end = min(start + limit, length)
        if end < length:
            cut = text.rfind(". ", start + limit // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + limit // 2, end)
            if cut > start:
                end = cut + 1
        parts.append(text[start:end].strip())
        if end >= length:
            break
        start = max(end - OVERLAP, start + 1)
    return [part for part in parts if part]


def _chunk_header(company: str, fiscal_year: int, doc_type: str, section: str) -> str:
    """A short context line: the generator's metadata header, repeated in the text for retrieval."""
    parts = [company, f"FY{fiscal_year}", doc_type.replace("_", " ")]
    if section:
        parts.append(section)
    return "[" + " | ".join(parts) + "]"


def _document_metadata(soup: BeautifulSoup) -> tuple[str, str, int]:
    """(company, doc_type, fiscal_year) of one document, and the corpus facts that follow from it.

    Reading one document also teaches the module which companies and which brokers the corpus has;
    parse_query and the abstention rules need that before the first question is asked.
    """
    body_text = _collapse((soup.body or soup).get_text(" ", strip=True))
    title = _collapse(soup.title.get_text(" ", strip=True)) if soup.title is not None else ""
    company = _document_company(soup)
    doc_type = _document_type(title, body_text)
    fiscal_year = _fiscal_year(title, body_text, doc_type)

    if company:
        _CORPUS_COMPANIES.setdefault(company_key(company), company)
    if doc_type == "research_report":
        broker = _document_broker(soup, title)
        if broker:
            _CORPUS_BROKERS.add(company_key(broker))
    return company, doc_type, fiscal_year


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    soup = _soup(html)
    company, doc_type, fiscal_year = _document_metadata(soup)

    units = _iter_units(soup)
    chunks: list[Document] = []
    current: list[str] = []
    current_section = ""
    current_size = 0

    def flush() -> None:
        nonlocal current, current_size
        if not current:
            return
        header = _chunk_header(company, fiscal_year, doc_type, current_section)
        content = "\n".join([header, *current]).strip()
        metadata = {"source": file_name, "company": company, "fiscal_year": fiscal_year,
                    "doc_type": doc_type, "section": current_section}
        if len(content) > MAX_CHUNK_CHARS:  # safety net, never expected
            for part in _split_long(content, CHUNK_LIMIT):
                chunks.append(Document(page_content=part, metadata=dict(metadata)))
        else:
            chunks.append(Document(page_content=content, metadata=metadata))
        current, current_size = [], 0

    for section, text in units:
        if not text:
            continue
        is_table = "\n" in text
        for piece in _split_long(text, PIECE_LIMIT, repeat_first=is_table):
            if current and section != current_section and current_size >= SECTION_FILL:
                flush()
            if not current:
                current_section = section
                current_size = len(_chunk_header(company, fiscal_year, doc_type, section)) + 1
            if current_size + len(piece) + 1 > CHUNK_LIMIT:
                flush()
                current_section = section
                current_size = len(_chunk_header(company, fiscal_year, doc_type, section)) + 1
            current.append(piece)
            current_size += len(piece) + 1
    flush()

    for number, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = number
    return chunks


# --- Task 2: query parsing and retrieval -------------------------------------------------------
# What chunk_document learned about the corpus: the companies it holds and the brokers that wrote
# its research reports. Both are used by parse_query and by the abstention rules.

_CORPUS_COMPANIES: dict[str, str] = {}
_CORPUS_BROKERS: set[str] = set()

_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
           "sep", "sept", "oct", "nov", "dec"}
_MONTH_RE = (r"(?:January|February|March|April|May|June|July|August|September|October|November"
             r"|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)")

# words that make a capitalised run a job title, a broker or a document section, never a company
_NOT_A_COMPANY = {"chief", "executive", "officer", "financial", "counsel", "chair", "chairman",
                  "director", "board", "committee", "audit", "auditor", "firm", "analyst", "president",
                  "management", "company", "companies", "group", "capital", "securities", "research",
                  "equity", "partners", "bank", "investment", "investments", "annual", "report",
                  "results", "review", "fiscal", "quarter", "usd", "fy"}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z&.\-]*|\d+")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.\-]*[a-z0-9]|[a-z0-9]")

_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "its", "it", "is",
         "are", "was", "were", "be", "been", "did", "do", "does", "what", "which", "who", "whom",
         "how", "much", "many", "that", "this", "these", "those", "with", "by", "as", "from", "than",
         "there", "their", "they", "has", "have", "had", "about", "into", "per", "million", "millions",
         "billion", "billions", "usd", "percent", "during", "after", "before", "between", "company",
         "companies", "year", "years", "report", "reports", "review", "one", "two", "end"}

# question word -> the words the documents use for the same thing
_EXPANSIONS = {
    "headquartered": ("headquarters", "hq", "based"),
    "headquarters": ("headquartered", "hq"),
    "city": ("headquarters", "headquartered"),
    "employees": ("headcount", "people", "staff", "workforce"),
    "headcount": ("employees", "people", "staff", "workforce"),
    "receivables": ("receivable",),
    "receivable": ("receivables",),
    "borrowings": ("borrowing", "debt"),
    "borrowing": ("borrowings", "debt"),
    "auditor": ("audit", "audited"),
    "audit": ("auditor", "audited"),
    "audited": ("audit", "auditor"),
    "cfo": ("chief", "financial", "officer"),
    "ceo": ("chief", "executive", "officer"),
    "repurchase": ("buyback",),
    "buyback": ("repurchase",),
    "guidance": ("guided", "guide", "outlook"),
    "guided": ("guidance", "guide", "outlook"),
    "guide": ("guidance", "guided", "outlook"),
    "rating": ("buy", "hold", "reduce"),
    "target": ("price",),
    "cash": ("equivalents",),
    "equivalents": ("cash",),
    "revenue": ("revenues",),
    "revenues": ("revenue",),
    "income": ("profit",),
    "profit": ("income",),
    "people": ("employees", "headcount", "staff", "workforce"),
    "staff": ("employees", "headcount", "people", "workforce"),
    "employ": ("employees", "headcount", "staff"),
    "employed": ("employees", "headcount", "staff"),
    "cost": ("costs", "expenses", "expense"),
    "costs": ("cost", "expenses", "expense"),
    "expense": ("expenses", "cost", "costs"),
    "expenses": ("expense", "cost", "costs"),
}

_RESEARCH_RE = re.compile(
    r"\b(?:target price|price target|rating|ratings|forecast\w*|estimat\w*|results review|valuation"
    r"|p/e|broker\w*|equity research|coverage|analyst\w*)\b", re.IGNORECASE)
_EARNINGS_RE = re.compile(
    r"\b(?:earnings call|conference call|fourth-quarter|fourth quarter|q4|guidance|guided|guid\w*"
    r"|share repurchase|buyback|repurchase programme|repurchase program)\b", re.IGNORECASE)


def _is_name_token(word: str) -> bool:
    """Is this word part of a company name (or another capitalised noun phrase)?"""
    if word.lower() in _MONTHS or word.lower() in _NOT_A_COMPANY:
        return False
    if re.fullmatch(r"(?:fy|q[1-4]|h[12])?\d{1,4}[a-z]?", word, re.IGNORECASE):
        return False
    if word.isupper():  # USD, FY, Q4, NYSE
        return False
    return word[:1].isupper()


def _name_runs(question: str) -> list[list[str]]:
    """Runs of capitalised words, e.g. ["Northgate", "Robotic", "Systems"]."""
    runs: list[list[str]] = []
    current: list[str] = []
    for word in _WORD_RE.findall(question):
        if _is_name_token(word):
            current.append(word)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _corpus_prefix(words: list[str]) -> str | None:
    """The longest company in the corpus whose name starts this run ("Lumen Harbor Foods' Bakery")."""
    best = ""
    for key in _CORPUS_COMPANIES:
        key_words = key.split()
        if len(key_words) <= len(words) and words[:len(key_words)] == key_words:
            if len(key_words) > len(best.split()):
                best = key
    return best or None


def _question_companies(question: str) -> list[str]:
    """Every company the question asks about, in the order the question names them."""
    found: list[str] = []
    seen: set[str] = set()
    for run in _name_runs(question):
        name = " ".join(run)
        key = company_key(name)
        if not key:
            continue
        matched = _corpus_prefix(key.split())
        if matched:  # a company the corpus has: use its own spelling
            if matched not in seen:
                seen.add(matched)
                found.append(_CORPUS_COMPANIES[matched])
            continue
        if key in seen:
            continue
        words = key.split()
        if len(words) < 2 or set(words) & _NOT_A_COMPANY:  # "Chief Executive Officer", "BuyFin Capital"
            continue
        if any(broker and broker in key for broker in _CORPUS_BROKERS):  # brokers are not companies
            continue
        # A company the corpus does not have. The questions do it on purpose ("Northgate Robotic
        # Systems" vs "Northgate Robotics"), so keep it as named, but only when it looks like a
        # near-miss of a company that is in the corpus (not "Asia Pacific", not "December").
        if not any(words[0] in other.split() for other in _CORPUS_COMPANIES):
            continue
        seen.add(key)
        found.append(name)
    return found


def _question_year(question: str) -> int | None:
    """The fiscal year the question names, not the year of the document that holds the answer."""
    matches = list(re.finditer(r"\bFY\s?((?:19|20)\d{2})\b", question, re.IGNORECASE))
    if matches:
        for match in matches:  # "in its FY2025 results review, what FY2026 revenue ..." -> 2026
            tail = question[match.end():match.end() + 30].lower()
            if not re.match(r"\s*(?:results review|annual report|report|review)", tail):
                return int(match.group(1))
        return int(matches[0].group(1))
    match = re.search(rf"\b\d{{1,2}}\s+{_MONTH_RE}\s+((?:19|20)\d{{2}})\b", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(?:in|for|at|during|of|since)\s+((?:19|20)\d{2})\b", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    years = re.findall(r"\b((?:19|20)\d{2})\b", question)
    return int(years[0]) if years else None


def _question_doc_type(question: str) -> str:
    if _RESEARCH_RE.search(question):
        return "research_report"
    if _EARNINGS_RE.search(question):
        return "earnings_call"
    return "annual_report"


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    return {"companies": _question_companies(question),
            "fiscal_year": _question_year(question),
            "doc_type": _question_doc_type(question)}


def _tokenize(text: str) -> list[str]:
    """Lower-case word tokens; a number keeps its decimal point (2,710.4 -> 710.4 and 2)."""
    lowered = str(text).lower().replace(",", " ")
    return [token for token in _TOKEN_RE.findall(lowered) if token not in _STOP]


def build_index(chunks: list[Document], tokenize: Any = None) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...

    `tokenize` only exists for the ablation study (ablation.py): it lets a variant index with the
    naive "lower case + whitespace" tokenizer of the Tutorial 7 baseline. The runner calls
    build_index(chunks), which gets the tokenizer below.
    """
    chunks = list(chunks)
    tokenize = tokenize or _tokenize
    bm25 = BM25Okapi([tokenize(chunk.page_content) for chunk in chunks]) if chunks else None
    by_company: dict[str, list[int]] = {}
    company_text: dict[str, list[str]] = {}
    years: dict[str, set[int]] = {}
    for position, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        key = company_key(meta.get("company"))
        by_company.setdefault(key, []).append(position)
        company_text.setdefault(key, []).append(_norm_text(chunk.page_content))
        if isinstance(meta.get("fiscal_year"), int):
            years.setdefault(key, set()).add(meta["fiscal_year"])
    return {"chunks": chunks, "bm25": bm25, "by_company": by_company,
            "company_text": {key: " ".join(texts) for key, texts in company_text.items()},
            "years": years, "companies": dict(_CORPUS_COMPANIES), "tokenize": tokenize}


def _year_ok(meta: dict, parsed: dict, window: bool = True) -> bool:
    """Can a chunk from this document hold a fact about the year the question names?

    The FY2025 annual report also shows the FY2024 figures, and FY2023 appears only in the FY2024
    report, so an annual report one year after the question's year is in scope. An earnings call or
    a research report reviews the year before the one it is about (FY2026 guidance in a FY2025 call).

    `window=False` is the stricter "only the year itself" rule, kept for the ablation study.
    """
    year = parsed.get("fiscal_year")
    fiscal_year = meta.get("fiscal_year")
    if not year or not isinstance(fiscal_year, int):
        return True
    if not window:
        return fiscal_year == year
    if meta.get("doc_type") == "annual_report":
        return fiscal_year in (year, year + 1)
    return fiscal_year in (year - 1, year)


def _in_scope(meta: dict, parsed: dict, window: bool = True) -> bool:
    if parsed.get("doc_type") and meta.get("doc_type") != parsed["doc_type"]:
        return False
    return _year_ok(meta, parsed, window)


def _query_terms(question: str, parsed: dict, tokenize: Any = None) -> list[str]:
    """The question's words, without the company names (already a filter) and with synonyms."""
    tokenize = tokenize or _tokenize
    drop: set[str] = set()
    for name in parsed.get("companies") or []:
        drop.update(tokenize(name))
    terms: list[str] = []
    for token in tokenize(question):
        if token in drop:
            continue
        terms.append(token)
        terms.extend(_EXPANSIONS.get(token, ()))
    return terms


# --- Task 3: abstention ------------------------------------------------------------------------
# The three ways the corpus can fail a question, checked in this order:
#   1. it names a company the corpus does not have ("Northgate Robotic Systems");
#   2. it names a year no document can cover ("total assets in FY2021");
#   3. it asks for a metric the documents never mention ("lobbying expenses").
# Each check only fires when the corpus is silent, so an answerable question never loses its chunks.

# Words that say how the question is asked, not what it asks for: "the value of X's inventories"
# asks about inventories, "what did X spend on lobbying" asks about lobbying.
_METRIC_FRAMING = {
    "value", "amount", "number", "figure", "level", "total", "overall", "combined", "exact",
    "spend", "spent", "spending", "pay", "paid", "earn", "earned", "record", "recorded", "post",
    "posted", "book", "booked", "report", "reported", "hold", "held", "carry", "carried",
    "generate", "generated", "announce", "announced", "set", "give", "gave", "make", "made",
    "come", "came", "take", "taken", "see", "show", "shown", "say", "said", "include", "including",
    "end", "ended", "begin", "beginning", "start", "close", "closed", "finish", "finished",
    "large", "largest", "small", "biggest", "high", "higher", "highest", "low", "lower", "lowest",
    "more", "most", "less", "least", "many", "same",
    # document and answer framing
    "results", "review", "report", "reports", "annual", "fiscal", "quarter", "quarterly",
    "fourth", "third", "second", "first", "call", "earnings", "conference", "guidance", "guided",
    "guide", "forecast", "forecasted", "estimate", "estimated", "outlook", "consensus",
    "q4", "fy", "company", "companies", "group", "business", "period", "average",
}

# Words that ask *where* or *who*, never *how much*: the documents answer them with "Headquarters:"
# or "Independent auditor", so they must not be read as a metric the corpus fails to mention.
_QUERY_WORDS = {"city", "town", "country", "headquartered", "headquarters", "hq", "located", "based",
                "auditor", "audit", "audited", "auditors", "firm", "cfo", "ceo", "officer"}

# Line items every annual report has. Two of these side by side say nothing about *which* line item
# is meant, so they are checked one by one and never as a phrase.
_GENERIC_HEADS = {"revenue", "revenues", "sales", "income", "profit", "loss", "assets", "asset",
                  "liabilities", "equity", "cash", "equivalents", "expenses", "expense", "cost",
                  "costs", "borrowing", "borrowings", "debt", "employees", "employee", "headcount",
                  "staff", "people", "personnel", "inventories", "inventory", "receivable",
                  "receivables", "payable", "payables", "goodwill", "dividends", "dividend", "margin",
                  "growth", "capital", "share", "shares", "earnings",
                  "employ", "employed", "employing", "hire", "hired", "workforce", "workers"}

_NUMERIC_QUESTION_RE = re.compile(
    r"\b(?:usd|us\$|million|millions|billion|billions|percent|percentage|bps|how\s+much"
    r"|how\s+many|what\s+(?:was|were|is|are)\s+the\s+(?:value|amount|number|level))\b",
    re.IGNORECASE)


def _norm_text(text: Any) -> str:
    """Lower case, hyphens opened up, punctuation gone: "Long-term borrowings" -> "long term borrowings"."""
    lowered = str(text).lower().replace("–", " ").replace("-", " ")
    lowered = re.sub(r"['’]s\b", " ", lowered)  # "Kestrel Semiconductor's" -> "kestrel semiconductor"
    return re.sub(r"[^a-z0-9 ]+", " ", lowered)


def _phrase_in(tokens: list[str], text: str) -> bool:
    """Do these words appear in this order in the (already normalised) text? Plurals are forgiven."""
    if not tokens:
        return True
    body = r"s?\s+".join(re.escape(token) for token in tokens)
    return re.search(rf"(?<![a-z0-9]){body}s?(?![a-z0-9])", text) is not None


def _metric_span(question: str, parsed: dict) -> list[str]:
    """The words of the question that name what is being asked for, in order.

    Everything that is not the metric is dropped: the company names (already a filter), the brokers
    that wrote the reports, the years, and the framing ("what was the value of ...").
    """
    drop = set(_STOP) | _METRIC_FRAMING | _QUERY_WORDS | _MONTHS
    for name in parsed.get("companies") or []:
        for token in re.findall(r"[a-z0-9]+", _norm_text(name)):
            drop.add(token)
    for broker in _CORPUS_BROKERS:
        for token in re.findall(r"[a-z0-9]+", _norm_text(broker)):
            drop.add(token)

    span: list[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9'’\-]*|\d+(?:\.\d+)?", question):
        for token in re.findall(r"[a-z0-9]+", _norm_text(word)):
            if token in drop or re.fullmatch(r"(?:(?:fy|q[1-4]|h[12])\d{0,4}|\d+(?:\.\d+)?)", token):
                continue
            span.append(token)
    return span


def _metric_pairs(question: str, parsed: dict) -> list[tuple[str, str]]:
    """The two-word windows of the question, as the question writes them.

    A pair the documents really use has to be adjacent in the question too, so the words between
    two metric words stay where they are: "change **in** working capital" keeps "working capital"
    together, while dropping the "in" would leave "change working", which no document ever says.
    Pairs that merely join two stop words ("customer in") are not evidence of anything and are
    dropped.
    """
    drop: set[str] = set()
    for name in parsed.get("companies") or []:
        drop.update(re.findall(r"[a-z0-9]+", _norm_text(name)))
    for broker in _CORPUS_BROKERS:
        drop.update(re.findall(r"[a-z0-9]+", _norm_text(broker)))

    def content(token: str) -> bool:
        return token not in _STOP and token not in _QUERY_WORDS

    pairs: list[tuple[str, str]] = []
    previous: str | None = None
    for word in re.findall(r"[A-Za-z][A-Za-z0-9'’\-]*|\d+(?:\.\d+)?", question):
        for token in re.findall(r"[a-z0-9]+", _norm_text(word)):
            if token in drop or re.fullmatch(r"(?:(?:fy|q[1-4]|h[12])\d{0,4}|\d+(?:\.\d+)?)", token):
                previous = None  # a name or a date: the words on either side are not adjacent
                continue
            if previous is not None and content(previous) and content(token):
                pairs.append((previous, token))
            previous = token
    return pairs


def _metric_disclosed(index: dict, keys: list[str], question: str, parsed: dict) -> bool:
    """Does the corpus ever mention what the question asks for, for the company it names?

    A numeric question whose metric is missing from the company's documents cannot be answered from
    this corpus: "lobbying expenses", "carbon offset purchases", "customer churn rate" and "revenue
    from its largest single customer" are asked about metrics no document reports. The metric is
    missing in one of two ways, and both are silent, cheap checks:

      * a word of it never appears ("lobbying", "carbon", "churn", "single"), so the documents never
        discuss it;
      * its words all appear, but never together: the report mentions "customer" and "largest"
        separately, yet has no "largest single customer" figure.

    Questions that ask for a name are left to the company and year checks: "which audit firm audited
    ..." is answered by the auditor's report even though the documents say "auditor", not "audit
    firm".
    """
    if not _NUMERIC_QUESTION_RE.search(question):
        return True
    span = _metric_span(question, parsed)
    if not span:
        return True

    texts = [index["company_text"][key] for key in keys if key in index["company_text"]]
    text = " ".join(texts) if texts else " ".join(index["company_text"].values())
    if not text:
        return True

    for word in span:  # the documents never use this word: the metric is not disclosed
        if len(word) > 2 and not any(_phrase_in([term], text)
                                     for term in (word, *_EXPANSIONS.get(word, ()))):
            return False
    if len(span) == 1:
        return True
    if all(word in _GENERIC_HEADS for word in span):
        return True  # "how many people did X employ": both words are everyday line-item words
    # every word is somewhere in the documents, but do they describe this metric? ("single customer")
    pairs = _metric_pairs(question, parsed)
    if not pairs:
        return True
    return any(_phrase_in([first, second], text) for first, second in pairs)


def _company_years(index: dict, keys: list[str]) -> set[int]:
    years: set[int] = set()
    for key in keys:
        years.update(index["years"].get(key, ()))
    return years


def _evidence_bonus(chunk: Document, parsed: dict, terms: list[str]) -> float:
    """A tie-breaker on top of BM25: right year in the text, and how much of the question it covers."""
    text = chunk.page_content.lower()
    bonus = 0.0
    year = parsed.get("fiscal_year")
    if year and str(year) in text:
        bonus += 1.0
    unique = {term for term in terms if len(term) > 2}
    if unique:
        bonus += 2.0 * sum(1 for term in unique if term in text) / len(unique)
    return bonus


@dataclass(frozen=True)
class _Config:
    """Which component of the pipeline a run switches on.

    retrieve() uses FULL, so what you read here is what is graded. The other combinations exist so
    that ablation.py can turn one component off at a time and measure it (Task 5) with the same
    code that is submitted, instead of a second implementation that might behave differently.
    """
    query_parse: bool = True    # parse_query: filter by company (and doc type, year) before ranking
    year_window: bool = True    # a document one year after the question's year still holds the figure
    per_company: bool = True    # a comparison gets chunks from each company, interleaved
    evidence_bonus: bool = True # tie-breakers on top of BM25: year in text, question coverage
    abstain: bool = True        # return [] when the corpus cannot answer (Task 3)


FULL = _Config()


def _retrieve_with(index: Any, question: str, cfg: _Config = FULL) -> list[Document]:
    """retrieve() with components switched on and off, for the ablation study."""
    chunks: list[Document] = index["chunks"]
    if not chunks:
        return []
    tokenize = index.get("tokenize") or _tokenize
    parsed = parse_query(question) if cfg.query_parse else {}
    keys = [company_key(name) for name in parsed.get("companies") or []]
    known = list(dict.fromkeys(key for key in keys if key in index["by_company"]))
    year = parsed.get("fiscal_year")

    # Task 3: say [] when the corpus cannot answer. Each gate below only fires when the corpus is
    # silent about the company, the year or the metric, so a question it can answer keeps its chunks.
    if cfg.abstain:
        if keys and not known:  # names a company the corpus does not have
            return []
        if known and isinstance(year, int):
            years = _company_years(index, known)
            if years and not any(abs(document_year - year) <= 1 for document_year in years):
                return []  # no document of that company can hold a figure for that year
        if not _metric_disclosed(index, known, question, parsed):
            return []  # asks for a metric the company's documents never mention

    if known:
        candidates = sorted({i for key in known for i in index["by_company"][key]})
    else:
        candidates = list(range(len(chunks)))

    in_scope = [i for i in candidates if _in_scope(chunks[i].metadata or {}, parsed, cfg.year_window)]
    if not in_scope:  # nothing of the parsed doc type: keep the company, ignore the type
        in_scope = [i for i in candidates if _year_ok(chunks[i].metadata or {}, parsed, cfg.year_window)]
    if not in_scope:
        in_scope = candidates

    terms = _query_terms(question, parsed, tokenize)
    bm25 = index.get("bm25")
    scores = bm25.get_scores(terms) if bm25 is not None and terms else None

    def rank(position: int) -> float:
        base = float(scores[position]) if scores is not None else 0.0
        if not cfg.evidence_bonus:
            return base
        return base + _evidence_bonus(chunks[position], parsed, terms)

    ranked = sorted(in_scope, key=rank, reverse=True)

    if cfg.per_company and len(known) >= 2:  # a comparison needs a chunk for each company, interleaved
        buckets = {key: [i for i in ranked
                         if company_key((chunks[i].metadata or {}).get("company")) == key]
                   for key in known}
        picked: list[int] = []
        while len(picked) < MAX_RETRIEVED:
            added = False
            for key in known:
                if buckets[key] and len(picked) < MAX_RETRIEVED:
                    picked.append(buckets[key].pop(0))
                    added = True
            if not added:
                break
        return [chunks[i] for i in picked]
    return [chunks[i] for i in ranked[:MAX_RETRIEVED]]


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.

    Company, document type and year from parse_query narrow the candidate set before BM25 ranks it;
    two tie-breakers put the chunk that states the figure first; a comparison alternates between the
    companies it names; and a question the corpus cannot answer gets [] instead of five wrong chunks.
    """
    return _retrieve_with(index, question, FULL)


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
