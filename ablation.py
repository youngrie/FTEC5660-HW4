#!/usr/bin/env python3
"""Task 5: the ablation study behind the "Homework 4 solution" section of README.md.

Each row of ablation.csv is one real run over public_test with one component switched on relative to
the row above it. The scores come from the grader's own functions (table_score, document_problems,
parse_score, retrieval_hit, answer_correct), so a row can be compared with results.csv directly.
Nothing here is estimated by hand: the Task 3 and Task 4 numbers are measured by calling the fixed
generator exactly as the runner does.

    python ablation.py --data-folder public_test              # full table -> ablation.csv
    python ablation.py --data-folder public_test --no-generate # skip Task 4 (no API key needed)
    python ablation.py --data-folder public_test --only B0 V7  # a few rows
    python ablation.py --data-folder public_test --diagnostics # the two Task 4 probes

Two things to know when reading the table:

  * B0's splitter is RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS, chunk_overlap=200).
    The handout writes "RecursiveCharacterTextSplitter()", whose 4000-character default produces
    chunks the runner drops for being longer than the 1500-character cap, which would leave B0 with
    no chunks at all and make every later row look better than it is for the wrong reason.
  * Task 2.2 is the score of the parse_query the pipeline actually uses. B0 to V3 do not parse the
    question (a plain BM25 search over the whole corpus), so they are credited 0 there even though
    hw4.parse_query exists; V4 onwards use it.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields as dataclass_fields, replace
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import hw4

NAIVE_CHUNK_CHARS = hw4.MAX_CHUNK_CHARS   # the cap the runner enforces, see the note above
NAIVE_OVERLAP = 200
WORKERS = 8


# ----------------------------------------------------------------------------- the naive components

def naive_table_text(table_html: str) -> str:
    """B0's table: whatever BeautifulSoup's get_text() makes of it, one long line."""
    return BeautifulSoup(table_html, "lxml").get_text(" ", strip=True)


def naive_tokenize(text: str) -> list[str]:
    """B0's tokenizer: lower case and split on whitespace."""
    return str(text).lower().split()


def naive_chunks(file_name: str, html: str, render_tables: bool) -> list[Document]:
    """B0's chunking: document text from get_text(), RecursiveCharacterTextSplitter, flat metadata.

    `render_tables` is the single difference V1 introduces: the table is replaced by the text
    table_to_text produces, so its values keep their row labels and column headers.
    """
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup
    company, doc_type, fiscal_year = hw4._document_metadata(soup)  # same extraction as Task 1.2
    for tag in body.find_all(["script", "style", "noscript"]):
        tag.decompose()
    if render_tables:
        for table in body.find_all("table"):
            table.replace_with(hw4.table_to_text(str(table)))
    text = body.get_text("\n", strip=True)
    metadata = {"source": file_name, "company": company, "fiscal_year": fiscal_year,
                "doc_type": doc_type}
    splitter = RecursiveCharacterTextSplitter(chunk_size=NAIVE_CHUNK_CHARS,
                                              chunk_overlap=NAIVE_OVERLAP)
    return [Document(page_content=piece, metadata=dict(metadata))
            for piece in splitter.split_text(text) if piece.strip()]


# ----------------------------------------------------------------------------- the configurations

# Every retrieval component off: a plain BM25 top-5 over the whole corpus, which is B0 and the three
# rows after it. Spelled out field by field rather than derived from hw4.FULL, so that adding a
# switch to the pipeline cannot silently switch it on in the baseline.
NAIVE = hw4._Config(query_parse=False, year_window=False, per_company=False,
                    evidence_bonus=False, abstain=False)
assert NAIVE == hw4._Config(False, False, False, False, False)


@dataclass(frozen=True)
class Pipeline:
    """One configuration: Task 1 components plus the retrieval switches of hw4._Config."""
    table_text: bool = False             # tables rendered with table_to_text
    structured: bool = False             # Task 1.2 chunking instead of the recursive splitter
    smart_tokenizer: bool = False        # hw4._tokenize instead of lower case + whitespace
    retrieval: hw4._Config = NAIVE

    def chunks(self, file_name: str, html: str) -> list[Document]:
        if self.structured:
            return hw4.chunk_document(file_name, html)
        return naive_chunks(file_name, html, self.table_text)

    def tokenize(self, text: str) -> list[str]:
        return hw4._tokenize(text) if self.smart_tokenizer else naive_tokenize(text)


def variants() -> list[tuple[str, str, Pipeline]]:
    """B0 and V1..V8, each one component away from the row above; V8 is the submitted pipeline."""
    rows: list[tuple[str, str, Pipeline]] = []
    pipe = Pipeline()
    rows.append(("B0", "naive RAG (Tutorial 7): get_text + recursive splitter + BM25 top-5 "
                        "over the whole corpus, no parsing, never returns []", pipe))

    pipe = replace(pipe, table_text=True)
    rows.append(("V1", "+ table_to_text: tables enter the index with their row labels and "
                        "column headers instead of a run of loose numbers", pipe))

    pipe = replace(pipe, structured=True)
    rows.append(("V2", "+ structured chunks: one section per chunk, tables kept whole with the "
                        "caption and header repeated, a context line on top", pipe))

    pipe = replace(pipe, smart_tokenizer=True)
    rows.append(("V3", "+ tokenizer: lower case, punctuation split, stop words dropped, "
                        "numbers keep their decimal point", pipe))

    # from here on the row number and the switch that changes must be stated explicitly: the year
    # window is still off at V4, otherwise V5 would be a no-op and V4 would be credited with it
    pipe = replace(pipe, retrieval=replace(pipe.retrieval, query_parse=True, year_window=False,
                                           per_company=False, evidence_bonus=False, abstain=False))
    rows.append(("V4", "+ parse_query: search only the company the question names, and only "
                        "the document type it asks for (the year must match exactly)", pipe))

    pipe = replace(pipe, retrieval=replace(pipe.retrieval, year_window=True))
    rows.append(("V5", "+ year window {Y, Y+1} for annual reports and {Y-1, Y} for calls and "
                        "reports, instead of the exact year", pipe))

    pipe = replace(pipe, retrieval=replace(pipe.retrieval, per_company=True))
    rows.append(("V6", "+ per-company retrieval: a comparison alternates between the companies "
                        "it names instead of returning five chunks of one", pipe))

    pipe = replace(pipe, retrieval=replace(pipe.retrieval, abstain=True))
    rows.append(("V7", "+ abstention: [] when the corpus has no such company, no such year, or "
                        "never mentions the metric", pipe))

    pipe = replace(pipe, retrieval=replace(pipe.retrieval, evidence_bonus=True))
    rows.append(("V8", "+ tie-breakers on top of BM25: the question's year in the text, and how "
                        "much of the question the chunk covers (= the submitted retrieve)", pipe))

    # Two invariants the table is worthless without, checked instead of assumed: every row changes
    # exactly one component relative to the row above, and the last row is the submitted retrieve().
    for (name_a, _, before), (name_b, _, after) in zip(rows, rows[1:]):
        changed = _changed(before, after)
        assert len(changed) == 1, f"{name_a} -> {name_b} changes {changed}, not one component"
    assert rows[-1][2].retrieval == hw4.FULL, "the last row must be the submitted retrieve()"
    return rows


def _changed(before: Pipeline, after: Pipeline) -> list[str]:
    """Which components differ between two configurations (a Pipeline field or a _Config field)."""
    changed = [f.name for f in dataclass_fields(Pipeline)
               if f.name != "retrieval" and getattr(before, f.name) != getattr(after, f.name)]
    changed += [f"retrieval.{f.name}" for f in dataclass_fields(hw4._Config)
                if getattr(before.retrieval, f.name) != getattr(after.retrieval, f.name)]
    return changed


# ----------------------------------------------------------------------------- one run

def run(pipeline: Pipeline, data: dict, generate_answers: bool) -> dict:
    """Score one configuration on public_test with the grader's functions."""
    started = time.perf_counter()
    corpus, questions = data["corpus"], data["questions"]
    ground_truth, docs_meta = data["ground_truth"], data["docs_meta"]
    tables = data["tables"]

    # Task 1.1: one table renderer per configuration
    render = hw4.table_to_text if pipeline.table_text else naive_table_text
    table_scores = []
    for table in tables:
        try:
            table_scores.append(hw4.table_score(render(table["html"]), table))
        except Exception:
            table_scores.append(0.0)

    # Task 1.2: chunks per document, then the same cap the runner applies
    chunks_by_file: dict[str, list[Document]] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(pipeline.chunks(file_name, html) or [])
        except Exception as e:  # a broken chunker must not stop the table
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    problems = {f: hw4.document_problems(chunks_by_file.get(f), errors.get(f), m, f)
                for f, m in docs_meta.items()}
    passed = sum(not p for p in problems.values()) / max(len(problems), 1)

    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= hw4.MAX_CHUNK_CHARS]
    valid_text = {c.page_content for c in all_chunks}

    index = hw4.build_index(all_chunks, tokenize=pipeline.tokenize)

    # Tasks 2.1, 2.2, 2.3, 3
    parsed: dict[str, dict] = {}
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        parsed[qid] = hw4.parse_query(question) if pipeline.retrieval.query_parse else {}
        try:
            result = list(hw4._retrieve_with(index, question, pipeline.retrieval) or [])
            valid[qid] = len(result) <= hw4.MAX_RETRIEVED and all(
                c.page_content in valid_text for c in result)
            retrieved[qid] = result[:hw4.MAX_RETRIEVED]
        except Exception:
            retrieved[qid], valid[qid] = [], False

    # Task 4: the fixed generator, as the runner calls it
    usage_before = hw4.GENERATOR_USAGE.summary()
    answers: dict[str, str] = {}
    if generate_answers:
        def answer(qid: str) -> str:
            if not retrieved[qid]:
                return "NOT_FOUND"
            try:
                return hw4.generate(questions[qid], retrieved[qid])
            except Exception as e:
                print(f"    generator failed on {qid}: {e!r}")
                return ""

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            answers = dict(zip(questions, pool.map(answer, questions)))
    usage_after = hw4.GENERATOR_USAGE.summary()

    # the runner's own definitions, so the numbers are comparable with results.csv
    graded = [qid for qid in questions if qid in ground_truth]
    answerable = [q for q in graded if ground_truth[q]["evidence"]]
    unanswerable = [q for q in graded if not ground_truth[q]["evidence"]]
    parse_mean = (sum(hw4.parse_score(parsed[q], ground_truth[q]) for q in graded) / len(graded)
                  if pipeline.retrieval.query_parse else 0.0)
    non_empty = sum(valid[q] and bool(retrieved[q]) for q in answerable) / max(len(answerable), 1)
    recall = sum(hw4.retrieval_hit(retrieved[q], ground_truth[q]) for q in answerable) / max(len(answerable), 1)
    kept = sum(bool(retrieved[q]) for q in answerable) / max(len(answerable), 1)
    rejected = sum(not retrieved[q] for q in unanswerable) / max(len(unanswerable), 1)
    accuracy = (sum(hw4.answer_correct(answers.get(q, ""), ground_truth[q]) for q in graded)
                / len(graded) if generate_answers else float("nan"))

    points = {
        "1.1": hw4.POINTS["1.1"] * sum(table_scores) / max(len(table_scores), 1),
        "1.2": hw4.POINTS["1.2"] * passed,
        "2.1": hw4.POINTS["2.1"] * non_empty,
        "2.2": hw4.POINTS["2.2"] * parse_mean,
        "2.3": hw4.POINTS["2.3"] * hw4.clip((recall - hw4.RECALL_ZERO)
                                            / (hw4.RECALL_FULL - hw4.RECALL_ZERO)),
        "3": hw4.POINTS["3"] * hw4.clip(kept + rejected - 1),
        "4": hw4.POINTS["4"] * accuracy if generate_answers else float("nan"),
    }
    return {
        "points": points,
        "raw": {"table_cells": sum(table_scores) / max(len(table_scores), 1), "docs_pass": passed,
                "non_empty": non_empty, "parse": parse_mean, "recall": recall,
                "kept": kept, "rejected": rejected, "accuracy": accuracy},
        "chunks": len(all_chunks),
        "seconds": time.perf_counter() - started,
        "gen_calls": usage_after["llm_calls"] - usage_before["llm_calls"],
        "gen_tokens": (usage_after["llm_input_tokens"] + usage_after["llm_output_tokens"]
                       - usage_before["llm_input_tokens"] - usage_before["llm_output_tokens"]),
    }


# ----------------------------------------------------------------------------- the probes

def _chunk_key(document: Document) -> tuple[str, str]:
    return (document.metadata.get("source", ""), document.page_content[:80])


def mechanism(data: dict) -> None:
    """Why a row can be flat: does the switch change anything at all, and where does it stop showing?

    A component can be working and still add no point, because the grader's metric is already at
    ceiling. These counts separate the two: how often the switch changes the chunks retrieve()
    returns, and how often it changes the answer.
    """
    corpus, questions = data["corpus"], data["questions"]
    ground_truth = data["ground_truth"]
    chunks = [c for f in sorted(corpus) for c in hw4.chunk_document(f, corpus[f])]
    index = hw4.build_index(chunks)

    def without(**overrides) -> hw4._Config:
        return replace(hw4.FULL, abstain=False, evidence_bonus=False, **overrides)

    pairs = [("V4 -> V5  year window", without(year_window=False, per_company=False),
              without(year_window=True, per_company=False)),
             ("V5 -> V6  per company", without(year_window=True, per_company=False),
              without(year_window=True, per_company=True)),
             ("V7 -> V8  tie-breakers", replace(hw4.FULL, evidence_bonus=False), hw4.FULL)]
    print("Does the switch change what retrieve() returns?")
    for label, before, after in pairs:
        a = {q: [_chunk_key(c) for c in hw4._retrieve_with(index, qq, before)]
             for q, qq in questions.items()}
        b = {q: [_chunk_key(c) for c in hw4._retrieve_with(index, qq, after)]
             for q, qq in questions.items()}
        same_set = sum(sorted(a[q]) == sorted(b[q]) for q in a)
        same_order = sum(a[q] == b[q] for q in a)
        print(f"  {label}: chunks differ on {60 - same_set}/60 questions, "
              f"order differs on {60 - same_order}/60")

    # the year window: how many questions does it keep inside the right document type?
    empty_strict = answerable = rescued = 0
    for qid, question in questions.items():
        parsed = hw4.parse_query(question)
        keys = [hw4.company_key(n) for n in parsed["companies"] if hw4.company_key(n) in index["by_company"]]
        if not keys:
            continue
        candidates = sorted({i for k in keys for i in index["by_company"][k]})
        strict = [i for i in candidates if hw4._in_scope(chunks[i].metadata, parsed, window=False)]
        if strict:
            continue
        empty_strict += 1
        if ground_truth[qid]["evidence"]:
            answerable += 1
            wide = [i for i in candidates if hw4._in_scope(chunks[i].metadata, parsed, window=True)]
            rescued += bool(wide)
    print(f"\nThe exact-year filter empties the candidate set on {empty_strict} questions "
          f"({answerable} of them answerable).")
    print(f"  the year window puts back in scope: {rescued} of those {answerable}")

    # the comparison guard: when would the top 5 be one company only?
    print("\nComparison questions whose five chunks all come from one company:")
    for label, config in (("no metadata filter (V3)", replace(NAIVE, per_company=False)),
                          ("metadata filter (V4)", without(year_window=False, per_company=False)),
                          ("metadata filter + guard (V6)", without(year_window=True, per_company=True))):
        leaks = [qid for qid, question in questions.items()
                 if ground_truth[qid]["category"] == "comparison"
                 and len({hw4.company_key(c.metadata.get("company"))
                          for c in hw4._retrieve_with(index, question, config)}) < 2]
        print(f"  {label:<30} {len(leaks):2}/12 {leaks}")


def task4_probes(data: dict) -> None:
    """Is Task 4 driven by the five chunks or by their order? Two runs of the fixed generator."""
    corpus, questions = data["corpus"], data["questions"]
    ground_truth = data["ground_truth"]
    chunks = [c for f in sorted(corpus) for c in hw4.chunk_document(f, corpus[f])]
    index = hw4.build_index(chunks)
    retrieved = {qid: list(hw4._retrieve_with(index, q, hw4.FULL)) for qid, q in questions.items()}
    graded = [q for q in questions if q in ground_truth]

    for label, transform in (("D1 only the top-ranked chunk", lambda picked: picked[:1]),
                             ("D2 the five chunks, shuffled", lambda picked: random.Random(7).sample(
                                 picked, len(picked)))):
        def answer(qid: str) -> str:
            picked = transform(retrieved[qid])
            if not picked:
                return "NOT_FOUND"
            try:
                return hw4.generate(questions[qid], picked)
            except Exception:
                return ""

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            answers = dict(zip(questions, pool.map(answer, questions)))
        correct = sum(hw4.answer_correct(answers[q], ground_truth[q]) for q in graded)
        print(f"  {label:<32} {correct}/{len(graded)} = {correct / len(graded):.1%}")


# ----------------------------------------------------------------------------- reporting

def markdown_table(rows: list[dict]) -> str:
    """The table for README.md, printed from the rows that were just written to ablation.csv."""
    columns = [("1.1", "8"), ("1.2", "7"), ("2.1", "5"), ("2.2", "10"), ("2.3", "15"),
               ("3", "10"), ("4", "20")]
    lines = ["| config | what changed | " + " | ".join(f"{k} /{v}" for k, v in columns)
             + " | total /75 | time | gen tokens |",
             "|---|---|" + "---|" * (len(columns) + 3)]
    for row in rows:
        cells = [f"{row[k]:.2f}" if row[k] == row[k] else "n/a" for k, _ in columns]
        total = sum(row[k] for k, _ in columns if row[k] == row[k])
        lines.append(f"| **{row['config']}** | {row['change']} | " + " | ".join(cells)
                     + f" | **{total:.2f}** | {row['seconds']:.0f}s | {row['generator_tokens']:,} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5 ablation study for FTEC5660 HW4")
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--output", default="ablation.csv")
    parser.add_argument("--only", nargs="*", help="run only these configurations, e.g. --only B0 V7")
    parser.add_argument("--no-generate", action="store_true", help="skip Task 4 (no API calls)")
    parser.add_argument("--diagnostics", action="store_true", help="also run the two Task 4 probes")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    data = {
        "corpus": {p.name: p.read_text(encoding="utf-8")
                   for p in sorted((folder / "corpus").glob("*.html"))},
        "questions": hw4.load_json(folder / "questions.json"),
        "ground_truth": hw4.load_json(folder / "ground_truth.json"),
        "docs_meta": hw4.load_json(folder / "docs.json"),
        "tables": hw4.load_json(folder / "tables.json"),
    }
    print(f"{len(data['corpus'])} documents, {len(data['questions'])} questions, "
          f"{len(data['tables'])} tables\n")

    rows = []
    for name, change, pipeline in variants():
        if args.only and name not in args.only:
            continue
        print(f"{name} ...", flush=True)
        result = run(pipeline, data, generate_answers=not args.no_generate)
        rows.append({"config": name, "change": change,
                     **{k: round(v, 4) for k, v in result["points"].items()},
                     **{k: round(v, 4) for k, v in result["raw"].items()},
                     "chunks": result["chunks"], "seconds": round(result["seconds"], 1),
                     "generator_calls": result["gen_calls"],
                     "generator_tokens": result["gen_tokens"]})
        total = sum(v for v in result["points"].values() if v == v)
        print(f"  total {total:5.2f} / 75   chunks {result['chunks']:4}   "
              f"{result['seconds']:4.1f}s   generator tokens {result['gen_tokens']}")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output} ({len(rows)} configurations)")
    print("\n" + markdown_table(rows))

    if args.diagnostics:
        print("\n--- mechanism: does each switch change what retrieve() returns? ---")
        mechanism(data)
        print("\n--- Task 4 probes on the submitted pipeline ---")
        task4_probes(data)


if __name__ == "__main__":
    main()
