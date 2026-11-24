"""Local evaluation for FTEC5660 HW4 Tasks 2 and 3 (no generator call, no API key needed).

Run:  cd FTEC5660-HW4 && ./.venv/Scripts/python.exe ../dev_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent / "FTEC5660-HW4"
sys.path.insert(0, str(REPO))

import hw4  # noqa: E402

DATA = REPO / "public_test"
corpus = {p.name: p.read_text(encoding="utf-8") for p in sorted((DATA / "corpus").glob("*.html"))}
questions = json.load(open(DATA / "questions.json", encoding="utf-8"))
truth = json.load(open(DATA / "ground_truth.json", encoding="utf-8"))

chunks_by_file = {f: hw4.chunk_document(f, h) for f, h in corpus.items()}
all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
              if len(c.page_content) <= hw4.MAX_CHUNK_CHARS]
by_text = {}
for c in all_chunks:
    by_text.setdefault(c.page_content, []).append(c)
index = hw4.build_index(all_chunks)

parse_scores, hits, kept, rejected, valid = [], [], [], [], []
fail_detail = []
retrieved_map = {}
for qid, question in questions.items():
    gold = truth[qid]
    parsed = hw4.parse_query(question)
    ps = hw4.parse_score(parsed, gold)
    parse_scores.append(ps)
    if ps < 1:
        fail_detail.append((qid, ps, parsed, gold["query"]))
    result = hw4.retrieve(index, question)
    ok = isinstance(result, list) and len(result) <= hw4.MAX_RETRIEVED
    kept_chunks = []
    for c in result:
        matches = by_text.get(c.page_content, [])
        if not matches:
            ok = False
            continue
        kept_chunks.append(c)
    retrieved_map[qid] = kept_chunks
    valid.append(ok and bool(kept_chunks) if gold["evidence"] else ok)
    if gold["evidence"]:
        hits.append(hw4.retrieval_hit(kept_chunks, gold))
        kept.append(bool(kept_chunks))
    else:
        rejected.append(not kept_chunks)

n_answerable, n_unanswerable = len(hits), len(rejected)
recall = sum(hits) / n_answerable
kept_rate = sum(kept) / n_answerable
rejected_rate = sum(rejected) / n_unanswerable
parse_mean = sum(parse_scores) / len(parse_scores)
clip = hw4.clip
print(f"Task 2.1 valid+non-empty : {sum(valid) / len(valid):6.1%}   -> {5 * sum(valid) / len(valid):.2f}/5")
print(f"Task 2.2 parse fields    : {parse_mean:6.1%}   -> {10 * parse_mean:.2f}/10")
print(f"Task 2.3 recall@5        : {recall:6.1%}   -> "
      f"{15 * clip((recall - 0.5) / 0.4):.2f}/15")
print(f"Task 3   kept/rejected   : {kept_rate:6.1%} / {rejected_rate:.1%}   -> "
      f"{10 * clip(kept_rate + rejected_rate - 1):.2f}/10")
print(f"\nanswerable {n_answerable}, unanswerable {n_unanswerable}")

print("\n--- parse failures ---")
for qid, ps, parsed, expected in fail_detail:
    print(f"{qid} ({ps:.2f})  got {parsed}")
    print(f"          want {expected}")

print("\n--- recall misses ---")
for qid, question in questions.items():
    gold = truth[qid]
    if not gold["evidence"] or retrieved_map.get(qid) is None:
        continue
    if not hw4.retrieval_hit(retrieved_map[qid], gold):
        got = [(c.metadata["source"], c.metadata["doc_type"], c.metadata["fiscal_year"]) for c in retrieved_map[qid]]
        print(f"{qid} [{gold['category']}] {question[:90]}")
        print(f"     got   {got}")
        print(f"     want  {gold['evidence']}")

print("\n--- false negatives on unanswerable (kept chunks) ---")
for qid, question in questions.items():
    gold = truth[qid]
    if not gold["evidence"] and retrieved_map.get(qid):
        print(f"{qid} {question[:95]}")
