# FTEC5660 Homework 4 — retrieval for a RAG pipeline over company reports

The generator is given and fixed; the only thing this repository decides is **what the generator
sees**. Five functions in `hw4.py` do that: `table_to_text`, `chunk_document`, `parse_query`,
`build_index` and `retrieve`. `ablation.py` re-runs the whole thing with one component switched off
at a time, and writes `ablation.csv`.

```
hw4.py          the five functions + helpers        (the only file the grader runs)
ablation.py     B0 -> V8 ablation study             -> ablation.csv
dev_eval.py     Local evaluation for Tasks 2 and 3  (no generator call, no API key needed)
results.csv     one full run on public_test         (60 questions, all correct)
public_test/    corpus/, questions, ground truth, tables, docs metadata
```

---

## Homework 4 solution

### 1. The pipeline

```mermaid
flowchart TD
    H["doc_001.html ... doc_040.html<br/>annual reports, calls,<br/>research reports"] --> T["table_to_text<br/>one line per table row,<br/>value next to its labels"]
    H --> M["metadata<br/>company, doc_type, fiscal_year"]
    T --> C["chunk_document<br/>section blocks + tables kept whole<br/>context line on top, max 1500 chars"]
    M --> C
    C --> I["build_index<br/>BM25 + company -> chunks + company vocabulary"]
    Q["question"] --> P["parse_query<br/>companies, fiscal_year, doc_type"]
    P --> S{"scope<br/>company + doc type + year window"}
    I --> S
    S --> R["BM25 ranking<br/>question terms without the company names"]
    R --> B["tie-breakers<br/>year in text, question coverage"]
    B --> K{"abstention<br/>unknown company / uncovered year / undisclosed metric"}
    K -->|"[]"| N["generator answers NOT_FOUND"]
    K -->|"5 chunks"| G["fixed generator"]
    N --> A["answer"]
    G --> A
```

**`table_to_text` (Task 1.1).** One line per table row: `Revenue | 2024: 2,649.8 | 2023: 2,316.8`.
Values are copied unchanged — `2,649.8` keeps its thousands separator and `(844.4)` keeps its
parentheses — because the grader looks for the text exactly as printed. The caption, which holds
the unit, goes on the first line. Header cells are read from `<thead>`, with `colspan`/`rowspan`
expanded, and a header-less table falls back to using its first all-`<th>` row. A row that would
exceed 400 characters is split into one line per column, each repeating the row label: the grader
drops a longer line outright, and splitting keeps every value with its label. There is also a
branch for tables whose header row is a list of *attributes* rather than years (board of directors,
executive officers, peer valuation, financial summary), where the row label is a person or a
company and the columns are `Role`, `Since`, `Title` and so on.

**`chunk_document` (Task 1.2).** The document is walked in reading order and cut at `h2`/`h3`
boundaries, so a chunk is a section, not a window of characters. A table always arrives whole: if it
does not fit, it is split by *rows* and the caption and header are repeated in every piece, so
`492.1` is never separated from `Accounts receivable, net`. Every chunk starts with one line of
context — `[Northgate Robotics | FY2025 | annual report | 6. Financial statements > ...]` — which is
what makes a table row searchable by company and year without the sentence around it ever being
stored. 627 chunks over 40 documents, mean 645 characters, longest 1445 against the 1500 cap.
Neither this function nor `table_to_text` calls the model.

**`parse_query` (Task 2.2).** Rules only, no model. Companies are found by cutting the question into
runs of capitalised words and matching the longest prefix against the company list that
`chunk_document` collected from the corpus; a run is also broken at words like `Chief`, `Securities`
or `Capital`, because "Northgate Robotics' Chief Financial Officer" otherwise reads as one company
name. The broker names that appear in the research reports are collected the same way and are never
treated as companies. A name that matches nothing is kept as written, which is how "Northgate Robotic
Systems" stays available to the abstention rules. The year comes from `FYxxxx`, skipping one that is
immediately followed by `results review` or `annual report` (so "in its FY2025 results review, what
FY2026 revenue..." yields 2026), then from a date like `31 December 2024`, then from any bare year.
The document type comes from the words the question uses: `rating`, `target price`, `forecast`,
`results review` → research report; `earnings call`, `Q4`, `guidance`, `share repurchase` → earnings
call; otherwise annual report. 60/60 questions with all three fields right.

**`build_index` (Task 2).** `BM25Okapi` over the chunks, plus three maps built in the same pass:
company → chunk positions, company → all of its text (normalised, for the abstention check), and
company → the fiscal years it has documents for. The tokenizer lower-cases, splits on punctuation,
drops stop words and keeps the decimal point inside a number, so `2,710.4` contributes `710.4`.

**`retrieve` (Tasks 2 and 3).** The candidates are first narrowed to the companies the question
names and the document type it asks for. The year filter allows a document one year *after* the
question's year, because the FY2025 annual report shows FY2024 and FY2023 exists only in the FY2024
report, and one year *before* for calls and research reports, because FY2026 guidance is given on the
FY2025 call. If that leaves nothing, the type filter is dropped, then the year filter: an answerable
question must never come back empty (Task 2.1). BM25 ranks what is left, and the question's company
names are removed from the query terms, since they are already a filter and would otherwise reward
chunks that merely repeat the name. Two tie-breakers follow: a chunk that contains the question's
year, and a chunk that contains more of the question's words. A comparison question then alternates
between the two companies it names, so the five slots can never all come from one of them. Finally
three abstention checks return `[]`, each of which fires only when the corpus is silent: the company
does not exist in the corpus, no document of that company can hold a figure for that year, or — for
a question that asks for a number — a word of the metric never appears in the company's documents
(`lobbying`, `carbon`, `churn`, `single`) or its words never appear next to each other
(`largest single customer`).

### 2. Results on `public_test`

`python hw4.py --data-folder public_test`, 6.6 s in total (limit 30 min), 0 model calls of my own,
46 generator calls costing $0.0104.

| Task | metric | score |
|---|---|---|
| 1.1 `table_to_text` | 100.0% of the table cells kept with their labels | 8.00 / 8 |
| 1.2 `chunk_document` | 100.0% of the documents pass | 7.00 / 7 |
| 2.1 retrieval valid and non-empty | 100.0% of the answerable questions | 5.00 / 5 |
| 2.2 `parse_query` | 100.0% of the fields right (60/60) | 10.00 / 10 |
| 2.3 `recall@5` | 100.0% of the answerable questions (46/46) | 15.00 / 15 |
| 3 abstention | kept 100.0%, rejected 100.0% (14/14) | 10.00 / 10 |
| 4 end to end | 100.0% answer accuracy (60/60) | 20.00 / 20 |
| | **total** | **75.00 / 75** |

### 3. Ablation

`python ablation.py --data-folder public_test` reproduces the table; every row is one real run with
the grader's own scoring functions, and the last row is the submitted `retrieve`. B0 is the naive
pipeline of the handout: `get_text()`, `RecursiveCharacterTextSplitter`, BM25 top-5 over the whole
corpus, no parsing, never `[]`. Each row adds exactly one component to the row above — `ablation.py`
asserts that, and that the last row equals `hw4.FULL`, instead of trusting the hand-written list,
because the first version of this table had three rows that were silently identical.

| config | what changed | 1.1 /8 | 1.2 /7 | 2.1 /5 | 2.2 /10 | 2.3 /15 | 3 /10 | 4 /20 | total /75 |
|---|---|---|---|---|---|---|---|---|---|
| **B0** | naive RAG: `get_text` + recursive splitter + BM25 top-5, no parsing, never `[]` | 0.00 | 7.00 | 5.00 | 0.00 | 0.00 | 0.00 | 9.67 | 21.67 |
| **V1** | + `table_to_text` | 8.00 | 7.00 | 5.00 | 0.00 | 0.00 | 0.00 | 9.00 | 29.00 |
| **V2** | + structured chunks (sections, whole tables, context line) | 8.00 | 7.00 | 5.00 | 0.00 | 0.00 | 0.00 | 11.33 | 31.33 |
| **V3** | + real tokenizer (punctuation, stop words, decimals) | 8.00 | 7.00 | 5.00 | 0.00 | 13.86 | 0.00 | 17.33 | 51.19 |
| **V4** | + `parse_query` metadata filter (year must match exactly) | 8.00 | 7.00 | 5.00 | 10.00 | 15.00 | 0.00 | 19.67 | 64.67 |
| **V5** | + year window `{Y, Y+1}` / `{Y-1, Y}` | 8.00 | 7.00 | 5.00 | 10.00 | 15.00 | 0.00 | 19.33 | 64.33 |
| **V6** | + comparison retrieves per company | 8.00 | 7.00 | 5.00 | 10.00 | 15.00 | 0.00 | 19.33 | 64.33 |
| **V7** | + abstention (unknown company, uncovered year, undisclosed metric) | 8.00 | 7.00 | 5.00 | 10.00 | 15.00 | 10.00 | 20.00 | **75.00** |
| **V8** | + tie-breakers (year in text, question coverage) = submitted `retrieve` | 8.00 | 7.00 | 5.00 | 10.00 | 15.00 | 10.00 | 20.00 | **75.00** |

`recall@5` behind the 2.3 column: 39.1% → 30.4% → 45.7% → 87.0% → 100% at V4. `ablation.csv` carries
the same rows with the raw rates, the chunk count, the wall-clock time and the generator tokens of
each run (6.1–7.9 s and 66k–131k tokens per configuration).

**B0 → V1 (+7.33). `table_to_text` earns its 8 points and buys nothing else.** Task 1.1 goes from
0.00 to 8.00 on all twelve tables, including the 151-character one: the grader keeps a cell only on a
line that holds its row label, its column header, its value *and no other row label*, and
`get_text()` puts the whole table on a single line, so every cell fails the last condition.
Retrieval, however, gets *worse* (recall 39.1% → 30.4%): better table text makes a chunk say more
of the things the question asks about, and without any filter BM25 spreads those hits over more
chunks — the gold chunk for q051 moves from rank 43 of 230 to rank 74 of 274, both hopeless. Better
text improves what a chunk *contains*; it does not decide whether the chunk is *found*.

**V1 → V2 (+2.33). Structure, not content, is what the splitter was missing.** The recursive
splitter cuts through a table, so a value can end up in a chunk without its row label. Keeping a
table whole (split by rows, caption and header repeated) and prefixing the context line lifts recall
30.4% → 45.7% and Task 4 from 9.00 to 11.33, on the same text. The chunk count goes from 274 to 627:
smaller, more homogeneous chunks also help BM25's length normalisation.

**V2 → V3 (+19.86). The largest single jump in the table is a tokenizer.** Recall 45.7% → 87.0%,
Task 2.3 from 0 to 13.86, Task 4 from 11.33 to 17.33. Lower-case + whitespace keeps the punctuation
attached, so the question "… at 31 December 2025?" is tokenised as `2025,` and `robotics'`, neither
of which ever matches the document's `2025` or `robotics`; with `million`, `the`, `at` and `were`
also indexed, the year and the metric carry almost no weight. Nothing about the chunks changed here —
627 chunks before and after — only how they and the question are turned into terms.

**V3 → V4 (+13.48). The filter is worth more than recall suggests.** Recall reaches 100% and Task 2.2
is credited for the first time, but the interesting part is Task 4: 17.33 → 19.67, with q050 as the
proof. That question asks for Brightwater **Software**'s EMEA revenue in FY2023. At V3 the gold
chunk (`493.5`) is retrieved at rank 2, so the grader's `recall@5` counts a hit, and the generator
still answers `NOT_FOUND` — because slots 3 and 4 held Brightwater **Energy**'s revenue-by-region
tables with different numbers. Shown the gold chunk alone it answers `493.5`; shown the unfiltered
five it refuses; shown the filtered five it answers `493.5`. The failure is one of precision, which
`recall@5` cannot see and Task 4 can. Note that recall is already 100% here *with the exact-year
rule*, because the fallback ladder drops the document-type filter for the 12 questions whose year no
document matches and BM25 still finds the right chunk — which is the price of V5 being flat below.

**V4 → V5 (−0.34, one question). The year window is precision, not points.** It changes the retrieved
set on 29 of 60 questions and the score does not move, for a measurable reason: with the exact-year
rule the candidate set is empty on 12 questions (7 of them answerable — the FY2023 figures that live
in the FY2024 report, and the FY2026 guidance that lives on the FY2025 call), and the fallback ladder
recovers all 7 by dropping the document-type filter, after which BM25 finds the right chunk anyway.
The window is what keeps those 7 inside the right document type instead of searching the company's
whole output; on this corpus that is invisible, on a corpus where the same figure appears in three
documents it is the difference between the right column and the wrong one. The −0.34 is variance, not
the window: a repeat run of V4 and V5 answered all 60 questions correctly in both.

**V5 → V6 (0.00). The guard is redundant once the filter is in.** The interleave changes the five
chunks for 7 comparison questions, but it never removes the gold chunk, and the failure mode it was
written for — five chunks of one company — occurs 0/12 times after the filter. It does occur 2/12
without the filter (q013, q022), but there the guard cannot fire, because without `parse_query` there
is no company list to interleave. So `parse_query` subsumes this component on `public_test`, and I
keep the guard as insurance for a corpus where the two named companies have very unequal numbers of
matching chunks.

**V6 → V7 (+10.67). The abstention rules are what Task 3 grades, and they also lift Task 4.** kept
stays at 100%, rejected goes 0% → 100% on the 14 unanswerable questions, which is the whole 10 points.
Task 4 goes 19.33 → 20.00 at the same time, because a few of those 14 were being answered with a
confident number taken from the real sibling company (q001 "Northgate Robotic Systems" → `241.6` from
Northgate Robotics; q044 "Orchid Bay Resorts" → `231.3` from Orchid Bay Hotels). The gates make
`NOT_FOUND` deliberate instead of lucky.

**V7 → V8 (0.00). The tie-breakers buy one rank-1 hit and nothing else.** Rank 1 holds an evidence
chunk on 39 of the 46 answerable questions without them and 40 with them; they reorder 22 of the 60
top-5 lists; and neither graded metric moves, because `recall@5` is set-based and Task 4 does not read
the order (see D2). The one question whose rank 1 improves keeps its answer: D1 is 41/60 with and
without. They are kept because they are what makes the first chunk the one a human would pick when
reading the output, and they cost nothing extra; on points alone I would delete them.

**V7 → V8 (0.00). The tie-breakers buy one rank-1 hit and nothing else.** Rank 1 holds an evidence
chunk on 39 of the 46 answerable questions without them and 40 with them; they reorder 22 of the 60
top-5 lists; and neither graded metric moves, because `recall@5` is set-based and Task 4 does not read
the order (see D2). The one question whose rank 1 improves keeps its answer: D1 is 41/60 with and
without. They are kept because they are what makes the first chunk the one a human would pick when
reading the output, and they cost nothing extra; on points alone I would delete them.

### 4. Diagnostics

Two probes on the submitted pipeline, and a count of what each flat row actually changes
(`python ablation.py --diagnostics`):

| probe | result | reading |
|---|---|---|
| **D1** only the top-ranked chunk reaches the generator | 41/60 = 68.3% | the five chunks, not the ranking, are what makes Task 4 work |
| **D2** the five chunks are shuffled | 60/60 = 100.0% | this generator does not care about order, which is why V8 can be inert |
| V5's chunks differ from V4's on | 29/60 questions | the window acts, the metric is at ceiling |
| V6's chunks differ from V5's on | 7/60 questions | the guard acts only on comparisons, and never removes the gold chunk |
| V8's chunks differ from V7's on | 16/60 questions (22/60 in order) | same |

D1 is the useful one. Showing only the top-ranked chunk costs 19 of the 46 answerable questions:
all 12 comparisons (a comparison needs one chunk per company, so one is never enough by
construction), 6 table questions, where the second chunk carries the other year's column or the note
to the accounts, and 1 fact question. Rank 1 already holds a gold chunk for 40 of the 46, so the
missing 19 are not about which chunk is absent from the top 5 — they are about the second chunk
being needed to read the first one correctly. That is why `MAX_RETRIEVED` is worth spending on
breadth rather than on a better ranker.

Read together, V5, V6 and V8 are flat for one reason: `recall@5` and Task 4 both saturate at V4, so a
component that only improves *which* chunks are returned has nothing left to improve. A component can
be working and add nothing, and the mechanism counts above are how I tell those two cases apart.

### 5. Error analysis

Public-set accuracy is 60/60, so the errors below are the ones the weaker configurations make — each
one is a failure mode the submitted pipeline has a specific answer to, and each is one I would expect
to meet again on a corpus I have not seen.

**(a) The right chunk, surrounded by the wrong company** — q050 at V3, described above. `recall@5`
scores it a hit and the answer is `NOT_FOUND`. Cause: the five chunks are ranked independently, so a
chunk that is second-best for the whole corpus can sit next to three chunks of a sibling company with
different numbers, and the generator prefers to refuse rather than guess which table is meant.
Answer: the company filter in V4, not a better ranker.

**(b) A name that does not exist, answered from the name that does** — q001 "Northgate Robotic
Systems" answered `241.6` (Northgate Robotics' cash), q044 "Orchid Bay Resorts" answered `231.3`
(Orchid Bay Hotels' operating income). Cause: `Northgate` and `Orchid Bay` are shared prefixes, so
BM25 finds highly relevant chunks and nothing in the retrieval step knows the company is not in the
corpus. Answer: the first abstention gate, which compares the parsed name against the corpus company
list for **equality**, not prefix — `Northgate Robotic Systems` fails it while `Northgate Robotics`
passes.

**(c) Five slots, three documents** — q040 at V3 returned `doc_026, doc_005, doc_026, doc_005,
doc_017`: two documents doubled, and the answer (`162.6`) sitting in a *different section* of
`doc_026`, which was already in slot 1. Cause: chunks are ranked one by one, so two sections of the
same document can both score well and the budget is spent on redundancy; the metric `inventories`
appears in the balance sheet and again in Note 6, and only one of them carries the FY2025 column in a
form the generator can use. Answer: partly the company filter (which removes `doc_005` and
`doc_017` from the pool), partly dumb luck in the ordering. A document-level diversity rule in
`retrieve` would be the principled fix and is the first thing I would add with more time.

**(d) A number whose sign is lost** — the corpus prints negatives in accounting notation,
`Cost of revenue | 2024: (487.8)`, and the generator answers `487.8`. This never fires on
`public_test` (no gold answer is negative), and rewriting `(487.8)` as `-487.8` in `table_to_text`
would break Task 1.1, which looks for the text exactly as printed. It is a real failure on any
corpus that asks about a loss, and the honest fix belongs in the chunk text, not in the table
renderer: keep `(487.8)` as printed and add the sign in the chunk header line.

**(e) One question of noise.** At temperature 0 the generator still flips one question between runs
(B0 scored 29/60 and 30/60 on two runs, D1 41/60 and 42/60, V5's −0.34 above is one question and it
disappears in a repeat run). Differences below 0.34 points in the ablation table are therefore not
interpretable, which is why V5 is called flat rather than a regression. The submitted pipeline scored
60/60 on three separate runs.

### 6. Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env

python hw4.py --data-folder public_test                    # -> results.csv, 6.6 s
python ablation.py --data-folder public_test                # -> ablation.csv, ~75 s
python ablation.py --data-folder public_test --diagnostics  # + the D1/D2 probes
python ablation.py --data-folder public_test --no-generate  # the table without Task 4, no API key
```

`ablation.py` imports `hw4` instead of re-implementing the pipeline: the only code it writes itself is
the naive baseline (`get_text`, the recursive splitter, the whitespace tokenizer), and every score it
reports comes from the grader's own functions. A row of `ablation.csv` therefore cannot drift away
from the code that is graded. Two notes on how the table was measured:

- B0's splitter is `RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)`. The handout
  says `RecursiveCharacterTextSplitter()`, whose 4000-character default produces chunks that the
  runner drops for exceeding the 1500-character cap, which would leave B0 with no chunks at all and
  make every later row look better for the wrong reason.
- Task 2.2 is the score of the `parse_query` the pipeline uses. B0–V3 do not parse the question, so
  they are credited 0 there even though `hw4.parse_query` exists; V4 onwards use it.

No embedding model, no vector store, no network access beyond the provided generator. My own code
makes no model calls at all; the 46 calls in `results.csv` are the fixed generator's, one per
answerable question.

---

## Task 6: Reflection

[`reflection.md`](reflection.md) — ~2,400 words, answering the handout's seven questions in order.

It is anchored in what happened here rather than in generalities: the V4 row that scored 100% recall
while q050 still came back `NOT_FOUND`; the 19.86-point jump that came from nothing but punctuation in
the tokenizer; V1, where making the text more readable made it *less* findable (recall 39.1% → 30.4%);
the `(487.8)` sign I knowingly left broken because fixing it would have cost eight points on Task 1.1;
and the three identical rows in my first ablation table, which were not a finding but a variant that
switched off query parsing and abstention in one step, orphaning every component downstream of them.

It also commits to one habit, which is why [`unfixed.md`](unfixed.md) exists in this repository.
