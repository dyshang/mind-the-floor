# Mind the Floor

Degenerate-baseline audit of agentic benchmarks: what does a policy that never
reads its input score, and how much of the published state of the art does that
floor already capture?

```
paper/       LaTeX (neurips_2026 dblblindworkshop), fig1
floorcheck/  pip package: closed-form floors + CLI (tests: pytest; examples/
             holds the two sample label files the docs' commands use)
scripts/     analytic_floors.py          — label-statistics floors (MMLU-Pro, GPQA, GSM8K)
             check_appendix_numbers.py   — recomputes every derived number the paper
                                           quotes and checks it against the printed
                                           value; offline, no dependencies
             gpqa_position_floor.py      — realization band for a constant-position
                                           policy on a shuffled 4-option split
             executed/                   — degenerate policies for BFCL, tau-bench, ALFWorld
                bfcl_score.py            — scores predictions with the official AST checker
                                           (needs a py3.11 venv with bfcl_eval; numpy pin
                                           breaks 3.13)
                tau_static_floor.py      — execution-free replication of the tau-bench
                                           floor as a counting identity over the task set
             collect_results.py          — merges raw outputs + sota.json into
                                           executed_floors.json / annotated analytic_floors.json
             make_figure1.py             — capture-ratio dot plot
results/     JSON/JSONL outputs (tracked; every table and figure rebuilds
             from these, and check_appendix_numbers.py verifies them offline).
             reporting_census.json holds the per-paper coding sheet behind the
             claim that floors are rarely reported; sota.json records, per
             ceiling, whether it is verified and same-split, and why the ones
             the paper prints as "---" were dropped
```

## Verify without installing anything

`check_appendix_numbers.py` recomputes every derived number the paper quotes
and checks it against the printed value. It is standard-library only, so it
needs no venv, no install, no downloads, and no API keys — and it exits
non-zero on any mismatch.

```bash
python3 scripts/check_appendix_numbers.py
```

The package itself is dependency-free; its tests need only `pytest`, which the
`test` extra installs:

```bash
python3 -m venv .venv
.venv/bin/pip install -e 'floorcheck/[test]'
.venv/bin/python -m pytest floorcheck/tests/
```

Every command in `floorcheck/README.md` and in the paper's Appendix D runs as
written against the two sample label files in `floorcheck/examples/`.

## Rebuild the tables and figure

```bash
.venv/bin/pip install -r requirements-repro.txt
.venv/bin/python scripts/analytic_floors.py        # GPQA needs `huggingface-cli login`
.venv/bin/python scripts/collect_results.py
.venv/bin/python scripts/make_figure1.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Executed group needs external harnesses (see each script's docstring):
BFCL (gorilla repo), tau-bench (pip, small API model for user sim, budget
capped), ALFWorld (pip + data download). Re-running them is optional: the
per-episode results are released as JSONL under `results/`, so the tables
rebuild without them.

Apache 2.0 licensed (`LICENSE`); `floorcheck` is Apache 2.0 too and has no dependencies. No
benchmark data is redistributed here — `results/` holds only per-task outcomes
(ids, rewards, pass/fail), never task content.
