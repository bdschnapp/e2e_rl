# Thesis

Build:

```bash
make
make view
make watch
make clean
```

Output:
- `build/main.pdf`

Relevant repo scripts:
- `python train.py ...`
- `python tune_controllers.py ...`
- `python scripts/generate_test_scenarios.py`
- `python benchmark.py ...`
- `python thesis/scripts/generate_results_tables.py`

Populate thesis result tables from experiment CSVs:

```bash
python thesis/scripts/generate_results_tables.py
```

This reads:
- `results/<scenario>/<obs_tag>/<reward>/aggregate.csv` from `eval.py`
- `results/**/summary_forward.csv` and `results/**/summary_reverse.csv` from `benchmark.py`

It then rewrites the result tables in `thesis/tables/`. Missing runs are left as `---`.

Layout:
- `chapters/` manuscript text
- `tables/` standalone tables
- `figures/` plots and diagrams
- `equations/` reusable equations
- `appendices/` appendix source
- `summaries/` short chapter notes
