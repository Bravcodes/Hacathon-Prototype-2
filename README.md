# Rankings vs. Reality v2

A quiz-first version of the prototype. The original app in `app/` is unchanged.

```bash
.venv/bin/pip install -r v2/requirements.txt
.venv/bin/streamlit run v2/app.py --server.port 8502   # v2
.venv/bin/streamlit run app/app.py                     # original (port 8501)
```

v2 is self-contained: everything it needs is inside this folder, so it also works
as the root of its own repo (e.g. on Streamlit Cloud, main file `app.py`).
`config.py`, `plots.py` and `overperformers.py` are copies of the files in `src/`,
and `data/results.csv` is a copy of the analysis output. Rerun `src/analysis.py` and
copy `data/results.csv` here again if the analysis changes. The small files in
`data/` are already committed. Rebuild them only if the raw Scorecard data changes:
`.venv/bin/python v2/prep_data.py` (needs the raw Scorecard folder).

## Who edits what (one file each = no merge conflicts)

| File | What it does |
|---|---|
| `app.py` | page switching, nav, footer only |
| `scoring.py` | all math: personalize, percentiles, points, value-added model, filters |
| `quiz.py` | the 7-question landing page; `answers_to_profile()` maps answers to weights |
| `results.py` | match cards, hidden gems, compare, look up a school, How it works |
| `explore.py` + `sections/` | the original tabs ("Explore the data") |
| `prep_data.py` | one-time extract of extra columns from the raw Scorecard files |

## Data notes

- Price for your income: `NPT41_PUB`–`NPT45_PUB` / `NPT41_PRIV`–`NPT45_PRIV`, falling
  back to the average net price (`NPT4_PUB` / `NPT4_PRIV`).
- Major fit: `PCIP` share of degrees (at least 1% = offers the major). Earnings and debt
  by major come from `Most-Recent-Cohorts-Field-of-Study.csv` (bachelor's, 5 years after
  graduating), falling back to school-wide values, which are labelled as such.
- Percentiles are computed among all 1,405 colleges, so filters never change a score.
