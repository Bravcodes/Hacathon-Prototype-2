"""
v2 scoring engine: the data science core. No page layout in here, so it's easy
to test from a notebook or the command line.

How a school's match score works:
  1. personalize(): pick the right number for THIS student, e.g. the net price
     for their family income bracket and earnings for their intended major.
  2. percentiles(): turn each factor into a 0-100 percentile among ALL 1,405
     colleges (higher = better; price and debt are flipped). Using all colleges
     means a school's score doesn't change when you filter the list.
  3. score(): points = percentile x weight / total weight. The points add up
     to a 0-100 match score, so every point can be explained.

Value added ("beats expectations"):
  We predict graduation rate and earnings from who a college enrolls and what
  it spends. value added = actual - predicted. See fit_value_added().
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

V2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_DIR))

import config  # noqa: E402  (v2's copy of paths/column names)
import overperformers  # noqa: E402  (v2's copy of the model code)

V2_DATA = config.DATA_DIR

# ---------------------------------------------------------------------------
# Factors: key -> (column made by personalize(), higher is better?, UI label)
# ---------------------------------------------------------------------------
FACTORS = {
    "affordability": ("your_price", False, "Affordability"),
    "earnings": ("your_earnings", True, "Earnings"),
    "graduation": ("grad_rate", True, "Graduation"),
    "low_debt": ("your_debt", False, "Low debt"),
    "mobility": ("pell_grad_rate", True, "Low-income students graduate"),
    "prestige": ("prestige_score", True, "Famous name"),
    "value_added": ("value_added", True, "Beats expectations"),
}

# Colors for the factor bars (same order as FACTORS, colorblind-checked)
FACTOR_COLORS = {
    "affordability": "#2a78d6", "earnings": "#eb6834", "graduation": "#1baf7a",
    "low_debt": "#8f5bd6", "mobility": "#d6a32a", "prestige": "#8a8984",
    "value_added": "#e34948",
}

# Slider starting points for the Explore page (0 = ignore, 10 = very important)
PRESETS = {
    "Student outcomes": {"affordability": 6, "earnings": 6, "graduation": 6, "low_debt": 4,
                         "mobility": 8, "prestige": 0, "value_added": 0},
    "Prestige only (like many rankings)": {**{k: 0 for k in FACTORS}, "prestige": 10},
    "Value added only": {**{k: 0 for k in FACTORS}, "value_added": 10},
    "Equal weights": {k: 5 for k in FACTORS},
}

# Quiz majors -> PCIP column (share of degrees in that field) and the 2-digit
# CIP family used in the field-of-study file. Undecided = no major fit.
MAJORS = {
    "Computer science": {"pcip": "PCIP11", "cip": "11"},
    "Engineering": {"pcip": "PCIP14", "cip": "14"},
    "Business": {"pcip": "PCIP52", "cip": "52"},
    "Biology / Pre-med": {"pcip": "PCIP26", "cip": "26"},
    "Health / Nursing": {"pcip": "PCIP51", "cip": "51"},
    "Psychology": {"pcip": "PCIP42", "cip": "42"},
    "Arts / Design": {"pcip": "PCIP50", "cip": "50"},
    "Undecided": {"pcip": None, "cip": None},
}
MIN_MAJOR_SHARE = 0.01  # "offers your major" = at least 1% of degrees in it

# Family income brackets, matching the Scorecard's NPT41..NPT45 columns
INCOME_BRACKETS = {1: "$0–30k", 2: "$30–48k", 3: "$48–75k", 4: "$75–110k", 5: "$110k+"}

# Census regions, built from the state code
REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_TO_REGION = {s: r for r, states in REGIONS.items() for s in states.split()}

# School size by undergrad count
SIZES = {"Small": (0, 5000), "Medium": (5000, 15000), "Large": (15000, np.inf)}


def locale_to_setting(code):
    """IPEDS locale codes: 11-13 city, 21-23 suburb, 31-43 town or rural."""
    if pd.isna(code):
        return None
    return {1: "City", 2: "Suburb", 3: "Town / rural", 4: "Town / rural"}.get(int(code) // 10)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_schools():
    """data/results.csv (read-only) + v2's extra columns, one row per school."""
    df = pd.read_csv(config.RESULTS_CSV)
    extra_path = V2_DATA / "extra.csv"
    if extra_path.exists():
        df = df.merge(pd.read_csv(extra_path), on="unitid", how="left")
    df["region"] = df["state"].map(STATE_TO_REGION).fillna("Other (territories)")
    df["setting"] = df["locale"].map(locale_to_setting) if "locale" in df else None
    return df


def load_field_of_study():
    """Major-specific earnings/debt (long format), or None if not prepared."""
    path = V2_DATA / "field_of_study.csv"
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Value-added model
# ---------------------------------------------------------------------------
VA_FEATURE_LABELS = {
    "pct_pell": "% low-income (Pell) students",
    "pct_first_gen": "% first-generation students",
    "admit_rate": "Admit rate",
    "sat_avg": "Average SAT",
    "sat_missing": "SAT not reported",
    "spend_per_student": "Instructional spending per student",
    "net_price": "Average net price",
}


def fit_value_added(df):
    """Predict grad rate (C150_4) and 10-yr earnings (MD_EARN_WNE_P10).

    Uses overperformers.py (copied from src/ so the models match the analysis):
      - linear regression + random forest, 80/20 train/test split -> R², MAE
      - permutation importance on the test set
      - every school's prediction comes from 5-fold cross-validation, so no
        school is judged by a model that trained on it

    Returns {"schools": df with predicted_/residual_ columns + value_added,
             "scores": accuracy table, "importance": importance table}
    """
    old = [c for c in df if c.startswith(("predicted_", "residual_"))]
    out, scores, importance, _ = overperformers.run(df.drop(columns=old))

    # One combined number: average z-score of the two residuals, as 0-100
    z = sum((out[c] - out[c].mean()) / out[c].std()
            for c in ["residual_grad_rate", "residual_earnings_10yr"]) / 2
    out["value_added"] = z.rank(pct=True) * 100
    importance["feature_label"] = importance["feature"].map(VA_FEATURE_LABELS)
    return {"schools": out, "scores": scores, "importance": importance}


@st.cache_resource(show_spinner="Training the value-added model (once)...")
def get_value_added(df):
    """Cached version for the app: trains once per server, not per click."""
    return fit_value_added(df)


# ---------------------------------------------------------------------------
# Personalize -> percentiles -> score
# ---------------------------------------------------------------------------
def personalize(df, income=None, major=None, fos=None):
    """Add the numbers that depend on who is asking.

    your_price     net price for the family's income bracket, else average net price
    your_earnings  earnings for the major (5 yrs after graduating), else the
                   school's overall median (10 yrs after starting college)
    your_debt      median debt for the major, else the school's overall median
    *_source       label saying which one was used (shown in the UI)
    major_share    share of the school's degrees in that major (0-1)
    """
    df = df.copy()

    # Price for your income bracket
    df["your_price"] = df["net_price"]
    df["price_source"] = "average for all families"
    col = f"net_price_inc{income}"
    if income and col in df:
        has = df[col].notna()
        df.loc[has, "your_price"] = df.loc[has, col]
        df.loc[has, "price_source"] = f"for families earning {INCOME_BRACKETS[income]}"

    # Earnings and debt for your major
    df["your_earnings"] = df["earnings_10yr"]
    df["your_debt"] = df["median_debt"]
    df["earn_source"] = "all majors, 10 yrs after starting"
    df["debt_source"] = "all majors"
    df["major_share"] = np.nan
    info = MAJORS.get(major) or {}
    if info.get("pcip") and info["pcip"] in df:
        df["major_share"] = df[info["pcip"]]
    if info.get("cip") and fos is not None:
        m = fos[fos["major"] == major].set_index("unitid")
        earn = df["unitid"].map(m["major_earnings_5yr"])
        debt = df["unitid"].map(m["major_debt"])
        df.loc[earn.notna(), "your_earnings"] = earn
        df.loc[earn.notna(), "earn_source"] = f"{major} majors, 5 yrs after graduating"
        df.loc[debt.notna(), "your_debt"] = debt
        df.loc[debt.notna(), "debt_source"] = f"{major} majors"
    return df


def percentiles(df):
    """One 0-100 column per factor. Missing values get 50 (neutral).

    Earnings and debt are ranked WITHIN each source group, so a school's
    5-year major earnings are only compared with other schools' 5-year major
    earnings (not with 10-year all-major earnings, which are higher).
    """
    out = pd.DataFrame(index=df.index)
    groups = {"earnings": "earn_source", "low_debt": "debt_source"}
    for key, (col, higher_is_better, _) in FACTORS.items():
        if col not in df:
            out[key] = 50.0
            continue
        values = df[col]
        if key in groups and groups[key] in df:
            pct = values.groupby(df[groups[key]]).rank(pct=True, ascending=higher_is_better)
        else:
            pct = values.rank(pct=True, ascending=higher_is_better)
        out[key] = (pct * 100).fillna(50.0)
    return out


def score(pcts, weights):
    """Points per factor (adds up to the 0-100 match score) + the total.

    Returns a DataFrame: one column per factor in FACTORS (points) + "score".
    """
    w = pd.Series({k: weights.get(k, 0) for k in pcts.columns}, dtype=float)
    if w.sum() == 0:
        points = pcts * 0.0
    else:
        points = pcts.mul(w / w.sum())
    return points.assign(score=points.sum(axis=1))


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def filter_reasons(df, max_price=None, regions=None, sizes=None, settings=None, major=None):
    """Why each school is filtered out (empty list = it passes).

    Returns a Series of lists, so the UI can say e.g. "Boston University is
    hidden because: costs more than your max" instead of silently hiding it.
    """
    reasons = pd.Series([[] for _ in range(len(df))], index=df.index)

    def flag(mask, text):
        for i in df.index[mask.fillna(False)]:
            reasons[i] = reasons[i] + [text]

    if max_price is not None:
        flag(df["your_price"] > max_price, f"costs more than ${max_price:,.0f}/yr")
    if regions:
        flag(~df["region"].isin(regions), "outside your region")
    if sizes:
        ok = pd.Series(False, index=df.index)
        for s in sizes:
            lo, hi = SIZES[s]
            ok |= df["undergrads"].between(lo, hi, inclusive="left")
        flag(~ok, "not the size you picked")
    if settings and "setting" in df:
        flag(~df["setting"].isin(settings), "not the setting you picked")
    if major and MAJORS.get(major, {}).get("pcip") and "major_share" in df:
        flag(df["major_share"].fillna(0) < MIN_MAJOR_SHARE, f"few or no {major} degrees")
    return reasons


# ---------------------------------------------------------------------------
# Admission fit: Likely / Match / Reach
# ---------------------------------------------------------------------------
VERY_SELECTIVE = 0.15  # admit rate below this = a reach for everyone
SELECTIVE = 0.30       # admit rate below this = never "likely"


def admission_fit(df, sat=None, act=None):
    """Label each school Likely / Match / Reach from the student's test score.

    Compares the score with the middle 50% (25th-75th percentile) of admitted
    students: below = Reach, inside = Match, above = Likely. Very selective
    schools are a Reach for everyone. Without a score (or for test-optional
    schools that report no range) we fall back to the admit rate alone.

    The Scorecard has no GPA data, so GPA is not used here.
    This label does NOT change the match score.

    Returns a DataFrame with "admit_fit" (Likely/Match/Reach/None) and "admit_note".
    """
    if sat:
        score, lo, hi, test = sat, df.get("sat25"), df.get("sat75"), "SAT"
    elif act:
        score, lo, hi, test = act, df.get("act25"), df.get("act75"), "ACT"
    else:
        score, lo, hi, test = None, None, None, None

    fit = pd.Series(None, index=df.index, dtype=object)
    note = ("Admits " + (df["admit_rate"] * 100).round().astype("Int64").astype(str)
            + "% of applicants").where(df["admit_rate"].notna(), "")

    # 1. Admit rate only (no score, or school reports no range)
    fit[df["admit_rate"] >= 0.75] = "Likely"
    fit[df["admit_rate"] < SELECTIVE] = "Reach"

    # 2. Score vs. the school's middle 50%, where we have both
    if score is not None and lo is not None:
        has = lo.notna() & hi.notna()
        fit[has] = np.select([score < lo[has], score < hi[has]], ["Reach", "Match"], "Likely")
        note[has] = (f"Your {test} {score:,} vs. middle 50% admitted: "
                     + lo[has].astype(int).map("{:,}".format) + "–"
                     + hi[has].astype(int).map("{:,}".format))

    # 3. Selectivity caps
    fit[(df["admit_rate"] < SELECTIVE) & (fit == "Likely")] = "Match"
    fit[df["admit_rate"] < VERY_SELECTIVE] = "Reach"
    return pd.DataFrame({"admit_fit": fit, "admit_note": note})


# ---------------------------------------------------------------------------
# Fragility (used by the Explore page)
# ---------------------------------------------------------------------------
def fragility(pcts, weights, n_runs=500, wiggle=0.2, top_n=20, seed=0):
    """Monte Carlo check: do small weight changes reshuffle the top of the list?

    In each run, every weight is multiplied by a random number between
    (1 - wiggle) and (1 + wiggle). Weights set to 0 stay 0.

    Returns:
      summary  - per school: how often it lands in the top N, and its
                 typical best/worst rank (5th and 95th percentile)
      overlap  - per run: how many of the original top N are still top N
    """
    base = np.array([weights.get(k, 0) for k in pcts.columns], dtype=float)
    if base.sum() == 0:
        return None, None

    rng = np.random.default_rng(seed)
    run_weights = base * rng.uniform(1 - wiggle, 1 + wiggle, size=(n_runs, len(base)))
    scores = pcts.to_numpy() @ run_weights.T          # schools x runs
    ranks = (-scores).argsort(axis=0).argsort(axis=0) + 1

    baseline = score(pcts, weights)["score"].rank(ascending=False, method="first")
    in_base_top = (baseline <= top_n).to_numpy()
    in_run_top = ranks <= top_n

    summary = pd.DataFrame({
        "baseline_rank": baseline.astype(int),
        "pct_runs_in_top": in_run_top.mean(axis=1) * 100,
        "best_rank": np.percentile(ranks, 5, axis=1).round().astype(int),
        "worst_rank": np.percentile(ranks, 95, axis=1).round().astype(int),
    }, index=pcts.index)
    return summary, in_run_top[in_base_top].sum(axis=0)
