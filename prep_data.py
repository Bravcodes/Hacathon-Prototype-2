"""
v2 data prep: pull the few extra columns v2 needs out of the big raw Scorecard
files and save them as two SMALL csvs in v2/data/ (so teammates don't need the
4 GB download to run the app).

Run once from the project folder (needs the raw Scorecard folder):
    .venv/bin/python v2/prep_data.py

Reads (never writes):  v2/data/results.csv, College_Scorecard_Raw_Data_*/...
Writes:                v2/data/extra.csv, v2/data/field_of_study.csv,
                       v2/data/model_scores.csv, v2/data/feature_importance.csv
"""

import sys
from pathlib import Path

import pandas as pd

V2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_DIR))

import config  # noqa: E402
import scoring  # noqa: E402

OUT_DIR = V2_DIR / "data"
FOS_CSV = config.RAW_DIR / "Most-Recent-Cohorts-Field-of-Study.csv"

# Column names checked against data.yaml (the official data dictionary):
# NPT4{1..5}_PUB / _PRIV = average net price for family income bracket 1..5
#   1 = $0-30k, 2 = $30,001-48k, 3 = $48,001-75k, 4 = $75,001-110k, 5 = $110k+
NPT_COLS = [f"NPT4{i}_{kind}" for i in range(1, 6) for kind in ("PUB", "PRIV")]
PCIP_COLS = sorted({m["pcip"] for m in scoring.MAJORS.values() if m["pcip"]})
# SAT{VR,MT}{25,75} = 25th/75th percentile of admitted students' SAT reading and
# math scores; ACTCM{25,75} = same for the ACT composite. (No GPA in the Scorecard.)
TEST_COLS = ["SATVR25", "SATVR75", "SATMT25", "SATMT75", "ACTCM25", "ACTCM75"]
EXTRA_COLS = ["UNITID", "CITY", "LOCALE"] + NPT_COLS + PCIP_COLS + TEST_COLS


def build_extra(unitids):
    """Income-bracket net prices, share of degrees by major, city/locale, test ranges."""
    raw = pd.read_csv(config.RAW_CSV, usecols=EXTRA_COLS,
                      na_values=config.NULL_MARKERS, low_memory=False)
    raw = raw[raw["UNITID"].isin(unitids)]

    out = raw[["UNITID", "CITY", "LOCALE"] + PCIP_COLS].rename(
        columns={"UNITID": "unitid", "CITY": "city", "LOCALE": "locale"})
    # Each school fills only the public OR the private column, so combine them
    for i in range(1, 6):
        out[f"net_price_inc{i}"] = raw[f"NPT4{i}_PUB"].fillna(raw[f"NPT4{i}_PRIV"])
    # Total SAT range = reading + math. (Adding two 25th percentiles only
    # approximates the 25th percentile of the total, which is fine for a label.)
    out["sat25"] = raw["SATVR25"] + raw["SATMT25"]
    out["sat75"] = raw["SATVR75"] + raw["SATMT75"]
    out["act25"], out["act75"] = raw["ACTCM25"], raw["ACTCM75"]
    return out


def build_field_of_study(unitids):
    """Bachelor's-level earnings and debt for each quiz major at each school.

    The file has one row per (school, 4-digit program, credential level). We
    keep bachelor's degrees (CREDLEV 3), group programs into the 2-digit
    family that matches each quiz major (e.g. 11xx = computer science), and
    average them, weighting by how many degrees each program awards.
    """
    cols = ["UNITID", "CIPCODE", "CREDLEV", "IPEDSCOUNT2",
            "EARN_MDN_5YR", "DEBT_ALL_STGP_EVAL_MDN"]
    fos = pd.read_csv(FOS_CSV, usecols=cols, na_values=config.NULL_MARKERS,
                      dtype={"CIPCODE": str}, low_memory=False)
    fos = fos[(fos["CREDLEV"] == 3) & fos["UNITID"].isin(unitids)].copy()
    fos["family"] = fos["CIPCODE"].str[:2]
    fos["w"] = fos["IPEDSCOUNT2"].fillna(1).clip(lower=1)

    rows = []
    for major, info in scoring.MAJORS.items():
        if not info["cip"]:
            continue
        sub = fos[fos["family"] == info["cip"]]
        for unitid, g in sub.groupby("UNITID"):
            rows.append({
                "unitid": int(unitid),
                "major": major,
                "major_earnings_5yr": weighted_mean(g, "EARN_MDN_5YR"),
                "major_debt": weighted_mean(g, "DEBT_ALL_STGP_EVAL_MDN"),
            })
    out = pd.DataFrame(rows)
    return out.dropna(subset=["major_earnings_5yr", "major_debt"], how="all")


def weighted_mean(g, col):
    """Average of `col` weighted by degrees awarded, ignoring missing values."""
    g = g.dropna(subset=[col])
    return None if g.empty else (g[col] * g["w"]).sum() / g["w"].sum()


def main():
    results = pd.read_csv(config.RESULTS_CSV)
    unitids = set(results["unitid"])
    OUT_DIR.mkdir(exist_ok=True)

    extra = build_extra(unitids)
    extra.to_csv(OUT_DIR / "extra.csv", index=False)
    print(f"extra.csv: {len(extra):,} schools")

    if FOS_CSV.exists():
        fos = build_field_of_study(unitids)
        fos.to_csv(OUT_DIR / "field_of_study.csv", index=False)
        print(f"field_of_study.csv: {len(fos):,} school x major rows")
    else:
        print("Field-of-study file not found: app will use institution-level values.")

    # Fit the value-added models once and save their accuracy + importance
    schools = scoring.load_schools()
    model = scoring.fit_value_added(schools)
    model["scores"].to_csv(OUT_DIR / "model_scores.csv", index=False)
    model["importance"].to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print(model["scores"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
