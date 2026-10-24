"""
Rankings vs. Reality v2 -- Streamlit app.

Run from the project folder (old app still runs with: streamlit run app/app.py):
    .venv/bin/streamlit run v2/app.py --server.port 8502

This file only loads data, switches pages and draws the nav + footer.
Each page lives in its own file:
    quiz.py      landing page (7-question quiz)
    results.py   match cards, hidden gems, compare, how it works
    explore.py   the original prototype's tabs ("Explore the data")
    scoring.py   all the math (no page layout)
"""

import sys
from pathlib import Path

import streamlit as st

V2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_DIR))  # so v2's own modules win over any same-named ones

import config  # noqa: E402
import explore  # noqa: E402
import quiz  # noqa: E402
import results  # noqa: E402
import scoring  # noqa: E402

st.set_page_config(page_title="Rankings vs. Reality", page_icon="🎓", layout="wide")

PAGES = {"quiz": "Take the quiz", "results": "Your matches",
         "explore": "Explore the data", "how": "How it works"}
FOOTER = ("A starting point, not advice. Talk to your counselor. "
          "Data: U.S. Dept. of Education College Scorecard.")


@st.cache_data(show_spinner="Loading colleges...")
def load_schools():
    """All 1,405 schools + value-added columns (model is trained once, cached)."""
    return scoring.get_value_added(scoring.load_schools())["schools"]


@st.cache_data
def load_field_of_study():
    return scoring.load_field_of_study()


def go_to(page):
    st.session_state["page"] = page


def nav():
    """Row of page buttons at the top. The current page is highlighted."""
    cols = st.columns(len(PAGES))
    for col, (key, label) in zip(cols, PAGES.items()):
        current = st.session_state["page"] == key
        if col.button(label, key=f"nav-{key}", width="stretch",
                      type="primary" if current else "tertiary"):
            go_to(key)
            st.rerun()


def main():
    if not config.RESULTS_CSV.exists():
        st.error("v2/data/results.csv not found. Copy it from data/results.csv.")
        st.stop()
    st.session_state.setdefault("page", "quiz")

    nav()
    schools, fos = load_schools(), load_field_of_study()
    page = st.session_state["page"]
    if page == "quiz":
        quiz.render(go_to)
    elif page == "results":
        results.render(schools, fos, go_to)
    elif page == "explore":
        explore.render(schools)
    else:
        results.how_it_works()

    st.divider()
    st.caption(FOOTER)


main()
