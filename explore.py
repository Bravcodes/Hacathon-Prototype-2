"""
"Explore the data" page: the original prototype's tabs, now on v2's scoring.

Each tab lives in its own file in v2/sections/ (copied from app/sections/).
"""

import streamlit as st

import scoring
from sections import about, fragility, map_view, ranking, scatter, sidebar, value_added


def render(schools):
    st.title("Explore the data")
    st.write(
        "Do college rankings measure what's good for students? Pick what matters to you "
        "in the sidebar and see how the list changes, especially for low-income and "
        "first-gen students."
    )
    all_schools = scoring.personalize(schools)
    weights = sidebar.weight_sliders()
    df = sidebar.filters(all_schools)
    if df.empty:
        st.warning("No schools match these filters.")
        return

    # Percentiles use ALL schools, so filtering doesn't change a school's score
    pcts = scoring.percentiles(all_schools).loc[df.index]
    scores = scoring.score(pcts, weights)["score"]

    tabs = st.tabs(["Your ranking", "Fragility", "Prestige vs. outcomes",
                    "Overperformers", "Map", "About & limitations"])
    with tabs[0]:
        table = ranking.render(df, scores)
    with tabs[1]:
        fragility.render(df, pcts, weights)
    with tabs[2]:
        scatter.render(df)
    with tabs[3]:
        value_added.render(df)
    with tabs[4]:
        map_view.render(table)
    with tabs[5]:
        about.render()
