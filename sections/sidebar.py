"""Sidebar: weight sliders and filters. Returns what the user picked."""

import streamlit as st

import scoring


def weight_sliders():
    """A preset picker plus one 0-10 slider per criterion."""
    st.sidebar.header("What matters to you?")
    preset = st.sidebar.selectbox("Start from", list(scoring.PRESETS))
    defaults = scoring.PRESETS[preset]

    weights = {}
    for key, (_, _, label) in scoring.FACTORS.items():
        # Putting the preset in the key resets the sliders when the preset changes
        weights[key] = st.sidebar.slider(label, 0, 10, defaults[key], key=f"{preset}-{key}")
    return weights


def filters(df):
    """Optional filters. Returns the filtered table."""
    st.sidebar.header("Filters")
    types = st.sidebar.multiselect(
        "Type", sorted(df["control_label"].unique()), default=sorted(df["control_label"].unique())
    )
    states = st.sidebar.multiselect("States (empty = all)", sorted(df["state"].unique()))
    min_size = st.sidebar.slider("Minimum undergrads", 0, 20000, 0, step=500)

    keep = df["control_label"].isin(types) & (df["undergrads"].fillna(0) >= min_size)
    if states:
        keep &= df["state"].isin(states)
    return df[keep]
