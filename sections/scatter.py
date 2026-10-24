"""Tab: prestige vs. outcomes scatter, with any school highlighted."""

import plotly.express as px
import streamlit as st

import config
import plots

# What the user can put on the y-axis
Y_OPTIONS = {
    "Outcome score (0-100)": "outcome_score",
    "Beats expectations (0-100)": "value_added",
    "Graduation rate: actual minus predicted": "residual_grad_rate",
    "Earnings: actual minus predicted ($)": "residual_earnings_10yr",
}


def school_picker(df):
    """Dropdown to choose which school gets the ring and label (BU by default)."""
    names = df.sort_values("name")[["unitid", "name"]]
    ids = names["unitid"].tolist()
    default = ids.index(config.BU_UNITID) if config.BU_UNITID in ids else 0
    unitid = st.selectbox("Highlight a school", ids, index=default,
                          format_func=dict(zip(names["unitid"], names["name"])).get)
    return unitid, df.loc[df["unitid"] == unitid, "name"].iloc[0]


def render(df):
    st.subheader("Prestige vs. outcomes")
    st.write(
        "Each dot is a school. Prestige combines admit rate, SAT, faculty pay and "
        "spending. The \"actual minus predicted\" options show how much a school beats "
        "what our model expected for its students and resources."
    )
    left, right = st.columns(2)
    y_label = left.selectbox("Y-axis", list(Y_OPTIONS))
    with right:
        unitid, name = school_picker(df)
    y = Y_OPTIONS[y_label]

    fig = px.scatter(
        df, x="prestige_score", y=y, color="control_label",
        color_discrete_map=plots.COLORS, hover_name="name",
        hover_data={"state": True, "control_label": False, "prestige_score": ":.0f", y: ":.2f"},
        labels={**plots.LABELS, y: y_label}, opacity=0.7, height=600,
        render_mode="svg",  # works even without WebGL (e.g. on a projector laptop)
    )
    fig.update_traces(marker=dict(size=8, line=dict(color=plots.SURFACE, width=1)))
    plots.style(fig, f"Prestige vs. {y_label.lower()}")
    plots.mark_school(fig, df, "prestige_score", y, unitid=unitid, label=name)
    st.plotly_chart(fig, width="stretch", theme=None)

    corr = df["prestige_score"].corr(df[y], method="spearman")
    st.caption(f"Spearman correlation between prestige and this measure: **{corr:.2f}** "
               "(1 = same order, 0 = no relationship).")
