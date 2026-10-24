"""Tab: how stable is the top 20 if the weights change a little?"""

import plotly.graph_objects as go
import streamlit as st

import plots
import scoring

TOP_N = 20
N_RUNS = 500


def rank_range_chart(top):
    """Each school's typical rank range across all runs (5th-95th percentile)."""
    top = top.iloc[::-1]  # best school at the top of the chart
    # One line per school from best to worst rank. "None" breaks the line between schools.
    xs, ys = [], []
    for _, row in top.iterrows():
        xs += [row["best_rank"], row["worst_rank"], None]
        ys += [row["name"], row["name"], None]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                               line=dict(color=plots.SINGLE, width=2)))
    fig.add_trace(go.Scatter(
        x=top["baseline_rank"], y=top["name"], mode="markers", name="Rank with your weights",
        marker=dict(size=10, color=plots.SINGLE, line=dict(color=plots.SURFACE, width=2)),
        customdata=top[["best_rank", "worst_rank", "pct_runs_in_top"]],
        hovertemplate=("%{y}<br>Your rank: %{x}<br>Usual range: %{customdata[0]}"
                       "-%{customdata[1]}<br>In top 20: %{customdata[2]:.0f}% of runs<extra></extra>"),
    ))
    fig.update_layout(xaxis_title="Rank (line = range in 90% of runs; dot = rank with your weights)",
                      height=650, showlegend=False)
    plots.style(fig, f"Top {TOP_N}: how much each rank moves")
    fig.update_layout(margin=dict(l=300))  # room for long school names
    return fig


def render(df, pcts, weights):
    st.subheader("Fragility check")
    st.write(
        f"We re-run the ranking {N_RUNS} times, nudging every weight up or down by a "
        "random amount. If the top of the list reshuffles a lot, the ranking depends "
        "on small, arbitrary choices more than on real differences between schools."
    )
    wiggle = st.slider("How much to nudge each weight (±%)", 5, 50, 20, step=5) / 100

    summary, overlap = scoring.fragility(pcts, weights, n_runs=N_RUNS, wiggle=wiggle, top_n=TOP_N)
    if summary is None:
        st.info("Set at least one weight above zero.")
        return

    summary = summary.join(df[["name", "state"]])
    ever_top = (summary["pct_runs_in_top"] > 0).sum()

    a, b, c = st.columns(3)
    a.metric(f"Original top {TOP_N} that stay, on average", f"{overlap.mean():.1f} of {TOP_N}")
    b.metric("Worst run", f"{overlap.min()} of {TOP_N}")
    c.metric(f"Schools that reach the top {TOP_N} in some run", ever_top)

    top = summary.sort_values("baseline_rank").head(TOP_N)
    st.plotly_chart(rank_range_chart(top), width="stretch", theme=None)

    st.dataframe(
        top[["baseline_rank", "name", "state", "pct_runs_in_top", "best_rank", "worst_rank"]],
        column_config={
            "baseline_rank": "Your rank",
            "name": "School",
            "state": "State",
            "pct_runs_in_top": st.column_config.ProgressColumn(
                f"% of runs in top {TOP_N}", min_value=0, max_value=100, format="%.0f%%"),
            "best_rank": "Best (5th pct)",
            "worst_rank": "Worst (95th pct)",
        },
        hide_index=True, width="stretch",
    )
