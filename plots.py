"""
Copied from src/plots.py so v2/ runs on its own.

Phase 2c: charts. Each function builds ONE Plotly figure and returns it.
`save()` writes it to figures/ as .html (interactive) and .png (for slides).

The app (Phase 3) can import these same functions, so charts look the same
everywhere.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config

# ---------------------------------------------------------------------------
# Shared look. Colors come from a colorblind-checked palette.
# ---------------------------------------------------------------------------
COLORS = {
    "Public": "#2a78d6",             # blue
    "Private nonprofit": "#eb6834",  # orange
    "For-profit": "#1baf7a",         # aqua
}
SINGLE = "#2a78d6"     # one-series charts
POSITIVE = "#2a78d6"   # above expectations
NEGATIVE = "#e34948"   # below expectations
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Friendly names for chart labels
LABELS = {
    "prestige_score": "Prestige score (0-100)",
    "outcome_score": "Outcome score (0-100)",
    "grad_rate": "Graduation rate",
    "earnings_10yr": "Median earnings, 10 yrs after entry ($)",
    "control_label": "Type",
    "pct_pell": "% Pell (low-income)",
    "pct_first_gen": "% first-generation",
    "admit_rate": "Admit rate",
    "sat_avg": "Average SAT",
    "sat_missing": "SAT not reported",
    "spend_per_student": "Spending per student",
    "net_price": "Net price",
}


def style(fig, title):
    """Apply the shared look: light background, thin gray grid, clear title."""
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=18, color=INK)),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=MUTED, size=13),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        margin=dict(l=60, r=30, t=70, b=60),
        legend=dict(title=None, orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def mark_school(fig, df, x, y, unitid=config.BU_UNITID, label="Boston University"):
    """Draw a ring and a label around one school (default: BU)."""
    row = df[df["unitid"] == unitid]
    if row.empty:
        return fig
    fig.add_trace(go.Scatter(
        x=row[x], y=row[y], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color=INK, width=2)),
    ))
    fig.add_annotation(
        x=row[x].iloc[0], y=row[y].iloc[0], text=f"<b>{label}</b>",
        showarrow=True, arrowhead=0, ax=40, ay=-40, font=dict(color=INK),
    )
    return fig


# ---------------------------------------------------------------------------
# The charts
# ---------------------------------------------------------------------------
def prestige_vs_outcome(df):
    """Each dot is a school: prestige (x) vs. outcomes (y)."""
    fig = px.scatter(
        df, x="prestige_score", y="outcome_score", color="control_label",
        color_discrete_map=COLORS, hover_name="name",
        hover_data={"state": True, "control_label": False,
                    "prestige_score": ":.0f", "outcome_score": ":.0f"},
        labels=LABELS, opacity=0.7, render_mode="svg",
    )
    fig.update_traces(marker=dict(size=8, line=dict(color=SURFACE, width=1)))
    style(fig, "Prestige vs. outcomes")
    return mark_school(fig, df, "prestige_score", "outcome_score")


def actual_vs_predicted(test_df, target):
    """Test-set check: dots on the diagonal = perfect predictions."""
    fig = px.scatter(test_df, x="predicted", y="actual", opacity=0.6,
                     color_discrete_sequence=[SINGLE])
    fig.update_traces(marker=dict(size=8, line=dict(color=SURFACE, width=1)))
    lo = min(test_df.min())
    hi = max(test_df.max())
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(color=MUTED, width=1, dash="dash"))
    name = LABELS.get(target, target)
    fig.update_layout(xaxis_title=f"Predicted {name.lower()}", yaxis_title=f"Actual {name.lower()}")
    if target == "grad_rate":  # show 0.25 as 25%
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
    return style(fig, f"Random forest on held-out schools: {name}")


def importance_bars(importance_df, target):
    """Which inputs matter most for predicting this outcome."""
    d = importance_df[importance_df["target"] == target].sort_values("forest_importance")
    d = d.assign(label=d["feature"].map(LABELS).fillna(d["feature"]))
    fig = px.bar(d, x="forest_importance", y="label", orientation="h",
                 color_discrete_sequence=[SINGLE])
    fig.update_layout(xaxis_title="Drop in R² when shuffled", yaxis_title=None)
    return style(fig, f"What predicts {LABELS.get(target, target).lower()}?")


def top_residuals(df, target, n=15):
    """Schools that beat their predicted outcome by the most."""
    col = f"residual_{target}"
    d = df.nlargest(n, col).sort_values(col)
    fig = px.bar(d, x=col, y="name", orientation="h", color_discrete_sequence=[POSITIVE],
                 hover_data={target: ":.2f", f"predicted_{target}": ":.2f"})
    fig.update_layout(xaxis_title="Actual minus predicted", yaxis_title=None, height=550)
    if target == "grad_rate":
        fig.update_xaxes(tickformat=".0%", title="Actual minus predicted (percentage points)")
    return style(fig, f"Top {n} overperformers: {LABELS.get(target, target).lower()}")


def gap_bars(df, n=10):
    """Biggest gaps between outcomes and prestige, both directions."""
    d = df.sort_values("gap")
    d = pd.concat([d.head(n), d.tail(n)])  # n worst + n best
    colors = [POSITIVE if g > 0 else NEGATIVE for g in d["gap"]]
    fig = go.Figure(go.Bar(
        x=d["gap"], y=d["name"], orientation="h", marker_color=colors,
        hovertemplate="%{y}<br>Outcome minus prestige: %{x:.0f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Outcome score minus prestige score", height=650)
    return style(fig, "Where outcomes and prestige disagree most")


def save(fig, filename):
    """Save as interactive HTML and, if possible, a PNG for slides."""
    config.FIGURES_DIR.mkdir(exist_ok=True)
    fig.write_html(config.FIGURES_DIR / f"{filename}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(config.FIGURES_DIR / f"{filename}.png", width=1100, height=650, scale=2)
    except Exception as err:  # PNG needs kaleido + Chrome; HTML still works without it
        print(f"  (skipped PNG for {filename}: {err.__class__.__name__})")
