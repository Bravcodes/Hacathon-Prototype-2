"""
Results page: match cards, hidden gems, compare two schools, look up any
school (Boston University by default), plus the "How it works" page.

Everything here is layout. The numbers come from scoring.py.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import plots
import quiz
import scoring

LABELS = {k: label for k, (_, _, label) in scoring.FACTORS.items()}


# ---------------------------------------------------------------------------
# Data for this student
# ---------------------------------------------------------------------------
def build(schools, fos, profile):
    """All schools, personalized + scored + flagged with filter reasons.

    Scores are computed on ALL schools first, then filtered, so a school's
    score never depends on which filters you picked.
    """
    t = scoring.personalize(schools, profile["income"], profile["major"], fos)
    pcts = scoring.percentiles(t)
    pts = scoring.score(pcts, profile["weights"])
    t = t.join(pcts.add_prefix("pct_")).join(pts.add_prefix("pts_"))
    t["match"] = pts["score"]
    t = t.join(scoring.admission_fit(t, profile.get("sat"), profile.get("act")))
    t["reasons"] = scoring.filter_reasons(
        t, profile["max_price"], profile["regions"], profile["sizes"],
        profile["settings"], profile["major"])
    if profile.get("hide_reach"):
        reach = t["admit_fit"] == "Reach"
        t.loc[reach, "reasons"] = t.loc[reach, "reasons"].map(lambda r: r + ["is a reach"])
    t["fits"] = t["reasons"].str.len() == 0
    return t.sort_values("match", ascending=False)


def active_factors(weights):
    return [k for k in scoring.FACTORS if weights.get(k, 0) > 0]


# ---------------------------------------------------------------------------
# Small pieces of a card
# ---------------------------------------------------------------------------
def money(x):
    return "n/a" if pd.isna(x) else f"${x:,.0f}"


def esc(text):
    """Streamlit markdown reads $...$ as math, so escape dollar signs."""
    return text.replace("$", "\\$")


def beats_badge(row):
    """Green badge (as markdown) if the school graduates more students than predicted."""
    gap = row["residual_grad_rate"] * 100
    if gap >= 2:
        return f":green-badge[:material/trending_up: Graduation beats expectations by +{gap:.0f} pts]"
    if gap <= -2:
        return f":gray-badge[Graduation {-gap:.0f} pts below expectations]"
    return ":gray-badge[Graduation about as expected]"


ADMIT_COLORS = {"Likely": "blue", "Match": "orange", "Reach": "red"}


def admit_badge(row):
    """Likely / Match / Reach badge (as markdown), or "" if we can't tell."""
    fit = row.get("admit_fit")
    if fit not in ADMIT_COLORS:
        return ""
    return f":{ADMIT_COLORS[fit]}-badge[:material/school: {fit}]"


def why_line(row, weights, major):
    """One sentence built from the two factors that earned the most points."""
    top = sorted(active_factors(weights), key=lambda k: row[f"pts_{k}"], reverse=True)[:2]
    parts = [f"{LABELS[k].lower()} (top {max(1, 100 - row[f'pct_{k}']):.0f}% of colleges)"
             for k in top]
    text = "**Why it's a match:** strong on " + " and ".join(parts)
    if major and pd.notna(row.get("major_share")):
        text += f". {row['major_share'] * 100:.0f}% of its degrees are in {major}"
    return text + "."


def factor_bar(row, weights):
    """One horizontal bar split into each factor's points (adds up to the score)."""
    fig = go.Figure()
    for k in active_factors(weights):
        p = row[f"pts_{k}"]
        fig.add_trace(go.Bar(
            x=[p], y=[""], orientation="h", name=f"{LABELS[k]} +{p:.0f} pts",
            marker=dict(color=scoring.FACTOR_COLORS[k], line=dict(color=plots.SURFACE, width=1)),
            hovertemplate=f"{LABELS[k]}: +{p:.1f} pts<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=130, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, 100], title=None, gridcolor=plots.GRID, ticksuffix=" pts"),
        legend=dict(orientation="h", y=-0.45, x=0, font=dict(size=12), traceorder="normal"),
        paper_bgcolor=plots.SURFACE, plot_bgcolor=plots.SURFACE, font=dict(color=plots.INK),
    )
    return fig


def expected_line(row):
    pred, actual = row["predicted_grad_rate"] * 100, row["grad_rate"] * 100
    return (f"Based on the students it enrolls, we predicted **{pred:.0f}%** would graduate. "
            f"**{actual:.0f}%** actually did.")


def card(row, profile, key):
    """The full match card for one school."""
    weights, major = profile["weights"], profile["major"]
    with st.container(border=True):
        head, score_col = st.columns([5, 1])
        with head:
            st.markdown(f"#### {row['name']}")
            size = "" if pd.isna(row["undergrads"]) else f" · {row['undergrads']:,.0f} undergrads"
            city = f"{row['city']}, " if isinstance(row.get("city"), str) else ""
            st.caption(f"{city}{row['state']} · {row['control_label']}{size}")
            st.markdown(f"{admit_badge(row)} {beats_badge(row)}")
            if row.get("admit_note"):
                st.caption(row["admit_note"])
        score_col.metric("Match", f"{row['match']:.0f}/100")

        if row["reasons"]:
            st.warning(esc("Doesn't fit your answers: " + "; ".join(row["reasons"]) + "."))

        cols = st.columns(4)
        with cols[0]:
            st.metric("Your price per year", money(row["your_price"]))
            st.caption(esc(f"{row['price_source'].capitalize()}. Sticker price: {money(row['sticker_price'])}"))
        with cols[1]:
            st.metric("Graduation chance", f"{row['grad_rate'] * 100:.0f}%")
            st.caption("Finish within 6 years")
        with cols[2]:
            st.metric("Typical earnings", money(row["your_earnings"]))
            st.caption(row["earn_source"].capitalize())
        with cols[3]:
            st.metric("Median debt", money(row["your_debt"]))
            st.caption(row["debt_source"].capitalize())

        st.markdown(why_line(row, weights, major))
        st.plotly_chart(factor_bar(row, weights), key=f"bar-{key}-{row['unitid']}",
                        width="stretch", theme=None, config={"displayModeBar": False})
        st.markdown(expected_line(row))


# ---------------------------------------------------------------------------
# Sections of the page
# ---------------------------------------------------------------------------
def bu_notice(t, n_shown):
    """Never hide Boston University without saying so."""
    bu = t[t["unitid"] == config.BU_UNITID]
    if bu.empty:
        return
    bu = bu.iloc[0]
    if not bu["fits"]:
        st.info(esc(f"**Boston University** isn't in your matches: it {'; '.join(bu['reasons'])}. "
                    "See its card anyway under **Look up any school** below."))
    else:
        rank = int((t[t["fits"]]["match"] > bu["match"]).sum()) + 1
        if rank > n_shown:
            st.caption(f"Boston University is #{rank} of your matches "
                       f"({bu['match']:.0f}/100). See it under **Look up any school**.")


def no_matches(t):
    """Explain which answer removes the most schools."""
    st.warning("No colleges fit all of your answers. Try loosening one of these:")
    counts = pd.Series([r for rs in t["reasons"] for r in rs]).value_counts()
    for reason, n in counts.items():
        st.write(esc(f"- {n:,} colleges {reason}"))


def hidden_gems(fit, top_ids, profile):
    """High value-added schools that are not very selective."""
    st.subheader("💎 Hidden gems")
    st.write("Less selective colleges (admit at least half of applicants) whose students "
             "do much better than we'd predict from who they enroll.")
    gems = fit[(fit["value_added"] >= 75) & (fit["admit_rate"] >= 0.5)
               & ~fit["unitid"].isin(top_ids)].head(4)
    if gems.empty:
        st.caption("None fit your answers. Try a wider region or a higher max price.")
        return
    for col, (_, row) in zip(st.columns(len(gems)), gems.iterrows()):
        with col, st.container(border=True):
            st.markdown(f"**{row['name']}**")
            st.caption(f"{row['state']} · admits {row['admit_rate'] * 100:.0f}%")
            st.markdown(f"{admit_badge(row)} {beats_badge(row)}")
            st.write(esc(f"Match {row['match']:.0f}/100 · {money(row['your_price'])}/yr"))
            st.caption(f"Predicted {row['predicted_grad_rate'] * 100:.0f}% would graduate; "
                       f"{row['grad_rate'] * 100:.0f}% did.")


def school_select(label, t, default_id, key):
    names = t.sort_values("name")
    ids = names["unitid"].tolist()
    lookup = dict(zip(names["unitid"], names["name"] + ", " + names["state"]))
    return st.selectbox(label, ids, index=ids.index(default_id) if default_id in ids else 0,
                        format_func=lookup.get, key=key)


def compare(t, fit, profile):
    """Pick two schools; see which wins each factor and by how many points."""
    st.subheader("⚖️ Compare two schools")
    first = fit["unitid"].iloc[0] if len(fit) else t["unitid"].iloc[0]
    second = config.BU_UNITID if first != config.BU_UNITID else t["unitid"].iloc[1]
    left, right = st.columns(2)
    with left:
        a_id = school_select("School A", t, first, "cmp-a")
    with right:
        b_id = school_select("School B", t, second, "cmp-b")
    a, b = t[t["unitid"] == a_id].iloc[0], t[t["unitid"] == b_id].iloc[0]

    rows = []
    for k in active_factors(profile["weights"]):
        pa, pb = a[f"pts_{k}"], b[f"pts_{k}"]
        winner = a["name"] if pa > pb else b["name"] if pb > pa else "Tie"
        rows.append({"Factor": LABELS[k], "School A": pa, "School B": pb,
                     "Winner": winner, "By": abs(pa - pb)})
    rows.append({"Factor": "Total match score", "School A": a["match"], "School B": b["match"],
                 "Winner": a["name"] if a["match"] > b["match"] else b["name"],
                 "By": abs(a["match"] - b["match"])})
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "School A": st.column_config.NumberColumn(f"A: {a['name']}", format="%.1f pts"),
            "School B": st.column_config.NumberColumn(f"B: {b['name']}", format="%.1f pts"),
            "By": st.column_config.NumberColumn("Wins by", format="%.1f pts"),
        },
    )
    wins_a = sum(r["Winner"] == a["name"] for r in rows[:-1])
    wins_b = sum(r["Winner"] == b["name"] for r in rows[:-1])
    st.caption(f"{a['name']} wins {wins_a} factors, {b['name']} wins {wins_b}. "
               "Points already include your weights.")


def look_up(t, profile):
    """Any school's card, Boston University by default, even if filtered out."""
    st.subheader("🔎 Look up any school")
    unitid = school_select("School", t, config.BU_UNITID, "lookup")
    card(t[t["unitid"] == unitid].iloc[0], profile, key="lookup")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def current_profile():
    """Profile from the quiz, or neutral defaults if the quiz wasn't taken."""
    answers = st.session_state.get("answers")
    if answers is None or not st.session_state.get("quiz_done"):
        st.info("These are general matches. Take the quiz to use your own price, major and priorities.")
        answers = dict(quiz.DEFAULTS, max_price=quiz.NO_LIMIT)
    return quiz.answers_to_profile(answers)


def render(schools, fos, go_to):
    st.title("Your matches")
    profile = current_profile()

    with st.sidebar:
        st.header("Your answers")
        a = st.session_state.get("answers") or {}
        for q in quiz.QUESTIONS:
            value = a.get(q["key"])
            if q["key"] == "scores":
                test = f"{a.get('test')} {a.get('test_score')}" if a.get("test_score") else "no test score"
                value = f"GPA {a['gpa']:.1f}, {test}" if a.get("gpa") else test
            if q["key"] == "max_price" and value is not None:
                value = "No limit" if value >= quiz.NO_LIMIT else f"${value:,.0f}/yr"
            if isinstance(value, list):
                value = ", ".join(value) or "none"
            st.caption(esc(f"**{q['title']}** {value or '—'}"))
        if st.button("Change my answers", width="stretch"):
            st.session_state["quiz_step"] = 0
            go_to("quiz")
            st.rerun()
        n = st.slider("How many matches to show", 5, 10, 8)
        profile["hide_reach"] = st.checkbox(
            "Hide reach schools", help="Reach = your test score is below the middle 50% of "
            "admitted students, or the college admits under 15% of applicants.")

    t = build(schools, fos, profile)
    fit = t[t["fits"]]
    st.write(f"**{len(fit):,}** of {len(t):,} colleges fit your answers. "
             "Here are your best matches, scored out of 100.")
    bu_notice(t, n)

    if fit.empty:
        no_matches(t)
    else:
        top = fit.head(n)
        for i, (_, row) in enumerate(top.iterrows(), start=1):
            st.markdown(f"**#{i}**")
            card(row, profile, key=f"top{i}")
        st.divider()
        hidden_gems(fit, set(top["unitid"]), profile)

    st.divider()
    compare(t, fit, profile)
    st.divider()
    look_up(t, profile)


def how_it_works():
    st.title("How it works")
    steps = [
        ("1. Score each factor 0–100",
         "For price, graduation, earnings, debt, name recognition and \"beats expectations\", "
         "every college gets a score from 0 to 100 compared with all 1,405 four-year colleges. "
         "80 means it does better than 80% of them."),
        ("2. Your answers set the weights",
         "Each factor counts once. Your two priorities count double. Price and earnings use "
         "your family's income bracket and your major when that data exists."),
        ("3. Add it up",
         "Each factor's score × its share of the weight = points. The points add up to the "
         "match score out of 100. The colored bar on every card shows exactly where they came from."),
    ]
    for col, (title, body) in zip(st.columns(3), steps):
        with col, st.container(border=True):
            st.markdown(f"### {title}")
            st.write(body)

    st.subheader("What does \"beats expectations\" mean?")
    st.write(
        "A college's graduation rate depends a lot on who it admits. So we trained a model "
        "that predicts each college's graduation rate and earnings from its students (share "
        "of low-income and first-generation students, SAT scores, admit rate) and its "
        "resources (spending per student, price). Colleges that do better than predicted "
        "\"beat expectations\": they seem to help students like you succeed."
    )
    scores = scoring.V2_DATA / "model_scores.csv"
    if scores.exists():
        s = pd.read_csv(scores).set_index(["target", "model"])
        st.caption(
            f"The model is right within about {s.loc[('grad_rate', 'forest'), 'mae'] * 100:.0f} "
            "percentage points on graduation rate for colleges it never saw "
            f"(R² = {s.loc[('grad_rate', 'forest'), 'r2']:.2f}). Details are in **Explore the data**.")

    st.subheader("Likely, Match or Reach?")
    st.write(
        "If you share an SAT or ACT score, we compare it with the middle 50% of students each "
        "college admitted: above that range is **Likely**, inside is **Match**, below is "
        "**Reach**. Colleges that admit under 15% of applicants are a Reach for everyone. "
        "Without a score we go by admit rate only. This label never changes the match score, "
        "and it ignores GPA, essays and activities, because colleges don't report those to "
        "the federal data."
    )

    st.subheader("What this score doesn't measure")
    st.markdown(
        "- **Campus life**: clubs, housing, food, how it feels to be there\n"
        "- **Safety** and support services\n"
        "- **The quality of a specific program** or professor\n"
        "- **Whether you'll get in**, or what aid *you* will get (check each college's "
        "net price calculator)\n"
        "- Earnings only count students who got federal financial aid, and depend on the "
        "job market where graduates live"
    )
