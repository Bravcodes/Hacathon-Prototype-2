"""
Results page: match cards, hidden gems, compare two schools, look up any
school (Boston University by default), plus the "How it works" page.

Everything here is layout. The numbers come from scoring.py.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import advisor
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
    # Admission chances first: it's one of the scored factors when there's a test score
    t = t.join(scoring.admission_fit(t, profile.get("sat"), profile.get("act")))
    pcts = scoring.percentiles(t)
    pts = scoring.score(pcts, profile["weights"])
    t = t.join(pcts.add_prefix("pct_")).join(pts.add_prefix("pts_"))
    t["match"] = pts["score"]
    t["reasons"] = scoring.filter_reasons(
        t, profile["max_price"], profile["regions"], profile["sizes"],
        profile["settings"], profile["major"])
    test = "SAT" if profile.get("sat") else "ACT"
    t.loc[t["too_low"], "reasons"] = t.loc[t["too_low"], "reasons"].map(
        lambda r: r + [f"usually admits students with a much higher {test}"])
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
    parts = [f"academic fit ({row['pct_admission']:.0f}/100 for your score)" if k == "admission"
             else f"{LABELS[k].lower()} (top {max(1, 100 - row[f'pct_{k}']):.0f}% of colleges)"
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
        paper_bgcolor=plots.CLEAR, plot_bgcolor=plots.CLEAR,
    )
    return fig


def expected_line(row):
    pred, actual = row["predicted_grad_rate"] * 100, row["grad_rate"] * 100
    return (f"Based on the students it enrolls, we predicted **{pred:.0f}%** of all its students "
            f"would graduate. **{actual:.0f}%** actually did.")


ADMIT_DOTS = {"Likely": "🟢 Likely", "Match": "🟠 Match", "Reach": "🔴 Reach"}


def card_title(i, row):
    """One-line summary shown on a folded card."""
    parts = [f"**#{i} · {row['name']}**", row["state"], f"Match {row['match']:.0f}/100",
             f"{money(row['your_price'])} per year"]
    if row.get("admit_fit") in ADMIT_DOTS:
        parts.append(ADMIT_DOTS[row["admit_fit"]])
    if row.get("added") == "prestige":
        parts.append("⭐ added for your strong scores")
    elif row["reasons"]:
        parts.append("⚠️ " + ("over your budget" if all(r.startswith(PRICE_REASON) for r in row["reasons"])
                             else "doesn't fit every answer"))
    return esc(" · ".join(parts))


def card(row, profile, key):
    """The full match card for one school."""
    weights, major = profile["weights"], profile["major"]
    with st.container(border=not key.startswith("top")):  # match cards sit in an expander
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
            st.metric("Graduation chance", f"{row['your_grad'] * 100:.0f}%")
            st.caption(f"Finish within 6 years ({row['grad_source']})")
        with cols[2]:
            st.metric("Typical earnings", money(row["your_earnings"]))
            st.caption(row["earn_source"].capitalize())
        with cols[3]:
            st.metric("Median debt", money(row["your_debt"]))
            st.caption(row["debt_source"].capitalize())

        st.markdown(why_line(row, weights, major))
        st.plotly_chart(factor_bar(row, weights), key=f"bar-{key}-{row['unitid']}",
                        width="stretch", theme="streamlit", config={"displayModeBar": False})
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
    st.write("Answers that rule out the most colleges (try loosening one):")
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
            st.write(esc(f"Match {row['match']:.0f}/100 · {money(row['your_price'])} per year"))
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
def balanced_top(fit, n):
    """About 3/8 Reach, 3/8 Match, 2/8 Likely (best match score in each), topped up
    with the next-best colleges if a group runs short; shown best-first."""
    want = {"Reach": round(n * 3 / 8), "Match": round(n * 3 / 8)}
    want["Likely"] = n - sum(want.values())
    picks = [fit[fit["admit_fit"] == label].head(k) for label, k in want.items()]
    chosen = pd.concat(picks)
    rest = fit[~fit.index.isin(chosen.index)].head(n - len(chosen))
    return pd.concat([chosen, rest]).sort_values("match", ascending=False)


# Extreme answers should never leave the page empty
PRICE_REASON = "costs more than"
PRESTIGE_TOP = 75                     # top 75 on our prestige score, or ...
PRESTIGE_ADMIT = 0.15                 # ... admits under 15% of applicants
STRONG_SCORE = {"sat": 1450, "act": 33}


def is_prestigious(t):
    return (t["prestige_score"].rank(ascending=False) <= PRESTIGE_TOP) | (t["admit_rate"] < PRESTIGE_ADMIT)


def score_gap(t, profile):
    """SAT points the student's score is below each college's 25th percentile
    (0 if at or above it, or if the student gave no score)."""
    sat = profile.get("sat") or scoring.ACT_TO_SAT.get(int(profile.get("act") or 0))
    if not sat or "sat_lo" not in t:
        return pd.Series(0.0, index=t.index)
    return (t["sat_lo"] - sat).clip(lower=0).fillna(0)


def strong_scorer(profile):
    return any((profile.get(k) or 0) >= v for k, v in STRONG_SCORE.items())


def pick_list(t, n, profile, balanced=False):
    """The colleges to show as match cards, plus notes explaining any fallback.

    1. Colleges that fit every answer, best match first (or a balanced mix).
    2. Too few? If the ONLY problem is price, add the most affordable colleges
       that match everything else (cheapest first). Otherwise add the colleges
       that miss the fewest answers. Their cards say what doesn't fit.
    3. Strong SAT/ACT: make sure at least one prestigious college is in the
       list (the best-matching one that fits, else the closest one).
    Returns (list DataFrame, list of notes). The "added" column says why a
    college that doesn't fit every answer is in the list: "budget", "closest"
    or "prestige" ("" for colleges that fit).
    """
    notes = []
    fit = t[t["fits"]]
    top = (balanced_top(fit, n) if balanced else fit.head(n)).assign(added="")

    if len(top) < n:
        rest = t[~t["fits"]]
        only_price = rest[rest["reasons"].map(lambda r: all(x.startswith(PRICE_REASON) for x in r))]
        if len(only_price):
            extra = only_price.nsmallest(n - len(top), "your_price").assign(added="budget")
            notes.append(("Nothing else is within your budget, so here are the **most affordable** "
                          "colleges that match everything else you chose (cheapest first)."
                          if len(top) else
                          "No college is within your budget for your family's income, so here are "
                          "the **most affordable** colleges that match everything else you chose "
                          "(cheapest first). Check each college's net price calculator: aid can "
                          "be higher than these averages."))
        else:
            # Fewest unmet answers first; among ties, the smallest gap between the
            # student's score and the college's typical range, then the best match
            order = rest.assign(_miss=rest["reasons"].str.len(),
                                _gap=score_gap(rest, profile)).sort_values(
                ["_miss", "_gap", "match"], ascending=[True, True, False])
            extra = order.head(n - len(top)).drop(columns=["_miss", "_gap"]).assign(added="closest")
            notes.append("No college fits every answer, so here are the ones that come **closest**. "
                         "Each card says what doesn't fit.")
        top = pd.concat([top, extra])

    if strong_scorer(profile) and not is_prestigious(t).loc[top.index].any():
        pres = t[is_prestigious(t)]
        best = pres[pres["fits"]].head(1)
        if best.empty:
            closest = pres["reasons"].str.len().min()
            best = pres[pres["reasons"].str.len() == closest].head(1)
            notes.append("With your scores you're competitive at top colleges, so we added the "
                         "closest **prestigious** college even though it doesn't fit every answer.")
        if len(top) >= n:
            top = top.iloc[: n - 1]
        top = pd.concat([top, best.assign(added="" if bool(best["fits"].iloc[0]) else "prestige")])
        # Fitting colleges first (best match first), then the added ones in order
        top = pd.concat([top[top["added"] == ""].sort_values("match", ascending=False),
                         top[top["added"] != ""]])
    return top, notes


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
        advisor.sidebar_chat(schools, fos)
        st.divider()
        st.header("Your answers")
        a = st.session_state.get("answers") or {}
        for q in quiz.QUESTIONS:
            value = a.get(q["key"])
            if q["key"] == "scores":
                test = f"{a.get('test')} {a.get('test_score')}" if a.get("test_score") else "no test score"
                value = f"GPA {a['gpa']:.1f}, {test}" if a.get("gpa") else test
            if q["key"] == "max_price" and value is not None:
                value = "No limit" if value >= quiz.NO_LIMIT else f"${value:,.0f} per year"
            if isinstance(value, list):
                value = ", ".join(value) or "none"
            st.caption(esc(f"**{q['title']}** {value or '—'}"))
        if st.button("Change my answers", width="stretch"):
            st.session_state["quiz_step"] = 0
            go_to("quiz")
            st.rerun()
        n = st.slider("How many matches to show", 5, 10, 8)
        open_all = st.toggle("Open all match cards", value=False)
        balanced = st.toggle("Balanced list: mix of Reach, Match and Likely",
                             help="Counselors usually suggest applying to a few of each. Shows about "
                                  "3 Reach, 3 Match and 2 Likely colleges (best matches in each).")
        profile["hide_reach"] = st.checkbox(
            "Hide reach schools", help="Reach = your test score is below the middle 50% of "
            "admitted students, or the college admits under 15% of applicants.")

    t = build(schools, fos, profile)
    fit = t[t["fits"]]
    if len(fit):
        st.write(f"**{len(fit):,}** of {len(t):,} colleges fit your answers. "
                 "Here are your best matches, scored out of 100.")
    else:
        st.write(f"None of the {len(t):,} colleges fits every answer, so we show the closest "
                 "options instead (scored out of 100).")
    test_blocks = t["too_low"].sum() if "too_low" in t else 0
    if test_blocks > len(t) / 2:
        st.info(f"Your test score rules out {test_blocks:,} colleges. Many colleges are "
                "**test-optional**: you can apply without sending a score. To see those options, "
                "change your answer to \"Haven't taken / not sending\" (**Change my answers** "
                "in the sidebar).")
    bu_notice(t, n)

    top, notes = pick_list(t, n, profile, balanced)
    for note in notes:
        st.info(esc(note))
    if fit.empty:
        with st.expander("What's ruling colleges out?"):
            no_matches(t)
    if not top.empty:
        mix = top["admit_fit"].value_counts()
        st.caption(f"Your list: {mix.get('Reach', 0)} Reach · {mix.get('Match', 0)} Match · "
                   f"{mix.get('Likely', 0)} Likely"
                   + ("" if balanced or mix.get("Reach", 0) < len(top) - 1 else
                      " — mostly reaches. Try **Balanced list** in the sidebar."))
        st.caption("Tap a college to open or fold its card.")
        for i, (_, row) in enumerate(top.iterrows(), start=1):
            with st.expander(card_title(i, row), expanded=open_all or i == 1):
                card(row, profile, key=f"top{i}")
        st.divider()
        hidden_gems(fit, set(top["unitid"]), profile)

    st.divider()
    compare(t, top, profile)
    st.divider()
    look_up(t, profile)


def how_it_works(schools):
    st.title("How it works")
    steps = [
        ("1. Score each factor 0–100",
         "For price, graduation, earnings, debt, name recognition and \"beats expectations\", "
         f"every college gets a score from 0 to 100 compared with all {len(schools):,} four-year "
         "colleges (online-only and mostly-online universities are left out). "
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
        "With a score, **academic fit counts double** in your match score: it's highest when your "
        "score is inside a college's typical range (or a bit above), and lower when you're far "
        "below *or far above* it. Far above is called undermatching: strong students often "
        "enroll where they're far above the typical student, and graduate less often. For "
        "colleges that report no scores (like Caltech or the UCs) we **estimate** the typical "
        "SAT with a model trained on colleges that do (it's off by about 50 points on average). "
        "A college drops out of your matches if your score is more than 100 SAT (3 ACT) points "
        "below its 25th percentile, or under 1400 SAT / 31 ACT for colleges that admit under 20%. "
        "Without a score we go by admit rate only and it doesn't change the score. GPA, essays "
        "and activities aren't included: colleges don't report them to the federal data."
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
