"""
"Rankings or Reality?" -- a 10-round higher-or-lower game.

Each round shows two real colleges (usually one famous, one not) and asks one
question, e.g. "Which costs a family earning $0-30k less?". The player picks,
we reveal the real numbers and a one-line lesson. The point: prestige tells you
less than you'd think about price, graduation and who a college helps.

Nothing is saved: the game lives in st.session_state["game"].
"""

import numpy as np
import pandas as pd
import streamlit as st

import scoring

N_ROUNDS = 10
FAMOUS_TOP = 100      # "famous" = top 100 on our prestige score
SURPRISE_SHARE = 0.6  # share of rounds where the less famous school wins

# ---------------------------------------------------------------------------
# Question types. Each one: the column, whether higher wins, the smallest gap
# that counts as a real difference, how to show a value, and the lesson.
# ---------------------------------------------------------------------------
QUESTIONS = {
    "price": {
        "ask": "Which one costs a family earning {bracket} <b>less</b> per year?",
        "col": "your_price", "higher_wins": False, "min_gap": 2000,
        "fmt": lambda v: f"${v:,.0f}/yr",
        "lesson": "Sticker price isn't what you pay. Wealthy colleges often give big grants "
                  "to lower-income families, so a 'famous' school can cost less than you'd guess.",
    },
    "grad": {
        "ask": "Which one graduates <b>more</b> of its students within 6 years?",
        "col": "grad_rate", "higher_wins": True, "min_gap": 0.05,
        "fmt": lambda v: f"{v * 100:.0f}%",
        "lesson": "Graduation rates mostly reflect who a college admits. Selective schools "
                  "start with students who were already very likely to finish.",
    },
    "pell_grad": {
        "ask": "Which one graduates <b>more</b> of its low-income (Pell grant) students?",
        "col": "pell_grad_rate", "higher_wins": True, "min_gap": 0.05,
        "fmt": lambda v: f"{v * 100:.0f}%",
        "lesson": "For low-income students, some less famous colleges do as well as, or better "
                  "than, schools with much bigger names.",
    },
    "beats": {
        "ask": "Which one <b>beats expectations</b> by more? (graduates more students than our "
               "model predicts from who it enrolls)",
        "col": "residual_grad_rate", "higher_wins": True, "min_gap": 0.05,
        "fmt": lambda v: f"{v * 100:+.0f} pts",
        "lesson": "Prestige barely predicts how much a college adds (correlation 0.12). "
                  "Some of the biggest over-performers are public universities you may never "
                  "have heard of.",
    },
    "earnings": {
        "ask": "Whose students earn <b>more</b> 10 years after starting college?",
        "col": "earnings_10yr", "higher_wins": True, "min_gap": 5000,
        "fmt": lambda v: f"${v:,.0f}",
        "lesson": "Earnings depend a lot on majors and local job markets, not just the college: "
                  "engineering-heavy and big-city schools often come out ahead.",
    },
    "debt": {
        "ask": "Whose graduates leave with <b>less</b> debt?",
        "col": "median_debt", "higher_wins": False, "min_gap": 2000,
        "fmt": lambda v: f"${v:,.0f}",
        "lesson": "Debt depends on price AND aid. A school with a high sticker price can still "
                  "leave its students with less debt.",
    },
    "pell_share": {
        "ask": "Which one enrolls <b>more</b> low-income (Pell grant) students?",
        "col": "pct_pell", "higher_wins": True, "min_gap": 0.05,
        "fmt": lambda v: f"{v * 100:.0f}% of students",
        "lesson": "Many of the most famous colleges enroll few low-income students. Access is "
                  "something rankings rarely reward.",
    },
}
# 10 rounds: price twice (it's the biggest surprise), everything else once or twice
SCHEDULE = ["price", "grad", "pell_grad", "beats", "earnings", "debt", "pell_share",
            "price", "beats", "pell_grad"]

# Title you earn at the end, by score
TITLES = [(10, "🏆 Myth Buster", "Perfect. You see right through the rankings."),
          (7, "🔍 Reality Checker", "You know what really matters."),
          (4, "🕵️ Data Detective", "Getting there. Some of these fool almost everyone."),
          (0, "🎓 Rankings Rookie", "Don't worry: most people guess the famous school every time.")]

GAME_CSS = """
<style>
.game-q {font-size: 1.9rem; font-weight: 700; line-height: 1.25; margin: 0.6rem 0 1rem;}
.game-school {font-size: 1.4rem; font-weight: 700;}
.game-big {font-size: 2.2rem; font-weight: 800;}
div[data-testid="stButton"] button p {font-size: 1.15rem;}
</style>
"""


# ---------------------------------------------------------------------------
# Building the rounds
# ---------------------------------------------------------------------------
def pick_pair(df, q, rng, surprise):
    """One famous school (A) and one other (B) whose values differ enough.

    surprise=True: B (the less famous one) must be the right answer.
    """
    col, hw, gap = q["col"], q["higher_wins"], q["min_gap"]
    d = df.dropna(subset=[col])
    famous = d[d["prestige_rank"] <= FAMOUS_TOP]
    others = d[d["prestige_rank"] > FAMOUS_TOP]
    for _ in range(500):
        a = famous.iloc[rng.integers(len(famous))]
        b = others.iloc[rng.integers(len(others))]
        diff = b[col] - a[col]
        if abs(diff) < gap:
            continue
        b_wins = (diff > 0) == hw
        if surprise and not b_wins:
            continue
        return a, b
    return None


def new_game(schools, income):
    """Pick 10 rounds of (question, school A, school B), in random left/right order."""
    rng = np.random.default_rng()
    df = scoring.personalize(schools, income=income)
    df["prestige_rank"] = df["prestige_score"].rank(ascending=False)
    rounds, used = [], set()
    for key in SCHEDULE:
        for _ in range(20):  # avoid showing the same school twice
            pair = pick_pair(df, QUESTIONS[key], rng, surprise=rng.random() < SURPRISE_SHARE)
            if pair and not {pair[0]["unitid"], pair[1]["unitid"]} & used:
                break
        pair = pair or pick_pair(df, QUESTIONS[key], rng, surprise=False)
        a, b = pair
        used |= {a["unitid"], b["unitid"]}
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        rounds.append({"key": key, "schools": [left.to_dict(), right.to_dict()]})
    return {"rounds": rounds, "i": 0, "score": 0, "streak": 0, "best": 0,
            "picked": None, "history": [], "income": income}


def winner_index(rnd):
    q = QUESTIONS[rnd["key"]]
    v = [s[q["col"]] for s in rnd["schools"]]
    better = v[0] > v[1] if q["higher_wins"] else v[0] < v[1]
    return 0 if better else 1


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------
def scoreboard(g):
    a, b, c = st.columns(3)
    a.metric("Score", f"{g['score']} / {N_ROUNDS}")
    b.metric("Streak", f"🔥 {g['streak']}" if g["streak"] else "0")
    c.metric("Best streak", g["best"])


def school_card(col, s, idx, g, rnd):
    """One school with its hints, plus the pick button (or the reveal)."""
    q = QUESTIONS[rnd["key"]]
    picked, win = g["picked"], winner_index(rnd)
    with col, st.container(border=True):
        st.markdown(f"<div class='game-school'>{s['name']}</div>", unsafe_allow_html=True)
        famous = "⭐ Famous name · " if s["prestige_rank"] <= FAMOUS_TOP else ""
        admit = "" if pd.isna(s["admit_rate"]) else f" · admits {s['admit_rate'] * 100:.0f}%"
        st.caption(f"{famous}{s['state']} · {s['control_label']}{admit}")
        if picked is None:
            if st.button("This one", key=f"pick-{g['i']}-{idx}", type="primary", width="stretch"):
                g["picked"] = idx
                correct = idx == win
                g["score"] += correct
                g["streak"] = g["streak"] + 1 if correct else 0
                g["best"] = max(g["best"], g["streak"])
                g["history"].append(correct)
                st.rerun()
        else:
            mark = "✅" if idx == win else ""
            st.markdown(f"<div class='game-big'>{q['fmt'](s[q['col']])} {mark}</div>",
                        unsafe_allow_html=True)
            if idx == picked:
                st.caption("Your pick")


def play_round(g):
    rnd = g["rounds"][g["i"]]
    q = QUESTIONS[rnd["key"]]
    st.progress((g["i"] + (g["picked"] is not None)) / N_ROUNDS,
                text=f"Round {g['i'] + 1} of {N_ROUNDS}")
    ask = q["ask"].format(bracket=scoring.INCOME_BRACKETS[g["income"]])
    st.markdown(f"<div class='game-q'>{ask}</div>", unsafe_allow_html=True)

    cols = st.columns(2)
    for idx, s in enumerate(rnd["schools"]):
        school_card(cols[idx], s, idx, g, rnd)

    if g["picked"] is not None:
        if g["picked"] == winner_index(rnd):
            st.success(f"**Correct!** {q['lesson']}")
        else:
            st.error(f"**Not quite.** {q['lesson']}")
        last = g["i"] == N_ROUNDS - 1
        if st.button("See my results →" if last else "Next round →", type="primary"):
            g["i"] += 1
            g["picked"] = None
            st.rerun()


def finish(g, go_to):
    title, line = next((t, l) for cutoff, t, l in TITLES if g["score"] >= cutoff)
    if g["score"] == N_ROUNDS:
        st.balloons()
    st.markdown(f"<div class='game-q' style='font-size:2.6rem'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"### You scored {g['score']} out of {N_ROUNDS}")
    st.write(line)
    st.write("Rounds: " + " ".join("✅" if c else "❌" for c in g["history"])
             + f" · Best streak: {g['best']}")
    st.info("**The big idea:** a famous name tells you who a college admits, not what it will "
            "cost you or how much it will help you. Now find colleges that fit *you*.")
    a, b = st.columns(2)
    if a.button("Find my colleges →", type="primary", width="stretch"):
        st.session_state["quiz_step"] = 0
        go_to("quiz")
        st.rerun()
    if b.button("Play again", width="stretch"):
        st.session_state.pop("game")
        st.rerun()


def render(schools, go_to):
    st.markdown(GAME_CSS, unsafe_allow_html=True)
    st.title("🎮 Rankings or Reality?")

    # Price questions use the quiz's income bracket if the student took it
    answers = st.session_state.get("answers") or {}
    income = {v: k for k, v in scoring.INCOME_BRACKETS.items()}.get(answers.get("income")) or 1

    if "game" not in st.session_state:
        st.write(
            f"Two real colleges, one question, {N_ROUNDS} rounds. Can you tell what a famous "
            "name really tells you? Every number comes from the U.S. Department of Education."
        )
        if st.button("Start playing", type="primary"):
            st.session_state["game"] = new_game(schools, income)
            st.rerun()
        return

    g = st.session_state["game"]
    scoreboard(g)
    if g["i"] >= N_ROUNDS:
        finish(g, go_to)
    else:
        play_round(g)
