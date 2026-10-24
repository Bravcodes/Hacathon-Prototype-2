"""
Landing page: a 7-question quiz for high-school students.

One question per screen, big buttons, Back / Next. Answers live only in
st.session_state (this browser tab's memory): nothing is saved or logged.

answers_to_profile() turns the answers into filters + weights for results.py.
"""

import streamlit as st

import scoring

# ---------------------------------------------------------------------------
# The questions. "kind" picks the widget:
#   one    = pick one option (big buttons)
#   many   = pick up to `max` options
#   slider = dollar slider
#   scores = GPA box + SAT/ACT choice + score box
# ---------------------------------------------------------------------------
NO_LIMIT = 90000  # slider's top value means "no limit"

QUESTIONS = [
    {"key": "major", "kind": "one", "title": "What do you want to study?",
     "options": list(scoring.MAJORS)},
    {"key": "scores", "kind": "scores", "title": "How are your grades and test scores?",
     "note": "🔒 Private, like everything here. We use your SAT or ACT to label colleges "
             "Likely, Match or Reach. Skipping the test is fine: many colleges are test-optional."},
    {"key": "income", "kind": "one", "title": "About how much does your family earn per year?",
     "note": "🔒 This stays private: it's never saved or sent anywhere. We only use it to "
             "look up what families like yours actually pay at each college.",
     "options": list(scoring.INCOME_BRACKETS.values()) + ["Not sure"]},
    {"key": "max_price", "kind": "slider", "title": "What's the most your family could pay per year?",
     "note": "After grants and scholarships (the \"net price\"), not the sticker price."},
    {"key": "region", "kind": "one", "title": "Where do you want to go to school?",
     "options": ["Anywhere"] + list(scoring.REGIONS)},
    {"key": "size", "kind": "one", "title": "What size school?",
     "note": "Small = under 5,000 undergrads · Medium = 5,000–15,000 · Large = 15,000+",
     "options": ["Any size"] + list(scoring.SIZES)},
    {"key": "setting", "kind": "one", "title": "What kind of place?",
     "options": ["Any", "City", "Suburb", "Town / rural"]},
    {"key": "priorities", "kind": "many", "max": 2, "title": "What matters most? Pick up to 2.",
     "note": "Your picks count double in your match score.",
     "options": ["Graduate on time", "Earn well", "Low debt", "Famous name",
                 "Helps students like me succeed"]},
]

DEFAULTS = {"major": None, "gpa": None, "test": None, "test_score": None, "income": None,
            "max_price": 35000, "region": None, "size": None, "setting": None, "priorities": []}

# Test choices -> (score box label, min, max, step, starting value)
TESTS = {"SAT": ("Total SAT score", 400, 1600, 10, 1200),
         "ACT": ("ACT composite score", 1, 36, 1, 24),
         "No score": None}

# Quiz priority -> scoring factor
PRIORITY_FACTORS = {
    "Graduate on time": "graduation",
    "Earn well": "earnings",
    "Low debt": "low_debt",
    "Famous name": "prestige",
    "Helps students like me succeed": "value_added",
}
BASE_WEIGHTS = {"affordability": 1, "graduation": 1, "earnings": 1, "low_debt": 1,
                "prestige": 1, "value_added": 1, "mobility": 0}

# Big text and big buttons. Injected only while the quiz is on screen.
QUIZ_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none;}
.quiz-title {font-size: 2.4rem; font-weight: 700; line-height: 1.2; margin: 1.5rem 0 0.5rem;}
.quiz-note {font-size: 1.15rem; color: #52514e; margin-bottom: 1rem;}
div[data-testid="stButton"] button {min-height: 3.6rem; border-radius: 0.8rem;}
div[data-testid="stButton"] button p {font-size: 1.25rem;}
</style>
"""


def answers_to_profile(a):
    """Quiz answers -> what scoring.py needs (filters + weights)."""
    income = {v: k for k, v in scoring.INCOME_BRACKETS.items()}.get(a["income"])
    weights = dict(BASE_WEIGHTS)
    for p in a["priorities"]:
        weights[PRIORITY_FACTORS[p]] *= 2
    return {
        "major": None if a["major"] in (None, "Undecided") else a["major"],
        "income": income,  # None = "Not sure" -> average net price
        "max_price": None if a["max_price"] >= NO_LIMIT else a["max_price"],
        "regions": None if a["region"] in (None, "Anywhere") else [a["region"]],
        "sizes": None if a["size"] in (None, "Any size") else [a["size"]],
        "settings": None if a["setting"] in (None, "Any") else [a["setting"]],
        "sat": a["test_score"] if a["test"] == "SAT" else None,
        "act": a["test_score"] if a["test"] == "ACT" else None,
        "gpa": a["gpa"],  # shown back to the student only: the Scorecard has no GPA data
        "weights": weights,
    }


def init_state():
    st.session_state.setdefault("quiz_step", 0)
    st.session_state["answers"] = {**DEFAULTS, **st.session_state.get("answers", {})}


def option_buttons(q, answers):
    """Big buttons in two columns. The selected one is highlighted."""
    cols = st.columns(2)
    for i, option in enumerate(q["options"]):
        if q["kind"] == "one":
            selected = answers[q["key"]] == option
        else:
            selected = option in answers[q["key"]]
        label = ("✓ " if selected else "") + option
        if cols[i % 2].button(label, key=f"q-{q['key']}-{i}", width="stretch",
                              type="primary" if selected else "secondary"):
            if q["kind"] == "one":
                answers[q["key"]] = option
            else:
                picked = answers[q["key"]]
                if selected:
                    picked.remove(option)
                elif len(picked) < q["max"]:
                    picked.append(option)
                else:
                    st.session_state["quiz_warning"] = f"You can pick up to {q['max']}."
            st.rerun()


def price_slider(q, answers):
    value = st.slider("Max per year ($)", 0, NO_LIMIT, answers["max_price"], step=1000,
                      label_visibility="collapsed", format="$%d")
    answers["max_price"] = value
    shown = "No limit" if value >= NO_LIMIT else f"\\${value:,.0f} per year"
    st.markdown(f"<div class='quiz-title' style='text-align:center'>{shown}</div>",
                unsafe_allow_html=True)


def score_inputs(answers):
    """GPA (optional), then pick SAT / ACT / no score, then type the score."""
    gpa = st.number_input("GPA (optional, unweighted, out of 4.0)", min_value=0.0, max_value=4.0,
                          value=answers["gpa"], step=0.1, format="%.1f", placeholder="e.g. 3.4")
    answers["gpa"] = gpa
    st.caption("Colleges don't report GPAs to the federal data we use, so your GPA isn't part "
               "of the Likely / Match / Reach label. Use it with your counselor.")

    st.markdown("**Which test score do you have?**")
    labels = {"SAT": "SAT", "ACT": "ACT", "No score": "Haven't taken / not sending"}
    for col, (test, label) in zip(st.columns(3), labels.items()):
        selected = answers["test"] == test
        if col.button(("✓ " if selected else "") + label, key=f"q-test-{test}", width="stretch",
                      type="primary" if selected else "secondary"):
            if answers["test"] != test:
                answers["test"], answers["test_score"] = test, None
            st.rerun()

    spec = TESTS.get(answers["test"])
    if spec:
        label, lo, hi, step, start = spec
        answers["test_score"] = int(st.number_input(
            label, min_value=lo, max_value=hi, step=step,
            value=answers["test_score"] or start, key=f"score-{answers['test']}"))


def is_answered(q, answers):
    if q["kind"] == "one":
        return answers[q["key"]] is not None
    if q["kind"] == "scores":
        return answers["test"] is not None
    return True  # slider always has a value; priorities are optional


def render(go_to):
    """Draw the current question. go_to(page) switches pages in app.py."""
    init_state()
    st.markdown(QUIZ_CSS, unsafe_allow_html=True)
    step, answers = st.session_state["quiz_step"], st.session_state["answers"]
    q = QUESTIONS[step]

    _, mid, _ = st.columns([1, 4, 1])
    with mid:
        st.progress((step + 1) / len(QUESTIONS), text=f"Question {step + 1} of {len(QUESTIONS)}")
        st.markdown(f"<div class='quiz-title'>{q['title']}</div>", unsafe_allow_html=True)
        if q.get("note"):
            st.markdown(f"<div class='quiz-note'>{q['note']}</div>", unsafe_allow_html=True)

        if q["kind"] == "slider":
            price_slider(q, answers)
        elif q["kind"] == "scores":
            score_inputs(answers)
        else:
            option_buttons(q, answers)
        if st.session_state.pop("quiz_warning", None):
            st.warning(f"You can pick up to {q['max']}. Tap one to remove it first.")

        st.write("")
        back, _, nxt = st.columns([1, 1, 1])
        if step > 0 and back.button("← Back", width="stretch"):
            st.session_state["quiz_step"] -= 1
            st.rerun()
        last = step == len(QUESTIONS) - 1
        if nxt.button("See my matches →" if last else "Next →", type="primary",
                      width="stretch", disabled=not is_answered(q, answers)):
            if last:
                st.session_state["quiz_done"] = True
                go_to("results")
            else:
                st.session_state["quiz_step"] += 1
            st.rerun()

        if step == 0:
            st.caption("Just want to look around? Use **Explore the data** at the top.")
