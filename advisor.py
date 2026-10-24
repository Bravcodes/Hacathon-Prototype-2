"""
AI advisor: a chat that can replace the quiz, then stays in the results sidebar
to answer questions about the student's matches. Runs on Groq.

How it works (a small tool-use loop):
  - The student types; we send the conversation to the model with 3 tools.
  - save_answers   fills in the same answers the quiz collects (and updates them later)
  - get_my_matches returns the student's current top matches from scoring.py
  - lookup_school  returns the numbers for any college by name
  - Context the model always gets:
      background_context()  how the app works, what each number means, key findings
      student_context()     the student's current answers and top 5 matches (live)
  - The model may only quote numbers from that context or a tool result.

Needs a Groq API key: GROQ_API_KEY in .streamlit/secrets.toml (local, git-ignored)
or in the app's Secrets on Streamlit Cloud. Without a key the quiz still works.

Privacy: unlike the quiz, chat messages (and the answers the advisor sees) are
sent to Groq's API. The page says so before anyone types.
"""

import difflib
import json
import os
import re
import time

import groq
import pandas as pd
import streamlit as st

import quiz
import scoring

# Groq's free tier allows 8,000 tokens per minute per model, so we keep every
# request small (~2,500 tokens) and wait briefly when the limit is hit.
# (We tried falling back to openai/gpt-oss-20b: it invented numbers, so we don't.)
MODELS = ["openai/gpt-oss-120b"]
MAX_TOOL_ROUNDS = 5   # safety stop for one student message
MAX_WAIT = 45         # total seconds we will wait for the per-minute limit to reset
HISTORY_MESSAGES = 12  # only the most recent messages are sent

# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
RULES = f"""You are a friendly college advisor inside "Rankings vs. Reality", a free app that
helps U.S. high-school students find four-year colleges. Many users are first-generation or
from low-income families. Write in plain, warm English a 15-year-old understands. You appear in a
narrow chat panel: at most 80 words unless the student asks for detail, no tables, no headings,
at most 3 short bullet points.

Two phases:
1. Getting to know the student (answers not saved yet). Ask one or two easy questions at a
   time to learn: intended major, rough family income bracket, SAT or ACT score (or none),
   GPA (optional), the most the family could pay per year after aid, region, school size,
   setting, and up to 2 priorities. If the student is unsure or wants to skip something, use
   the neutral option ("Not sure", "Anywhere", "Any size", "Any", "No score", no limit =
   90000). As soon as you know enough (or the student asks to see colleges), call
   save_answers, then tell them their matches are ready on the page.
2. After answers are saved, the student sees match cards and you sit in a sidebar. Answer
   questions using the STUDENT CONTEXT and the tools. If they want to change something
   ("cheaper", "only the Midwest", "I got a 1400"), call save_answers with the full updated
   set (start from their current answers in the STUDENT CONTEXT).

Rules:
- Every number you mention (price, graduation rate, earnings, debt, admit rate, scores) must
  come from the BACKGROUND, the STUDENT CONTEXT, or a tool result. If you need a number you
  don't have, call lookup_school FIRST. Never guess, never use outside knowledge, and never say
  things like "elite schools usually..." If the tool has no data, say the data doesn't have it.
- To explain why a college ranks where it does, use its points_by_factor (the points each
  factor added to its match score).
- When you quote earnings or debt, say what they cover using earnings_basis / debt_basis
  (e.g. "all majors, 10 years after starting" is NOT the same as "CS majors").
- "Likely / Match / Reach" comes only from SAT/ACT ranges and admit rates. The data has no
  GPA, essays or activities, so never claim to know someone's chances of getting in.
- Match scores and "beats expectations" are estimates from public data, not advice. Now and
  then suggest each college's net price calculator and talking to a counselor.
- Don't ask for the student's name, school, address, email or contact details. If they share
  personal details, don't repeat them back.
- Stay on college choice. Politely decline unrelated requests.
- Valid answer options: majors {list(scoring.MAJORS)}; income brackets
  {list(scoring.INCOME_BRACKETS.values()) + ["Not sure"]}; regions
  {["Anywhere"] + list(scoring.REGIONS)}; sizes {["Any size"] + list(scoring.SIZES)};
  settings ["Any", "City", "Suburb", "Town / rural"]; priorities {list(quiz.PRIORITY_FACTORS)}.
"""


@st.cache_data
def background_context(schools):
    """How the app works and what the data says, with numbers computed from our data."""
    n = len(schools)
    rho_raw = schools["prestige_score"].corr(schools["grad_rate"], method="spearman")
    rho_va = schools["prestige_score"].corr(schools["residual_grad_rate"], method="spearman")
    over = schools.nlargest(3, "residual_grad_rate")
    over_txt = "; ".join(f"{r['name']} ({r['grad_rate'] * 100:.0f}% graduate vs. "
                         f"{r['predicted_grad_rate'] * 100:.0f}% predicted)"
                         for _, r in over.iterrows())
    acc = ""
    path = scoring.V2_DATA / "model_scores.csv"
    if path.exists():
        s = pd.read_csv(path).set_index(["target", "model"])
        acc = (f"On colleges it never saw, the model's graduation-rate predictions are off by "
               f"{s.loc[('grad_rate', 'forest'), 'mae'] * 100:.0f} points on average "
               f"(R² {s.loc[('grad_rate', 'forest'), 'r2']:.2f}); earnings by "
               f"${s.loc[('earnings_10yr', 'forest'), 'mae']:,.0f} (R² "
               f"{s.loc[('earnings_10yr', 'forest'), 'r2']:.2f}).")
    bu = schools[schools["unitid"] == 164988]
    bu_txt = ""
    if len(bu):
        b = bu.iloc[0]
        bu_txt = (f"Boston University (the host school): {b['grad_rate'] * 100:.0f}% graduate "
                  f"(predicted {b['predicted_grad_rate'] * 100:.0f}%), admits "
                  f"{b['admit_rate'] * 100:.0f}%, average net price ${b['net_price']:,.0f}, "
                  f"median earnings ${b['earnings_10yr']:,.0f}.")

    return f"""BACKGROUND (how this app works; use it to explain things):
- Data: U.S. Dept. of Education College Scorecard; {n:,} public and private nonprofit
  four-year colleges. No commercial rankings.
- Match score (0-100): each factor (affordability, earnings, graduation, low debt, famous name,
  beats expectations) is scored 0-100 vs. all {n:,} colleges; the 2 priorities count double;
  points add up to the score. Filters choose which colleges show but never change a score.
- "Your price": average net price for families in the student's income bracket. Sticker
  price = full list price. Earnings: the student's major (5 yrs after graduating) when
  available, else the college overall (10 yrs after starting).
- "Beats expectations": a model predicts graduation and earnings from who a college enrolls
  (% low-income, % first-gen, SAT, admit rate) and its resources. Actual minus predicted. {acc}
- Likely / Match / Reach: the student's SAT/ACT vs. the middle 50% of admitted students
  (below = Reach, inside = Match, above = Likely). Admit rate under 15% = Reach for everyone.
  With a score, "Academic fit" counts double: 100 when the score is inside the college's
  middle 50% (or a bit above), lower when far below or far above ("undermatching"). For
  colleges that report no scores the typical SAT is estimated by a model (about 50 points off on
  average). Not a fit if the score is >100 SAT (3 ACT) points below the 25th percentile, or
  under 1400 SAT / 31 ACT when the college admits under 20%. "Balanced list" in the sidebar
  shows a mix of about 3 Reach, 3 Match, 2 Likely.
- Key findings: prestige goes with raw graduation rates (correlation {rho_raw:.2f}) but barely
  with beating expectations ({rho_va:.2f}), so prestige mostly reflects who a college admits.
  Biggest graduation over-performers: {over_txt}.
- {bu_txt}
- Limits: earnings cover only federal-aid students; value added is noisy for small colleges;
  correlation is not causation; small colleges are under-represented (privacy suppression);
  nothing here measures campus life, safety or a specific program.
"""


# ---------------------------------------------------------------------------
# Client and tools
# ---------------------------------------------------------------------------
KEY_NAMES = ("groq_api_key", "groq_key", "groq")


def secret_names():
    """Names (never values) of the secrets this app can see, for the help message."""
    try:
        return list(st.secrets.keys())
    except Exception:  # no secrets file / no Secrets set on Streamlit Cloud
        return []


def api_key():
    """The Groq key from Streamlit secrets or the environment. Forgiving about
    the exact name (GROQ_API_KEY, groq_api_key, GROQ_KEY) and also accepts a
    [groq] section with api_key = "..."."""
    key = None
    for name in secret_names():
        value = st.secrets[name]
        if isinstance(value, str) and name.lower() in KEY_NAMES:
            key = value
        elif hasattr(value, "get") and name.lower() == "groq":
            key = value.get("api_key") or value.get("GROQ_API_KEY")
    key = key or os.environ.get("GROQ_API_KEY")
    return key.strip().strip('"') if key else None


@st.cache_resource
def get_client(key):
    return groq.Groq(api_key=key, max_retries=0, timeout=60)


def available():
    return bool(api_key())


ANSWER_OPTIONS = {
    "major": list(scoring.MAJORS),
    "income": list(scoring.INCOME_BRACKETS.values()) + ["Not sure"],
    "test": ["SAT", "ACT", "No score"],
    "region": ["Anywhere"] + list(scoring.REGIONS),
    "size": ["Any size"] + list(scoring.SIZES),
    "setting": ["Any", "City", "Suburb", "Town / rural"],
}
_ANSWER_PROPS = {
    **{k: {"type": "string", "enum": v} for k, v in ANSWER_OPTIONS.items()},
    "test_score": {"type": ["integer", "null"],
                   "description": "Total SAT (400-1600) or ACT composite (1-36); null if no score"},
    "gpa": {"type": ["number", "null"], "description": "Unweighted GPA out of 4.0, or null"},
    "max_price": {"type": "integer", "description": "Most the family could pay per year after "
                                                    "aid, in dollars. 90000 means no limit."},
    "priorities": {"type": "array", "items": {"type": "string", "enum": list(quiz.PRIORITY_FACTORS)},
                   "description": "Up to 2. They count double in the match score."},
}


def _tool(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOLS = [
    _tool("save_answers", "Save (or replace) the student's full set of answers. The app then "
          "shows their match cards. Always send every field.", _ANSWER_PROPS, list(_ANSWER_PROPS)),
    _tool("get_my_matches", "The student's saved answers, how many colleges fit them, and their "
          "top 5 matches with price, graduation, earnings, debt and admission label.", {}, []),
    _tool("lookup_school", "Look up any of the colleges by name (partial names are fine). Returns "
          "up to 2 best matches with their numbers for this student.",
          {"name": {"type": "string"}}, ["name"]),
]


def current_table(schools, fos):
    """All schools scored for the student's current answers (same as results.py)."""
    import results  # imported here: results.py imports this file too

    answers = {**quiz.DEFAULTS, "max_price": quiz.NO_LIMIT, **st.session_state.get("answers", {})}
    return results.build(schools, fos, quiz.answers_to_profile(answers))


def school_facts(row, weights=None):
    """Plain-English numbers for one school, rounded and without empty fields
    (short, because every token counts against Groq's per-minute limit)."""
    def num(v, digits=0):
        return None if pd.isna(v) else round(float(v), digits) if digits else int(round(float(v)))

    facts = {
        "name": row["name"], "state": row["state"], "type": row["control_label"],
        "match_score": num(row["match"]), "fits_answers": bool(row["fits"]),
        "doesnt_fit_because": list(row["reasons"]) or None,
        "your_price_per_yr": num(row["your_price"]), "price_basis": row["price_source"],
        "sticker_price": num(row["sticker_price"]),
        "grad_rate_pct": num(row["grad_rate"] * 100),
        "predicted_grad_rate_pct": num(row["predicted_grad_rate"] * 100),
        "low_income_grad_rate_pct": num(row["pell_grad_rate"] * 100),
        "earnings": num(row["your_earnings"]), "earnings_basis": row["earn_source"],
        "median_debt": num(row["your_debt"]), "debt_basis": row["debt_source"],
        "admit_rate_pct": num(row["admit_rate"] * 100),
        "admission": row.get("admit_fit"), "admission_note": row.get("admit_note") or None,
        "beats_expectations_0_100": num(row["value_added"]),
        "pct_degrees_in_major": num(row["major_share"] * 100, 1),
    }
    if weights:
        facts["points_by_factor"] = {
            scoring.FACTORS[k][2]: num(row[f"pts_{k}"], 1)
            for k in scoring.FACTORS if weights.get(k, 0) > 0}
    return {k: v for k, v in facts.items() if v is not None}


def current_weights():
    answers = {**quiz.DEFAULTS, "max_price": quiz.NO_LIMIT, **st.session_state.get("answers", {})}
    return quiz.answers_to_profile(answers)["weights"]


def student_context(schools, fos):
    """Live context: the student's answers and top 5 matches, rebuilt every message."""
    if not st.session_state.get("quiz_done"):
        return "STUDENT CONTEXT: no answers saved yet (phase 1)."
    t = current_table(schools, fos)
    fit = t[t["fits"]]
    top = [{k: s.get(k) for k in ("name", "state", "match_score", "your_price_per_yr",
                                  "grad_rate_pct", "earnings", "earnings_basis", "median_debt",
                                  "admission")}
           for s in (school_facts(r) for _, r in fit.head(5).iterrows())]
    weights = current_weights()
    for s, (_, r) in zip(top[:3], fit.head(3).iterrows()):
        s["points_by_factor"] = school_facts(r, weights)["points_by_factor"]
    return ("STUDENT CONTEXT (phase 2, the student is looking at their match cards):\n"
            + json.dumps({"answers": st.session_state.get("answers", {}),
                          "colleges_that_fit": int(len(fit)), "top_5_matches": top}))


def validate_answers(a):
    """Check save_answers input. Returns (clean answers, list of problems)."""
    problems = [f"{k} must be one of {opts}" for k, opts in ANSWER_OPTIONS.items()
                if a.get(k) not in opts]
    pri = [p for p in a.get("priorities") or [] if p in quiz.PRIORITY_FACTORS]
    clean = {**{k: a.get(k) for k in ANSWER_OPTIONS},
             "priorities": list(dict.fromkeys(pri))[:2]}
    try:
        clean["max_price"] = max(0, min(int(a.get("max_price") or quiz.NO_LIMIT), quiz.NO_LIMIT))
    except (TypeError, ValueError):
        problems.append("max_price must be a whole number of dollars")
    score = a.get("test_score")
    if clean["test"] in ("SAT", "ACT") and score is not None:
        lo, hi = (400, 1600) if clean["test"] == "SAT" else (1, 36)
        try:
            clean["test_score"] = max(lo, min(int(score), hi))
        except (TypeError, ValueError):
            problems.append("test_score must be a whole number")
    else:
        clean["test"], clean["test_score"] = "No score", None
    try:
        gpa = a.get("gpa")
        clean["gpa"] = None if gpa is None else max(0.0, min(float(gpa), 4.0))
    except (TypeError, ValueError):
        clean["gpa"] = None
    return clean, problems


def run_tool(name, args, schools, fos):
    """Run one tool call and return (result text, whether answers were saved)."""
    if name == "save_answers":
        clean, problems = validate_answers(args)
        if problems:
            return json.dumps({"saved": False, "fix_these": problems}), False
        st.session_state["answers"] = {**quiz.DEFAULTS, **clean}
        st.session_state["quiz_done"] = True
        fit = current_table(schools, fos)["fits"].sum()
        return json.dumps({"saved": True, "colleges_that_fit": int(fit)}), True

    t = current_table(schools, fos)
    if name == "get_my_matches":
        fit = t[t["fits"]]
        return json.dumps({
            "answers_saved": bool(st.session_state.get("quiz_done")),
            "answers": st.session_state.get("answers", {}),
            "colleges_that_fit": int(len(fit)),
            "top_matches": [school_facts(r, current_weights()) for _, r in fit.head(5).iterrows()],
        }), False

    if name == "lookup_school":
        query = str(args.get("name", "")).lower().strip()
        names = t["name"].str.lower()
        hits = t[names.str.contains(query, regex=False)] if query else t.iloc[0:0]
        if hits.empty:
            close = difflib.get_close_matches(query, names.tolist(), n=3, cutoff=0.5)
            hits = t[names.isin(close)]
        if hits.empty:
            return json.dumps({"found": 0, "note": "No college with that name in our data."}), False
        return json.dumps({"found": int(len(hits)),
                           "schools": [school_facts(r, current_weights())
                                       for _, r in hits.head(2).iterrows()]}), False

    return json.dumps({"error": f"Unknown tool {name}"}), False


# ---------------------------------------------------------------------------
# The conversation loop
# ---------------------------------------------------------------------------
def recent(msgs, turn_start):
    """The messages we actually send: the last few, starting at a student message
    (so a tool call is never split from its result). Tool results from earlier
    turns are shortened, since the STUDENT CONTEXT already has the current state."""
    start = max(0, min(turn_start, len(msgs) - HISTORY_MESSAGES))
    while start > 0 and msgs[start]["role"] != "user":
        start -= 1
    out = []
    for i, m in enumerate(msgs[start:], start):
        if m["role"] == "tool" and i < turn_start:
            m = {**m, "content": "(earlier lookup; ask again if needed)"}
        out.append(m)
    return out


def create(client, messages):
    """One model call. Over the per-minute limit: wait for the reset and retry,
    for up to MAX_WAIT seconds in total; after that, give up (the UI says why)."""
    waited = 0.0
    while True:
        try:
            return client.chat.completions.create(
                model=MODELS[0], messages=messages, tools=TOOLS, tool_choice="auto",
                max_completion_tokens=1500, reasoning_effort="low",
            )
        except groq.RateLimitError as e:
            wait = float(e.response.headers.get("retry-after") or 99) + 0.5
            if waited + wait > MAX_WAIT:
                raise
            time.sleep(wait)
            waited += wait


def send(user_text, schools, fos):
    """Add the student's message, let the model answer (calling tools as needed).

    st.session_state["chat"] holds the conversation (user, assistant and tool
    messages). The two context messages are rebuilt fresh for every request.
    Returns True if the advisor saved answers during this turn.
    """
    msgs = st.session_state.setdefault("chat", [])
    turn_start = len(msgs)
    msgs.append({"role": "user", "content": user_text})
    client, saved = get_client(api_key()), False

    for _ in range(MAX_TOOL_ROUNDS):
        context = [{"role": "system", "content": RULES + "\n" + background_context(schools)},
                   {"role": "system", "content": student_context(schools, fos)}]
        try:
            resp = create(client, context + recent(msgs, turn_start))
        except groq.AuthenticationError:
            st.session_state["chat_error"] = "The Groq API key was rejected. Check GROQ_API_KEY."
            break
        except groq.RateLimitError as e:
            wait = e.response.headers.get("retry-after")
            st.session_state["chat_error"] = ("The advisor is busy (free plan limit). Try again in "
                                              f"{wait or 'about 30'} seconds.")
            break
        except groq.APIStatusError as e:
            st.session_state["chat_error"] = f"The advisor hit an error ({e.status_code}). Try again."
            break
        except groq.APIConnectionError:
            st.session_state["chat_error"] = "Couldn't reach the advisor. Check the internet connection."
            break

        m = resp.choices[0].message
        calls = m.tool_calls or []
        # Keep only what the API accepts back (drop the model's private reasoning)
        msg = {"role": "assistant", "content": m.content or ""}
        if calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.function.name,
                                               "arguments": c.function.arguments}} for c in calls]
        msgs.append(msg)
        if not calls:
            break

        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
                text, did_save = run_tool(c.function.name, args, schools, fos)
            except json.JSONDecodeError:
                text, did_save = json.dumps({"error": "Arguments were not valid JSON."}), False
            saved |= did_save
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": text})
    return saved


# ---------------------------------------------------------------------------
# Drawing the chat
# ---------------------------------------------------------------------------
PRIVACY = ("🔒 Messages you type here, and the answers the advisor works with, are sent to an "
           "AI service (Groq) to write replies. They aren't saved by this app. Don't share your "
           "name or contact details. The quiz never sends anything.")


TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")


def tidy(text):
    """Drop markdown tables (too wide for the chat panel) and escape $ for Streamlit."""
    lines = [l for l in text.splitlines() if not TABLE_LINE.match(l)]
    return "\n".join(lines).replace("$", "\\$")


def show_history():
    """Student messages and the advisor's text; saves appear as small notes."""
    for m in st.session_state.get("chat", []):
        if m["role"] == "user":
            st.chat_message("user").write(m["content"])
        elif m["role"] == "assistant":
            saves = any(c["function"]["name"] == "save_answers" for c in m.get("tool_calls", []))
            if m["content"] or saves:
                with st.chat_message("assistant", avatar="🎓"):
                    if m["content"]:
                        st.markdown(tidy(m["content"]))
                    if saves:
                        st.caption("✓ Updated your answers")
    if st.session_state.get("chat_error"):
        st.error(st.session_state.pop("chat_error"))


def no_key_message():
    names = secret_names()
    seen = ", ".join(f"`{n}`" for n in names) if names else "none"
    st.info("The AI advisor needs a Groq API key. On Streamlit Cloud: **⋮ → Settings → "
            "Secrets**, paste one line `GROQ_API_KEY = \"gsk_...\"` (with the quotes), "
            "**Save**, then reload this page. Locally: put the same line in "
            "`.streamlit/secrets.toml`. The quiz works without it.\n\n"
            f"Secrets this app can see right now: {seen}")


def chat_page(schools, fos, go_to):
    """Full-page chat that replaces the quiz."""
    _, mid, _ = st.columns([1, 4, 1])
    with mid:
        st.title("💬 Chat with an AI advisor")
        st.caption(PRIVACY)
        if not available():
            no_key_message()
        elif not st.session_state.get("chat"):
            with st.chat_message("assistant", avatar="🎓"):
                st.markdown("Hi! I'll ask a few quick questions and then show you colleges that "
                            "fit. To start: what do you think you'd like to study? "
                            "(\"No idea yet\" is a great answer.)")
        show_history()
        if st.button("← Use the quiz instead"):
            st.session_state["quiz_step"] = 0
            go_to("quiz")
            st.rerun()

    if available() and (prompt := st.chat_input("Type your answer...")):
        with mid:
            st.chat_message("user").write(prompt)
            with st.spinner("Thinking... (on the free plan this can take up to a minute)"):
                saved = send(prompt, schools, fos)
        if saved:
            go_to("results")
        st.rerun()


def sidebar_chat(schools, fos):
    """The same conversation, in the results sidebar."""
    st.header("💬 Ask the AI advisor")
    if not available():
        st.caption("Off: no API key set.")
        return
    st.caption("Ask about any college, or say things like \"show me cheaper ones\". " + PRIVACY)
    with st.container(height=380):
        if not st.session_state.get("chat"):
            st.caption("Try: \"Why is #1 my top match?\" or \"How does BU compare?\"")
        show_history()
    if prompt := st.chat_input("Ask about your matches..."):
        with st.spinner("Thinking... (on the free plan this can take up to a minute)"):
            send(prompt, schools, fos)
        st.rerun()
