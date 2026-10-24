"""
AI advisor: a Claude-powered chat that can replace the quiz, then stays in the
results sidebar to answer questions about the student's matches.

How it works (a small tool-use loop):
  - The student types; we send the conversation to Claude with 3 tools.
  - save_answers   fills in the same answers the quiz collects (and updates them later)
  - get_my_matches returns the student's current top matches from scoring.py
  - lookup_school  returns the numbers for any college by name
  - Claude only talks about numbers it got from a tool, so it can't invent data.

Needs an Anthropic API key: ANTHROPIC_API_KEY in .streamlit/secrets.toml (local,
never commit it) or in the app's Secrets on Streamlit Cloud. Without a key the
quiz still works and the advisor shows how to turn it on.

Privacy: unlike the quiz, chat messages (and the answers the advisor sees) are
sent to Anthropic's API. The page says so before anyone types.
"""

import difflib
import json
import os

import anthropic
import pandas as pd
import streamlit as st

import quiz
import scoring

MODEL = "claude-opus-5-5"
MAX_TOOL_ROUNDS = 8  # safety stop for one student message

SYSTEM = f"""You are a friendly college advisor inside "Rankings vs. Reality", a free app that
helps U.S. high-school students find four-year colleges using federal College Scorecard data.
Many users are first-generation or from low-income families. Write in plain, warm English a
15-year-old understands. Keep replies short: 2-4 sentences, or a short list.

Two phases:
1. Getting to know the student (no answers saved yet). Ask one or two easy questions at a
   time to learn: intended major, rough family income bracket, SAT or ACT score (or none),
   GPA (optional), the most the family could pay per year after aid, region, school size,
   setting, and up to 2 priorities. If the student is unsure or wants to skip something, use
   the neutral option ("Not sure", "Anywhere", "Any size", "Any", "No score", no limit =
   90000). As soon as you know enough (or the student asks to see colleges), call
   save_answers, then tell them their matches are ready on the page.
2. After answers are saved, the student sees match cards and you sit in a sidebar. Answer
   questions with get_my_matches and lookup_school. If they want to change something
   ("cheaper", "only the Midwest", "I got a 1400"), call get_my_matches to see their current
   answers, then save_answers with the full updated set.

Rules:
- Every number you mention (price, graduation rate, earnings, debt, admit rate, scores)
  must come from a tool result in this conversation. Never guess or use outside knowledge
  for numbers. If a tool has no data, say so.
- "Likely / Match / Reach" comes only from SAT/ACT ranges and admit rates. The data has no
  GPA, essays or activities, so never claim to know someone's chances of getting in.
- Match scores and "beats expectations" are estimates from public data, not advice. Now and
  then suggest checking each college's net price calculator and talking to a counselor.
- Don't ask for the student's name, school, address, email or any contact details. If they
  share personal details, don't repeat them back.
- Stay on college choice. Politely decline unrelated requests.
- Income bracket labels: {", ".join(scoring.INCOME_BRACKETS.values())}.
"""

_SCHOOL_PROPS = {
    "major": {"type": "string", "enum": list(scoring.MAJORS)},
    "income": {"type": "string", "enum": list(scoring.INCOME_BRACKETS.values()) + ["Not sure"]},
    "test": {"type": "string", "enum": ["SAT", "ACT", "No score"]},
    "test_score": {"type": ["integer", "null"],
                   "description": "Total SAT (400-1600) or ACT composite (1-36); null if no score"},
    "gpa": {"type": ["number", "null"], "description": "Unweighted GPA out of 4.0, or null"},
    "max_price": {"type": "integer",
                  "description": "Most the family could pay per year after aid, in dollars. "
                                 "90000 means no limit."},
    "region": {"type": "string", "enum": ["Anywhere"] + list(scoring.REGIONS)},
    "size": {"type": "string", "enum": ["Any size"] + list(scoring.SIZES)},
    "setting": {"type": "string", "enum": ["Any", "City", "Suburb", "Town / rural"]},
    "priorities": {"type": "array", "items": {"type": "string",
                                              "enum": list(quiz.PRIORITY_FACTORS)},
                   "description": "Up to 2. They count double in the match score."},
}

TOOLS = [
    {
        "name": "save_answers",
        "description": "Save (or replace) the student's full set of answers. The app then "
                       "shows their match cards. Always send every field.",
        "strict": True,
        "input_schema": {"type": "object", "properties": _SCHOOL_PROPS,
                         "required": list(_SCHOOL_PROPS), "additionalProperties": False},
    },
    {
        "name": "get_my_matches",
        "description": "The student's saved answers, how many colleges fit them, and their "
                       "top matches with price, graduation, earnings, debt and admission label.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [],
                         "additionalProperties": False},
    },
    {
        "name": "lookup_school",
        "description": "Look up any of the 1,405 colleges by name (partial names are fine). "
                       "Returns up to 3 best matches with their numbers for this student.",
        "strict": True,
        "input_schema": {"type": "object",
                         "properties": {"name": {"type": "string"}},
                         "required": ["name"], "additionalProperties": False},
    },
]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
def api_key():
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # no secrets.toml at all
        key = None
    return key or os.environ.get("ANTHROPIC_API_KEY")


@st.cache_resource
def get_client(key):
    return anthropic.Anthropic(api_key=key)


def available():
    return bool(api_key())


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def current_table(schools, fos):
    """All schools scored for the student's current answers (same as results.py)."""
    import results  # imported here: results.py imports this file too

    answers = {**quiz.DEFAULTS, "max_price": quiz.NO_LIMIT, **st.session_state.get("answers", {})}
    return results.build(schools, fos, quiz.answers_to_profile(answers))


def school_facts(row):
    """Plain-English numbers for one school, rounded so they read well."""
    def num(v, digits=0):
        return None if pd.isna(v) else round(float(v), digits)

    return {
        "name": row["name"], "state": row["state"],
        "city": row["city"] if isinstance(row.get("city"), str) else None,
        "type": row["control_label"], "undergrads": num(row["undergrads"]),
        "match_score_out_of_100": num(row["match"]),
        "fits_your_answers": bool(row["fits"]), "why_it_doesnt_fit": list(row["reasons"]),
        "price_per_year_for_you": num(row["your_price"]), "price_basis": row["price_source"],
        "sticker_price": num(row["sticker_price"]),
        "graduation_rate_pct": num(row["grad_rate"] * 100),
        "predicted_graduation_rate_pct": num(row["predicted_grad_rate"] * 100),
        "low_income_graduation_rate_pct": num(row["pell_grad_rate"] * 100),
        "typical_earnings": num(row["your_earnings"]), "earnings_basis": row["earn_source"],
        "median_debt": num(row["your_debt"]), "debt_basis": row["debt_source"],
        "admit_rate_pct": num(row["admit_rate"] * 100),
        "admission_label": row.get("admit_fit"), "admission_note": row.get("admit_note"),
        "beats_expectations_0_to_100": num(row["value_added"]),
        "pct_of_degrees_in_your_major": num(row["major_share"] * 100, 1),
    }


def run_tool(name, args, schools, fos):
    """Run one tool call and return (result text, whether answers were saved)."""
    if name == "save_answers":
        a = dict(args)
        a["priorities"] = list(dict.fromkeys(a["priorities"]))[:2]
        a["max_price"] = max(0, min(int(a["max_price"]), quiz.NO_LIMIT))
        if a["test"] == "No score" or a["test_score"] is None:
            a["test"], a["test_score"] = "No score", None
        else:
            lo, hi = (400, 1600) if a["test"] == "SAT" else (1, 36)
            a["test_score"] = max(lo, min(int(a["test_score"]), hi))
        if a["gpa"] is not None:
            a["gpa"] = max(0.0, min(float(a["gpa"]), 4.0))
        st.session_state["answers"] = {**quiz.DEFAULTS, **a}
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
            "top_matches": [school_facts(r) for _, r in fit.head(8).iterrows()],
        }), False

    if name == "lookup_school":
        query = args["name"].lower()
        names = t["name"].str.lower()
        hits = t[names.str.contains(query, regex=False)]
        if hits.empty:
            close = difflib.get_close_matches(query, names.tolist(), n=3, cutoff=0.5)
            hits = t[names.isin(close)]
        if hits.empty:
            return json.dumps({"found": 0, "note": "No college with that name in our data."}), False
        return json.dumps({"found": int(len(hits)),
                           "schools": [school_facts(r) for _, r in hits.head(3).iterrows()]}), False

    return json.dumps({"error": f"Unknown tool {name}"}), False


# ---------------------------------------------------------------------------
# The conversation loop
# ---------------------------------------------------------------------------
def send(user_text, schools, fos):
    """Add the student's message, let Claude answer (calling tools as needed).

    Returns True if the advisor saved answers during this turn.
    The history is append-only: we never edit earlier messages.
    """
    msgs = st.session_state.setdefault("chat", [])
    msgs.append({"role": "user", "content": user_text})
    client, saved = get_client(api_key()), False

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            resp = client.beta.messages.create(
                model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=msgs,
                output_config={"effort": "low"},  # quick, chatty replies
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                cache_control={"type": "ephemeral"},
            )
        except anthropic.AuthenticationError:
            st.session_state["chat_error"] = "The API key was rejected. Check ANTHROPIC_API_KEY."
            break
        except anthropic.RateLimitError:
            st.session_state["chat_error"] = "The advisor is busy. Try again in a minute."
            break
        except anthropic.APIStatusError as e:
            st.session_state["chat_error"] = f"The advisor hit an error ({e.status_code}). Try again."
            break
        except anthropic.APIConnectionError:
            st.session_state["chat_error"] = "Couldn't reach the advisor. Check the internet connection."
            break

        if resp.stop_reason == "refusal":
            st.session_state["chat_error"] = "The advisor can't help with that one. Try asking another way."
            break
        msgs.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                text, did_save = run_tool(block.name, block.input, schools, fos)
                saved |= did_save
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": text})
        msgs.append({"role": "user", "content": results})
    return saved


# ---------------------------------------------------------------------------
# Drawing the chat
# ---------------------------------------------------------------------------
PRIVACY = ("🔒 Messages you type here, and the answers the advisor works with, are sent to "
           "Anthropic's Claude AI to write replies. They aren't saved by this app. Don't share "
           "your name or contact details. The quiz never sends anything.")


def show_history():
    """Student messages and the advisor's text; tool calls appear as small notes."""
    for m in st.session_state.get("chat", []):
        if m["role"] == "user" and isinstance(m["content"], str):
            st.chat_message("user").write(m["content"])
        elif m["role"] == "assistant":
            with st.chat_message("assistant", avatar="🎓"):
                for block in m["content"]:
                    if block.type == "text":
                        st.markdown(block.text.replace("$", "\\$"))
                    elif block.type == "tool_use" and block.name == "save_answers":
                        st.caption("✓ Updated your answers")
    if st.session_state.get("chat_error"):
        st.error(st.session_state.pop("chat_error"))


def no_key_message():
    st.info("The AI advisor needs an Anthropic API key. Add `ANTHROPIC_API_KEY = \"...\"` to "
            "`.streamlit/secrets.toml` (locally) or to the app's **Secrets** on Streamlit "
            "Cloud, then reload. The quiz works without it.")


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
            with st.spinner("Thinking..."):
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
    st.caption("Ask about any college, or say things like \"show me cheaper ones\". "
               + PRIVACY)
    with st.container(height=380):
        if not st.session_state.get("chat"):
            st.caption("Try: \"Why is #1 my top match?\" or \"How does BU compare?\"")
        show_history()
    if prompt := st.chat_input("Ask about your matches..."):
        with st.spinner("Thinking..."):
            send(prompt, schools, fos)
        st.rerun()
