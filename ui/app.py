"""Streamlit chat interface for the Enterprise Agentic RAG API, with reasoning-step transparency.

Run:  streamlit run ui/app.py
Env:  RAG_API_URL (default http://localhost:8000), RAG_API_KEY (bearer token, if the API requires one)
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import httpx
import streamlit as st

DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("RAG_API_KEY", "")
REQUEST_TIMEOUT = 120.0

DATASET_FILTERS = {"All documents": None, "True data": {"dataset": "true"}, "Noisy data": {"dataset": "noisy"}}
NODE_ICONS = {
    "guardrails_input": ":material/shield:",
    "planner": ":material/route:",
    "retriever": ":material/search:",
    "responder": ":material/chat:",
    "guardrails_output": ":material/verified_user:",
}

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon=":material/neurology:", layout="centered")


# ---------- API client ----------


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def api_query(api_url: str, api_key: str, question: str, thread_id: str, filters: dict | None) -> dict[str, Any]:
    """Call POST /query. Returns the JSON body, or {"error": "..."} with a user-readable message."""
    payload: dict[str, Any] = {"question": question, "thread_id": thread_id}
    if filters:
        payload["filters"] = filters
    try:
        response = httpx.post(f"{api_url}/query", json=payload, headers=_headers(api_key), timeout=REQUEST_TIMEOUT)
    except httpx.TimeoutException:
        return {"error": "The request timed out. The service may be under load; please try again."}
    except httpx.HTTPError as exc:
        return {"error": f"Could not reach the API at {api_url} ({type(exc).__name__})."}

    if response.status_code == 200:
        return response.json()
    detail = _detail(response)
    if response.status_code == 401:
        return {"error": "Authentication failed: check the API key in the sidebar."}
    if response.status_code == 429:
        retry = response.headers.get("Retry-After", "a few")
        return {"error": f"Rate limit reached. Try again in {retry} seconds."}
    if response.status_code in (502, 503):
        return {"error": f"Service temporarily unavailable: {detail}"}
    return {"error": f"Request failed ({response.status_code}): {detail}"}


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text[:200]


def api_query_stream(
    api_url: str, api_key: str, question: str, thread_id: str, filters: dict | None, outcome: dict[str, Any]
):
    """Yield answer text from POST /query/stream as it arrives; fills ``outcome`` with the final
    ``done`` payload, an ``error`` message, or ``retracted=True`` when the output guardrail rejected it."""
    payload: dict[str, Any] = {"question": question, "thread_id": thread_id}
    if filters:
        payload["filters"] = filters
    try:
        with httpx.stream("POST", f"{api_url}/query/stream", json=payload, headers=_headers(api_key),
                          timeout=REQUEST_TIMEOUT) as response:
            if response.status_code != 200:
                response.read()
                outcome["error"] = _error_message(response)
                return
            event = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    if event == "token":
                        yield data["text"]
                    elif event == "retract":
                        outcome["retracted"] = True
                    elif event == "done":
                        outcome["done"] = data
                    elif event == "error":
                        outcome["error"] = f"Service error ({data.get('status')}): {data.get('detail')}"
    except httpx.TimeoutException:
        outcome["error"] = "The request timed out. The service may be under load; please try again."
    except httpx.HTTPError as exc:
        outcome["error"] = f"Could not reach the API at {api_url} ({type(exc).__name__})."


def _error_message(response: httpx.Response) -> str:
    detail = _detail(response)
    if response.status_code == 401:
        return "Authentication failed: check the API key in the sidebar."
    if response.status_code == 429:
        return f"Rate limit reached. Try again in {response.headers.get('Retry-After', 'a few')} seconds."
    if response.status_code in (502, 503):
        return f"Service temporarily unavailable: {detail}"
    return f"Request failed ({response.status_code}): {detail}"


def api_feedback(api_url: str, api_key: str, trace_id: str, rating: str, comment: str | None = None) -> bool:
    try:
        response = httpx.post(
            f"{api_url}/feedback", json={"trace_id": trace_id, "rating": rating, "comment": comment},
            headers=_headers(api_key), timeout=10,
        )
    except httpx.HTTPError:
        return False
    return response.status_code == 204


@st.cache_data(ttl=15, show_spinner=False)
def api_ready(api_url: str) -> tuple[bool, str]:
    try:
        response = httpx.get(f"{api_url}/ready", timeout=5)
    except httpx.HTTPError:
        return False, "API unreachable"
    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    if response.status_code == 200:
        return True, "Connected"
    return False, body.get("error", f"HTTP {response.status_code}")


# ---------- Rendering ----------


def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})", icon=":material/menu_book:"):
        for c in citations:
            location = f" · {c['location']}" if c.get("location") else ""
            score = f" · relevance {c['score']:.2f}" if c.get("score") is not None else ""
            st.markdown(f"**[{c['ref']}] {c.get('source') or 'unknown'}**{location}{score}")
            if c.get("dataset") == "noisy":
                st.badge("noisy dataset", icon=":material/warning:", color="orange")
            st.caption(c.get("snippet", "").replace("\n", " ") + "…")


def render_steps(steps: list[dict[str, Any]], latency_ms: float | None, model: str | None) -> None:
    if not steps:
        return
    total = f" · {latency_ms / 1000:.1f}s" if latency_ms else ""
    with st.expander(f"Reasoning steps ({len(steps)}){total}", icon=":material/account_tree:"):
        for step in steps:
            icon = NODE_ICONS.get(step["node"], ":material/circle:")
            st.markdown(f"{icon} **{step['node'].replace('_', ' ').title()}** — {step['summary']} "
                        f"<span style='opacity:0.6'>({step.get('latency_ms', 0):.0f} ms)</span>",
                        unsafe_allow_html=True)
            detail = {k: v for k, v in (step.get("detail") or {}).items() if v not in (None, "", [], {})}
            if detail:
                st.json(detail, expanded=False)
        if model:
            st.caption(f"Model: {model}")


def render_assistant(message: dict[str, Any], show_steps: bool) -> None:
    if message.get("error"):
        st.error(message["error"], icon=":material/error:")
        return
    if message.get("blocked"):
        st.warning(message["content"], icon=":material/block:")
    else:
        st.markdown(message["content"])
    if message.get("standalone_question") and message["standalone_question"] != message.get("question"):
        st.caption(f"Interpreted as: _{message['standalone_question']}_")
    render_citations(message.get("citations", []))
    if show_steps:
        render_steps(message.get("steps", []), message.get("latency_ms"), message.get("model"))
    if message.get("trace_id") and not message.get("blocked"):
        render_feedback(message["trace_id"])


def _on_feedback(trace_id: str) -> None:
    value = st.session_state.get(f"fb_{trace_id}")
    if value is None:
        return
    ok = api_feedback(st.session_state.api_url, st.session_state.api_key, trace_id, "up" if value == 1 else "down")
    st.session_state[f"fb_status_{trace_id}"] = "Thanks for the feedback!" if ok else "Couldn't send feedback."


def render_feedback(trace_id: str) -> None:
    st.feedback("thumbs", key=f"fb_{trace_id}", on_change=_on_feedback, args=(trace_id,))
    if status_text := st.session_state.get(f"fb_status_{trace_id}"):
        st.caption(status_text)


def _chain(first: str, rest):
    yield first
    yield from rest


# ---------- Session state ----------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex


def new_conversation() -> None:
    st.session_state.messages = []
    st.session_state.thread_id = uuid.uuid4().hex


# ---------- Sidebar ----------

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL).rstrip("/")
    api_key = st.text_input("API key", value=DEFAULT_API_KEY, type="password", help="Leave empty if the API has no auth")
    st.session_state.api_url, st.session_state.api_key = api_url, api_key
    ok, status_text = api_ready(api_url)
    st.badge(status_text, icon=":material/check_circle:" if ok else ":material/error:", color="green" if ok else "red")

    st.divider()
    scope = st.segmented_control("Search scope", list(DATASET_FILTERS), default="All documents")
    show_steps = st.toggle("Show reasoning steps", value=True)
    use_streaming = st.toggle("Stream answers", value=True, help="Show the answer as it is generated (/query/stream)")
    st.button("New conversation", icon=":material/add_comment:", on_click=new_conversation, width="stretch")
    st.caption(f"Conversation `{st.session_state.thread_id[:8]}`")

# ---------- Main ----------

st.title("Enterprise RAG Assistant")
st.caption("Ask about Kubernetes Jobs, CronJobs, autoscaling and Databricks job automation.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant(message, show_steps)

if question := st.chat_input("Ask a question about the documentation"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    filters = DATASET_FILTERS[scope or "All documents"]
    with st.chat_message("assistant"):
        if use_streaming:
            outcome: dict[str, Any] = {}
            live = st.empty()
            with live.container():
                with st.spinner("Checking guardrails, planning and searching…"):
                    tokens = api_query_stream(api_url, api_key, question, st.session_state.thread_id, filters, outcome)
                    first = next(tokens, None)  # wait for the first token under the spinner
                if first is not None:
                    st.write_stream(_chain(first, tokens))
            live.empty()  # replaced by the final rendering (which also handles a retracted answer)
            result = outcome.get("done") or {"error": outcome.get("error", "The stream ended without an answer.")}
        else:
            with st.spinner("Checking guardrails, planning, searching and answering…"):
                result = api_query(api_url, api_key, question, st.session_state.thread_id, filters)
        if "error" in result:
            message = {"role": "assistant", "content": "", "error": result["error"]}
        else:
            message = {"role": "assistant", "question": question, **result, "content": result["answer"]}
        st.session_state.messages.append(message)
        render_assistant(message, show_steps)
