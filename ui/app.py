"""Streamlit UI: submits a query to the FastAPI backend, polls for the result, and
shows the agent's step-by-step reasoning log live while it runs, not just the final
answer, since watching *how* a multi-agent graph got somewhere is the actual point
of building one instead of calling a single LLM.
"""
import os
import time

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Battery Electrode Screening Agent", layout="wide")
st.title("Battery Electrode Screening Agent")
st.caption(
    "A LangGraph agent that searches Materials Project's real electrode data, "
    "proposes new candidates by isovalent substitution when nothing existing fits, "
    "screens them with a local ML interatomic potential, and checks arXiv for "
    "whether anyone has actually studied them."
)

with st.sidebar:
    st.header("Your API key")
    st.caption(
        "This runs on your own key, not a shared one, so one visitor's usage "
        "can't rate-limit or bill another's. Nothing is stored server-side: the "
        "key is sent with this request only and used for this run."
    )
    provider_label = st.radio(
        "Provider",
        ["Anthropic (Claude) — recommended", "OpenAI"],
        help=(
            "Recommended: Anthropic. This agent's report-writing and critic/retry "
            "loop are multi-step tool-calling work, and it was built and tested "
            "against Claude Sonnet. OpenAI's gpt-4o-mini is supported as a cheaper "
            "alternative but hasn't had the same testing depth here."
        ),
    )
    provider = "anthropic" if provider_label.startswith("Anthropic") else "openai"
    key_help_url = (
        "https://console.anthropic.com/settings/keys"
        if provider == "anthropic"
        else "https://platform.openai.com/api-keys"
    )
    user_api_key = st.text_input(
        f"{'Anthropic' if provider == 'anthropic' else 'OpenAI'} API key",
        type="password",
        placeholder="sk-...",
    )
    st.caption(f"[Get a key]({key_help_url})")

with st.form("query_form"):
    query = st.text_area(
        "Describe the electrode you're looking for",
        value="Find a Li-ion cathode candidate with an average voltage above 3.5V, "
              "that doesn't use cobalt.",
        height=80,
    )
    submitted = st.form_submit_button("Screen candidates")


def _poll_job(job_id, log_placeholder, timeout_s=180):
    seen_log_lines = 0
    start = time.time()
    while time.time() - start < timeout_s:
        response = requests.get(f"{API_BASE_URL}/query/{job_id}", timeout=30)
        response.raise_for_status()
        status = response.json()

        log_lines = status.get("log", [])
        if len(log_lines) > seen_log_lines:
            log_placeholder.code("\n".join(log_lines), language=None)
            seen_log_lines = len(log_lines)

        if status["status"] in ("completed", "failed"):
            return status
        time.sleep(1.5)
    raise TimeoutError("job did not finish in time")


def _render_candidates_table(result):
    rows = []
    for c in result.get("mp_candidates", []):
        rows.append({
            "formula": c.get("formula_discharge") or c.get("formula_charge"),
            "source": "Materials Project (DFT-verified)",
            "working_ion": c.get("working_ion"),
            "average_voltage": c.get("average_voltage"),
            "capacity_grav": c.get("capacity_grav"),
            "stability_charge": c.get("stability_charge"),
            "energy_per_atom": None,
        })
    for c in result.get("screened_candidates", []):
        rows.append({
            "formula": c.get("formula"),
            "source": "MLIP-screened proposal" + ("" if c.get("converged") else " (did not converge)"),
            "working_ion": None,
            "average_voltage": None,
            "capacity_grav": None,
            "stability_charge": None,
            "energy_per_atom": c.get("energy_per_atom"),
        })
    if not rows:
        return None
    return pd.DataFrame(rows)


if submitted and not user_api_key.strip():
    st.warning("Add your API key in the sidebar first, this demo doesn't run on a shared one.")
    st.stop()

if submitted and query.strip():
    with st.spinner("Submitting query..."):
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query, "provider": provider, "api_key": user_api_key.strip()},
            timeout=30,
        )
        if response.status_code == 400:
            st.error(f"Couldn't start the run: {response.json().get('detail', response.text)}")
            st.stop()
        response.raise_for_status()
        job_id = response.json()["job_id"]

    st.subheader("Agent trace")
    log_placeholder = st.empty()

    try:
        status = _poll_job(job_id, log_placeholder)
    except TimeoutError:
        st.error("The agent didn't finish in time, try again or check the API logs.")
        st.stop()

    if status["status"] == "failed":
        st.error(f"Agent run failed: {status.get('error')}")
        st.stop()

    result = status["result"]

    st.subheader("Report")
    st.markdown(result.get("final_report", "no report produced"))

    st.subheader("Candidates")
    table = _render_candidates_table(result)
    if table is not None:
        st.dataframe(table, use_container_width=True)
        numeric = table.dropna(subset=["capacity_grav"]) if "capacity_grav" in table else None
        if numeric is not None and not numeric.empty:
            st.bar_chart(numeric.set_index("formula")["capacity_grav"])
    else:
        st.info("No candidates were found or proposed for this query.")

    with st.expander("Literature evidence"):
        for formula, hits in result.get("literature", {}).items():
            st.markdown(f"**{formula}**")
            if not hits:
                st.markdown("_no arXiv hits found_")
            for h in hits:
                st.markdown(f"- [{h['title']}]({h['url']}) ({h['published']})")
