"""Streamlit Community Cloud entry point.

Community Cloud runs exactly one process with no way to also host a separate FastAPI
service behind it for free, so this calls the LangGraph agent directly in-process
instead of going through api/main.py over HTTP. The real FastAPI backend still exists,
is still fully tested, and is still what ui/app.py and Docker/Render use, this file
exists only because Community Cloud's single-process, no-card free tier cannot run two
linked services the way Docker Compose or Render can.

Secrets come from st.secrets (Community Cloud's own secrets manager, set under app
Settings > Secrets as TOML) rather than a .env file, and are copied into os.environ at
startup since the tool modules (Materials Project, LangSmith) read real environment
variables, the same code path Docker and local runs already use.

The LLM key is the one exception: it is deliberately NOT read from st.secrets or
os.environ. Community Cloud runs one shared process for every visitor, so a key baked
into a process-wide global (env var, or an @st.cache_resource-wrapped graph) would be
reused across concurrent users, one visitor's key answering another's query. Each
visitor pastes their own key in the sidebar instead, and it is passed as an explicit
function argument all the way to the LLM client, never touching a shared variable.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# st.secrets raises if no secrets.toml exists at all, which is the normal case when
# testing this locally against a real .env instead (see scripts/run_demo.py's pattern),
# so a missing secrets file falls back to whatever is already in the environment.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except Exception:
    pass

from agent.graph import build_graph
from agent.llm import get_chat_model
from agent.tools.arxiv_search import search_literature
from agent.tools.materials_project import search_electrodes
from agent.tools.mlip_relax import relax_and_screen
from agent.tools.substitution import propose_substitutions
from observability.logging_config import configure_langsmith, configure_logging

configure_logging()
configure_langsmith()

st.set_page_config(page_title="Battery Electrode Screening Agent", layout="wide")
st.title("Battery Electrode Screening Agent")
st.caption(
    "A LangGraph agent that searches Materials Project's real electrode data, "
    "proposes new candidates by isovalent substitution when nothing existing fits, "
    "screens them with a local ML interatomic potential, and checks arXiv for "
    "whether anyone has actually studied them. Full source and architecture docs: "
    "https://github.com/ibtisamkhan96/battery-electrode-screening-agent"
)

with st.sidebar:
    st.header("Your API key")
    st.caption(
        "This demo runs on your own key, not a shared one, so one visitor's usage "
        "can't rate-limit or bill another's. Nothing is stored: the key lives only "
        "in this browser tab's session and is used for this run only."
    )
    provider_label = st.radio(
        "Provider",
        ["Anthropic (Claude), recommended", "OpenAI"],
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


def _build_graph(provider, api_key):
    llm = get_chat_model(provider=provider, api_key=api_key)
    return build_graph(llm, search_electrodes, propose_substitutions, relax_and_screen, search_literature)


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
    return pd.DataFrame(rows) if rows else None


with st.form("query_form"):
    query = st.text_area(
        "Describe the electrode you're looking for",
        value="Find a Li-ion cathode candidate with an average voltage above 3.5V, "
              "that doesn't use cobalt.",
        height=80,
    )
    submitted = st.form_submit_button("Screen candidates")

if submitted and not user_api_key.strip():
    st.warning("Add your API key in the sidebar first, this demo doesn't run on a shared one.")
    st.stop()

if submitted and query.strip():
    log_placeholder = st.empty()
    log_placeholder.info("Running the agent, this can take 30-90 seconds...")

    try:
        graph = _build_graph(provider, user_api_key.strip())
        start = time.time()
        result = graph.invoke({"raw_query": query})
        log_placeholder.success(f"Done in {time.time() - start:.0f}s")
    except Exception as e:
        log_placeholder.error(f"Agent run failed: {e}")
        st.stop()

    st.subheader("Agent trace")
    st.code("\n".join(result.get("log", [])), language=None)

    st.subheader("Report")
    st.markdown(result.get("final_report", "no report produced"))

    st.subheader("Candidates")
    table = _render_candidates_table(result)
    if table is not None:
        st.dataframe(table, use_container_width=True)
    else:
        st.info("No candidates were found or proposed for this query.")

    with st.expander("Literature evidence"):
        for formula, hits in result.get("literature", {}).items():
            st.markdown(f"**{formula}**")
            if not hits:
                st.markdown("_no arXiv hits found_")
            for h in hits:
                st.markdown(f"- [{h['title']}]({h['url']}) ({h['published']})")
