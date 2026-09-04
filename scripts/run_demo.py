"""Runs the agent end to end from the command line, no API/UI involved, meant for
recording the demo video or a quick sanity check with real keys.

Usage:
    python scripts/run_demo.py "Find a Li-ion cathode above 3.5V that doesn't use cobalt"
"""
import sys

sys.path.insert(0, ".")

from agent.graph import build_graph
from agent.llm import get_chat_model
from agent.tools.arxiv_search import search_literature
from agent.tools.materials_project import search_electrodes
from agent.tools.mlip_relax import relax_and_screen
from agent.tools.substitution import propose_substitutions
from observability.logging_config import configure_langsmith, configure_logging


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    query = sys.argv[1]

    configure_logging()
    configure_langsmith()

    llm = get_chat_model()
    app = build_graph(llm, search_electrodes, propose_substitutions, relax_and_screen, search_literature)

    print(f"\nRunning: {query}\n" + "-" * 60)
    result = app.invoke({"raw_query": query})

    print("\n" + "=" * 60)
    print("AGENT TRACE")
    print("=" * 60)
    for line in result.get("log", []):
        print(f"  - {line}")

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result.get("final_report", "no report produced"))


if __name__ == "__main__":
    main()
