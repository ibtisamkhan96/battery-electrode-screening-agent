"""Writes the final report a person actually reads: ranked candidates, their real
evidence (Materials Project data, MLIP stability estimate, literature hits), and the
honest caveats, not a summary that hides how thin the results were if they were thin.
"""
import logging

logger = logging.getLogger("battery_agent.report_writer")

_SYSTEM_PROMPT = (
    "Write a materials scientist a clear, structured markdown report on candidate "
    "battery electrode materials. Rank candidates with the strongest real evidence "
    "first (a Materials Project verified entry outranks an MLIP-screened proposal, "
    "which outranks a proposal with no stability data at all). For each candidate, "
    "state what evidence backs it and what is still unverified. If the results are "
    "weak or thin, say so plainly rather than overselling them. Cite literature hits "
    "by title when they exist for a candidate."
)


def make_report_writer_node(llm):
    def report_writer(state):
        payload = {
            "original_request": state["raw_query"],
            "constraints": state["constraints"],
            "materials_project_candidates": state.get("mp_candidates", []),
            "mlip_screened_candidates": state.get("screened_candidates", []),
            "literature": state.get("literature", {}),
            "rounds_run": state.get("iteration", 0) + 1,
            "critic_verdict": state.get("critic_verdict", {}),
        }
        response = llm.invoke([
            ("system", _SYSTEM_PROMPT),
            ("human", f"Data to report on:\n{payload}"),
        ])
        report_text = response.content if hasattr(response, "content") else str(response)
        logger.info("final report written")
        return {
            "final_report": report_text,
            "log": ["report_writer: composed final report"],
        }

    return report_writer
