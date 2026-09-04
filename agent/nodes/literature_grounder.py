"""Checks whether anyone has actually studied each surviving candidate.

Runs on the combined candidate set, Materials Project matches and MLIP-screened
proposals alike, since "has this been studied" matters just as much for a real
database entry as for a newly proposed one. Only formulas not already looked up in
an earlier pass get searched, `literature` accumulates across loop iterations rather
than re-querying arXiv for a candidate that survived from the previous round.
"""
import logging

logger = logging.getLogger("battery_agent.literature_grounder")


def make_literature_grounder_node(search_fn):
    """search_fn matches agent.tools.arxiv_search.search_literature's signature."""

    def literature_grounder(state):
        already_known = state.get("literature", {})
        formulas = set()
        for c in state.get("mp_candidates", []):
            formulas.add(c["formula_discharge"] or c["formula_charge"])
        for c in state.get("screened_candidates", []):
            formulas.add(c["formula"])

        new_results = {}
        for formula in formulas:
            if formula in already_known:
                continue
            hits = search_fn(formula, max_results=3)
            new_results[formula] = [
                {"title": h.title, "published": h.published, "url": h.url}
                for h in hits
            ]

        logger.info(f"literature search covered {len(new_results)} new formulas")
        return {
            "literature": new_results,
            "log": [f"literature_grounder: searched arXiv for {len(new_results)} formulas not already checked"],
        }

    return literature_grounder
