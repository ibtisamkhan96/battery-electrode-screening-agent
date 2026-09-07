"""Checks whether anyone has actually studied each surviving candidate.

Runs on the combined candidate set, Materials Project matches and MLIP-screened
proposals alike, since "has this been studied" matters just as much for a real
database entry as for a newly proposed one. Only formulas not already looked up in
an earlier pass get searched, `literature` accumulates across loop iterations rather
than re-querying arXiv for a candidate that survived from the previous round.
"""
import logging
import time

logger = logging.getLogger("battery_agent.literature_grounder")

# A courtesy delay between successive arXiv calls. Raising Materials Project's fetch
# limit (see agent/nodes/db_search.py) means this loop can now run for far more
# formulas in one pass than before, up to 15 instead of 5, and pushing that many
# requests through with no pacing tripped arXiv's rate limit mid-run, confirmed live:
# a 15-candidate search hit a real HTTP 429 that exhausted the tool's own retry
# budget and crashed the whole graph. One second between calls is well inside
# arXiv's documented courtesy window and costs little next to the 30-90s the agent
# already takes end to end.
_REQUEST_DELAY_S = 1.0


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
        made_request = False
        for formula in formulas:
            if formula in already_known:
                continue
            if made_request:
                time.sleep(_REQUEST_DELAY_S)
            try:
                hits = search_fn(formula, max_results=3)
            except Exception as e:
                # A literature miss on one formula shouldn't take down the whole
                # report: arXiv rate-limiting or a transient network error means
                # "couldn't check this one," not "the agent failed," so it's
                # recorded as no evidence found rather than crashing the graph.
                logger.warning(f"literature search for {formula!r} failed, treating as no hits found: {e}")
                hits = []
            made_request = True
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
