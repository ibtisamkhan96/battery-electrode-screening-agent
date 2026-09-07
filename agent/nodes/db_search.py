"""Searches Materials Project's real electrode dataset against the parsed constraints.

Kept as a thin node around the tool function on purpose, all the real logic (field
names, the stability filter, why the electrode endpoint rather than plain summary
search) lives in agent/tools/materials_project.py where it was verified, this node's
only job is translating graph state into that function's arguments and back.
"""
import logging
from dataclasses import asdict

logger = logging.getLogger("battery_agent.db_search")

TARGET_CANDIDATES = 5  # "sufficient" threshold: stop proposing new candidates once we have this many

# How many real matches to actually request from Materials Project. Deliberately larger
# than TARGET_CANDIDATES: requesting exactly the threshold means every search that clears
# it reports back that exact same number, which looks like a coincidence but is really the
# fetch cap, not the true count. Fetching more than the threshold needs lets "returned N
# candidates" reflect how many real matches actually exist (up to this limit), while the
# threshold below still decides whether that's enough to stop.
FETCH_LIMIT = 15


def make_db_search_node(search_fn):
    """search_fn matches agent.tools.materials_project.search_electrodes's signature,
    injected rather than imported directly so tests can pass a fake with no live
    MP_API_KEY or network call involved."""

    def db_search(state):
        c = state["constraints"]
        candidates = search_fn(
            working_ion=c.get("working_ion"),
            elements=c.get("elements_include") or None,
            exclude_elements=c.get("elements_exclude") or None,
            average_voltage=(c.get("average_voltage_min"), c.get("average_voltage_max")),
            capacity_grav=(c.get("capacity_grav_min"), None),
            num_candidates=FETCH_LIMIT,
        )
        candidate_dicts = [asdict(candidate) for candidate in candidates]
        logger.info(f"Materials Project electrode search returned {len(candidate_dicts)} candidates")
        return {
            "mp_candidates": candidate_dicts,
            "log": [f"db_search: found {len(candidate_dicts)} verified electrode candidates in Materials Project"],
        }

    return db_search


def needs_more_candidates(state):
    """The first conditional edge: skip straight to literature grounding when
    Materials Project alone already has enough DFT-verified candidates, no reason to
    spend an MLIP relaxation on a composition that already has real ground-truth data."""
    return "propose" if len(state.get("mp_candidates", [])) < TARGET_CANDIDATES else "skip_to_literature"
