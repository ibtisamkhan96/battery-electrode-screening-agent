"""Proposes new candidate compositions when Materials Project didn't have enough on its own.

Picks a parent formula to substitute from: the strongest real Materials Project match
when one exists (closest thing to ground truth for these exact constraints), or a
well-known seed framework for the requested chemistry when Materials Project found
nothing at all. On a second pass through this node (the critic sent the graph back),
elements already tried in the previous round get added to the exclusion set, so a
retry explores new chemistry instead of regenerating the same rejected candidates.
"""
import logging
from dataclasses import asdict

logger = logging.getLogger("battery_agent.candidate_proposer")

# well-known, real, commonly cited framework compositions per (working_ion, application),
# used only when Materials Project's own search came back with nothing to build from
_SEED_FRAMEWORKS = {
    ("Li", "cathode"): "LiFePO4",
    ("Li", "anode"): "Li4Ti5O12",
    ("Na", "cathode"): "NaFePO4",
    ("Mg", "cathode"): "MgFePO4F",
    ("Zn", "cathode"): "ZnMn2O4",
}


def _pick_parent(state):
    mp_candidates = state.get("mp_candidates", [])
    if mp_candidates:
        return mp_candidates[0]["formula_discharge"] or mp_candidates[0]["formula_charge"]

    c = state["constraints"]
    working_ion = c.get("working_ion") or "Li"
    application = c.get("application") or "cathode"
    seed = _SEED_FRAMEWORKS.get((working_ion, application))
    if seed:
        return seed
    # last resort: a generic, real, widely studied Li-ion cathode framework
    return "LiFePO4"


def make_candidate_proposer_node(propose_fn):
    """propose_fn matches agent.tools.substitution.propose_substitutions's signature."""

    def candidate_proposer(state):
        c = state["constraints"]
        already_tried = {
            cand["replacement_element"]
            for cand in state.get("screened_candidates", [])
        }
        exclude = set(c.get("elements_exclude") or []) | already_tried

        parent_formula = _pick_parent(state)
        proposals = propose_fn(parent_formula, exclude_elements=list(exclude), max_candidates=5)
        proposal_dicts = [asdict(p) for p in proposals]

        logger.info(f"proposed {len(proposal_dicts)} candidates from parent {parent_formula}")
        return {
            "proposed_candidates": proposal_dicts,
            "log": [
                f"candidate_proposer: substituted from {parent_formula}, "
                f"excluding {sorted(exclude) or 'nothing'}, proposed {len(proposal_dicts)} candidates"
            ],
        }

    return candidate_proposer
