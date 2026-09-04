"""Relaxes every proposed candidate that actually has a starting structure and records
a stability estimate, so the critic has more to go on than "this formula exists."

A proposal with no structure (propose_substitutions was called without a parent
structure to substitute into, no Materials Project host structure was available) is
skipped and logged honestly rather than silently dropped or fabricated a fake energy.
"""
import logging

logger = logging.getLogger("battery_agent.mlip_screener")


def make_mlip_screener_node(relax_fn):
    """relax_fn matches agent.tools.mlip_relax.relax_and_screen's signature."""

    def mlip_screener(state):
        proposals = state.get("proposed_candidates", [])
        screened = []
        skipped = 0

        for proposal in proposals:
            structure = proposal.get("structure")
            if structure is None:
                skipped += 1
                continue
            result = relax_fn(structure)
            screened.append({
                **proposal,
                "energy_per_atom": result["energy_per_atom"],
                "converged": result["converged"],
                "num_relax_steps": result["num_steps"],
            })

        logger.info(f"MLIP-screened {len(screened)} candidates, skipped {skipped} with no structure")
        log_line = f"mlip_screener: relaxed {len(screened)} candidates"
        if skipped:
            log_line += f", skipped {skipped} with no starting structure to relax"
        return {
            "screened_candidates": screened,
            "log": [log_line],
        }

    return mlip_screener
