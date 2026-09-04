"""Decides whether the current candidate set actually answers the user's request, or
whether the graph should loop back for another round of proposals.

The LLM makes the qualitative call (a converged-but-marginal candidate is a real,
judgment-shaped question a fixed threshold cannot capture well), but a hard,
deterministic floor always wins over it: once max_iterations is reached, the graph
proceeds to the report regardless of what the critic would prefer, so a model that
kept asking for "just one more round" could never turn this into an unbounded loop.
That split, LLM judgment inside a deterministic guardrail, is deliberate, not a
missing feature.
"""
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger("battery_agent.critic")

_SYSTEM_PROMPT = (
    "You are reviewing candidate battery electrode materials found for a materials "
    "scientist's request. Decide whether this candidate set is good enough to report, "
    "or whether another round of candidate generation is worth trying. A candidate set "
    "is generally sufficient if it has at least one Materials Project match, or at "
    "least one MLIP-screened candidate that converged with a plausible (negative, not "
    "wildly so) energy per atom. Be honest if the results are weak."
)


class CriticVerdict(BaseModel):
    sufficient: bool
    reason: str = Field(description="one or two sentences explaining the call")


def _summarize_candidates(state):
    mp = state.get("mp_candidates", [])
    screened = state.get("screened_candidates", [])
    converged = [c for c in screened if c.get("converged")]
    return (
        f"{len(mp)} Materials Project verified candidates. "
        f"{len(screened)} MLIP-screened proposed candidates, {len(converged)} of which converged. "
        f"Converged candidates: {[(c['formula'], round(c['energy_per_atom'], 3)) for c in converged]}"
    )


def make_critic_node(llm):
    structured_llm = llm.with_structured_output(CriticVerdict)

    def critic(state):
        summary = _summarize_candidates(state)
        result = structured_llm.invoke([
            ("system", _SYSTEM_PROMPT),
            ("human", f"Original request: {state['raw_query']}\n\nResults so far: {summary}"),
        ])
        verdict = result.model_dump()
        logger.info(f"critic verdict: {verdict}")
        return {
            "critic_verdict": verdict,
            "log": [f"critic: {'sufficient' if verdict['sufficient'] else 'insufficient'}, {verdict['reason']}"],
        }

    return critic


def loop_or_finish(state):
    """The second conditional edge, and the graph's actual cycle: back to
    candidate_proposer for another round, or on to the final report."""
    verdict = state.get("critic_verdict", {})
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)

    if verdict.get("sufficient") or iteration >= max_iterations:
        return "finish"
    return "retry"


def increment_iteration(state):
    return {"iteration": state.get("iteration", 0) + 1}
