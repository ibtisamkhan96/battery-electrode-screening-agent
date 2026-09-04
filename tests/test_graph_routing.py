"""Proves the graph's two real conditional edges actually route, without a network
call or an API key anywhere in the test: every tool and the LLM are stand-ins injected
through the same build_graph(...) parameters production code uses, so this test
exercises the real graph object, not a simplified copy of it.
"""
from dataclasses import dataclass

from agent.graph import build_graph
from agent.tools.materials_project import ElectrodeCandidate
from agent.tools.substitution import ProposedCandidate
from agent.tools.arxiv_search import LiteratureHit
from agent.nodes.intake import ConstraintsOutput
from agent.nodes.critic import CriticVerdict


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeStructuredRunnable:
    def __init__(self, canned_output):
        self._canned_output = canned_output

    def invoke(self, messages):
        return self._canned_output


class FakeChatModel:
    """Stands in for a real ChatAnthropic/ChatOpenAI instance. with_structured_output
    returns a canned answer keyed by which Pydantic model was requested, and plain
    invoke() (report_writer's call) returns a fixed fake report."""

    def __init__(self, constraints_output, critic_output):
        self._constraints_output = constraints_output
        self._critic_output = critic_output

    def with_structured_output(self, model_cls):
        if model_cls is ConstraintsOutput:
            return _FakeStructuredRunnable(self._constraints_output)
        if model_cls is CriticVerdict:
            return _FakeStructuredRunnable(self._critic_output)
        raise ValueError(f"unexpected structured output model: {model_cls}")

    def invoke(self, messages):
        return _FakeMessage("STUB REPORT: " + str(len(messages)) + " messages received")


def _make_electrode_candidate(formula):
    return ElectrodeCandidate(
        battery_id=f"mp-{formula}", material_ids=["mp-1", "mp-2"], working_ion="Li",
        formula_charge=formula, formula_discharge=formula, average_voltage=3.5,
        capacity_grav=150.0, capacity_vol=500.0, energy_grav=525.0,
        stability_charge=0.01, stability_discharge=0.02, max_delta_volume=0.03,
    )


def test_skip_to_literature_branch_when_mp_has_enough_candidates():
    """db_search returns 5+ real candidates: candidate_proposer and mlip_screener
    should never run at all, the graph should go straight to literature grounding."""
    calls = {"propose": 0, "relax": 0, "search_electrodes": 0, "search_literature": 0}

    def fake_search_electrodes(**kwargs):
        calls["search_electrodes"] += 1
        return [_make_electrode_candidate(f"LiXPO4_{i}") for i in range(5)]

    def fake_propose(*args, **kwargs):
        calls["propose"] += 1
        return []

    def fake_relax(*args, **kwargs):
        calls["relax"] += 1
        return {"energy_per_atom": -5.0, "converged": True, "num_steps": 10}

    def fake_search_literature(formula, **kwargs):
        calls["search_literature"] += 1
        return [LiteratureHit(title=f"paper about {formula}", published="2024-01-01",
                               arxiv_id="1234.5678", url="http://arxiv.org/abs/1234.5678",
                               summary="a paper")]

    llm = FakeChatModel(
        constraints_output=ConstraintsOutput(working_ion="Li", application="cathode"),
        critic_output=CriticVerdict(sufficient=True, reason="plenty of verified candidates"),
    )

    app = build_graph(llm, fake_search_electrodes, fake_propose, fake_relax, fake_search_literature)
    result = app.invoke({"raw_query": "find a Li cathode"})

    assert calls["propose"] == 0, "candidate_proposer ran even though MP already had enough candidates"
    assert calls["relax"] == 0, "mlip_screener ran even though there were no proposals to screen"
    assert len(result["mp_candidates"]) == 5
    assert result["final_report"].startswith("STUB REPORT")
    print("PASSED: skip-to-literature branch")


def test_retry_loop_respects_max_iterations():
    """db_search returns nothing, critic always says insufficient: the graph must
    loop through candidate_proposer/mlip_screener/literature_grounder/critic more than
    once, but must still terminate once max_iterations is hit, not loop forever."""
    calls = {"propose": 0, "critic_llm": 0}

    def fake_search_electrodes(**kwargs):
        return []

    def fake_propose(parent_formula, exclude_elements=None, max_candidates=5):
        calls["propose"] += 1
        return [ProposedCandidate(
            formula="LiMnPO4", parent_formula=parent_formula, substituted_element="Fe",
            replacement_element="Mn", rationale="test substitution", structure=None,
        )]

    def fake_relax(*args, **kwargs):
        return {"energy_per_atom": -5.0, "converged": True, "num_steps": 10}

    def fake_search_literature(formula, **kwargs):
        return []

    class _CountingCriticRunnable:
        """Counts .invoke() calls, not construction: with_structured_output(...) is
        called once at graph-build time by design (agent/nodes/critic.py builds the
        structured wrapper once and reuses it), so the call count that actually
        reflects how many times the critic reasoned lives on invoke(), not here."""

        def invoke(self, messages):
            calls["critic_llm"] += 1
            return CriticVerdict(sufficient=False, reason="still too thin")

    class AlwaysInsufficientChatModel(FakeChatModel):
        def with_structured_output(self, model_cls):
            if model_cls is CriticVerdict:
                return _CountingCriticRunnable()
            return super().with_structured_output(model_cls)

    llm = AlwaysInsufficientChatModel(
        constraints_output=ConstraintsOutput(working_ion="Li", application="cathode"),
        critic_output=None,
    )

    app = build_graph(llm, fake_search_electrodes, fake_propose, fake_relax, fake_search_literature)
    result = app.invoke({"raw_query": "find an obscure Li cathode"}, config={"recursion_limit": 50})

    # max_iterations defaults to 2 (set inside the intake node): one initial proposer
    # pass plus up to 2 retries = 3 total calls to candidate_proposer, never unbounded
    assert calls["propose"] == 3, f"expected exactly 3 candidate_proposer calls, got {calls['propose']}"
    assert calls["critic_llm"] == 3, f"expected exactly 3 critic calls, got {calls['critic_llm']}"
    assert result["final_report"].startswith("STUB REPORT")
    print("PASSED: retry loop terminates at max_iterations instead of looping forever")


if __name__ == "__main__":
    test_skip_to_literature_branch_when_mp_has_enough_candidates()
    test_retry_loop_respects_max_iterations()
    print("ALL GRAPH ROUTING TESTS PASSED")
