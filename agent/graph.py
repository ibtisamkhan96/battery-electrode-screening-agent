"""Wires the state graph together.

Two real conditional edges, not one straight pipeline:

  db_search --[enough MP candidates already?]--> literature_grounder (skip proposing)
            --[not enough]--> candidate_proposer -> mlip_screener -> literature_grounder

  literature_grounder -> critic --[sufficient, or out of retries]--> report_writer -> END
                                --[insufficient, retries left]--> candidate_proposer (loop)

The second edge is the actual cycle: LangGraph's reason for existing over a plain
linear CrewAI-style pipeline is that a graph, unlike a crew, can route back to an
earlier node based on what a later one decided, exactly the generate -> screen ->
critique -> refine loop that real materials-discovery pipelines (GNoME, A-Lab-style
screening) run in practice.
"""
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.intake import make_intake_node
from agent.nodes.db_search import make_db_search_node, needs_more_candidates
from agent.nodes.candidate_proposer import make_candidate_proposer_node
from agent.nodes.mlip_screener import make_mlip_screener_node
from agent.nodes.literature_grounder import make_literature_grounder_node
from agent.nodes.critic import make_critic_node, loop_or_finish, increment_iteration
from agent.nodes.report_writer import make_report_writer_node


def build_graph(llm, search_electrodes_fn, propose_substitutions_fn, relax_and_screen_fn, search_literature_fn):
    """Every tool and the LLM are injected as parameters, not imported inside node
    modules and called directly, so this same function builds a fully real graph in
    production and a fully mocked one in tests, with no code path that only exists
    for testing."""
    graph = StateGraph(AgentState)

    graph.add_node("intake", make_intake_node(llm))
    graph.add_node("db_search", make_db_search_node(search_electrodes_fn))
    graph.add_node("candidate_proposer", make_candidate_proposer_node(propose_substitutions_fn))
    graph.add_node("mlip_screener", make_mlip_screener_node(relax_and_screen_fn))
    graph.add_node("literature_grounder", make_literature_grounder_node(search_literature_fn))
    graph.add_node("critic", make_critic_node(llm))
    graph.add_node("increment_iteration", increment_iteration)
    graph.add_node("report_writer", make_report_writer_node(llm))

    graph.set_entry_point("intake")
    graph.add_edge("intake", "db_search")

    graph.add_conditional_edges(
        "db_search",
        needs_more_candidates,
        {"propose": "candidate_proposer", "skip_to_literature": "literature_grounder"},
    )
    graph.add_edge("candidate_proposer", "mlip_screener")
    graph.add_edge("mlip_screener", "literature_grounder")
    graph.add_edge("literature_grounder", "critic")

    graph.add_conditional_edges(
        "critic",
        loop_or_finish,
        {"retry": "increment_iteration", "finish": "report_writer"},
    )
    graph.add_edge("increment_iteration", "candidate_proposer")
    graph.add_edge("report_writer", END)

    return graph.compile()
