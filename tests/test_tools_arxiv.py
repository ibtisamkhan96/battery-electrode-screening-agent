"""A real integration test against arXiv's live API, no mocking: this is the same
call this project verified directly (and where a real category-scoping bug was found
and fixed) before the rest of the literature_grounder node was built around it."""
from agent.tools.arxiv_search import search_literature


def test_battery_relevant_composition_returns_on_topic_hits():
    hits = search_literature("LiMnPO4", max_results=5)
    assert len(hits) > 0, "expected at least one real hit for a well-studied cathode composition"
    # every returned title should be plausibly on-topic, not the unrelated "hollow
    # cathode" plasma-physics results the unscoped version of this query returned
    # during development
    lowered_titles = " ".join(h.title.lower() for h in hits)
    assert "hollow cathode" not in lowered_titles
    print(f"PASSED: {len(hits)} on-topic hits for LiMnPO4")


def test_obscure_composition_returns_empty_list_not_an_error():
    hits = search_literature("Li3MgP7O21", max_results=5)
    assert isinstance(hits, list)
    print(f"PASSED: obscure composition returned {len(hits)} hits without raising")


if __name__ == "__main__":
    test_battery_relevant_composition_returns_on_topic_hits()
    test_obscure_composition_returns_empty_list_not_an_error()
