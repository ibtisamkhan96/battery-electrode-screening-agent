"""Pure logic, no network, no mocking needed: these run the real substitution tool
exactly as written."""
from pymatgen.core import Lattice, Structure

from agent.tools.substitution import propose_substitutions


def test_transition_metal_substitution_finds_real_olivine_family():
    """This is the case that caught a real bug during development: a naive same-group
    rule missed Fe -> Mn/Ni entirely, since transition metals substitute across the
    same period, not the same group. This test locks that fix in."""
    candidates = propose_substitutions("LiFePO4", max_candidates=10)
    formulas = {c.formula for c in candidates}
    assert "LiMnPO4" in formulas, "missing the real, textbook olivine substitution Fe -> Mn"
    assert "LiNiPO4" in formulas, "missing the real, textbook olivine substitution Fe -> Ni"
    print("PASSED: transition-metal substitution finds the real olivine family")


def test_exclude_elements_is_respected():
    candidates = propose_substitutions("LiFePO4", exclude_elements=["Co", "Mn", "Ni", "Cu", "Ti", "V", "Cr", "Zn"],
                                        max_candidates=10)
    replacements = {c.replacement_element for c in candidates}
    assert "Co" not in replacements
    assert "Mn" not in replacements
    print("PASSED: excluded elements never appear as a replacement")


def test_structure_substitution_preserves_site_count_and_swaps_correct_element():
    parent = Structure.from_spacegroup(
        "Pnma", Lattice.orthorhombic(6.0, 10.4, 4.7), ["Li", "Fe", "P", "O"],
        [[0, 0, 0], [0.28, 0.25, 0.97], [0.09, 0.42, 0.42], [0.1, 0.45, 0.75]],
    )
    candidates = propose_substitutions("LiFePO4", parent_structure=parent, max_candidates=3)
    assert candidates, "expected at least one candidate"
    for c in candidates:
        assert c.structure is not None
        assert len(c.structure) == len(parent), "substitution changed the site count"
        assert "Fe" not in [str(s.specie) for s in c.structure], "old element still present after substitution"
    print("PASSED: structure substitution preserves geometry and swaps the right element")


def test_no_structure_given_returns_formula_only_candidates():
    candidates = propose_substitutions("LiFePO4", max_candidates=3)
    assert all(c.structure is None for c in candidates)
    print("PASSED: no parent structure means no fabricated structure on the candidates")


if __name__ == "__main__":
    test_transition_metal_substitution_finds_real_olivine_family()
    test_exclude_elements_is_respected()
    test_structure_substitution_preserves_site_count_and_swaps_correct_element()
    test_no_structure_given_returns_formula_only_candidates()
    print("ALL SUBSTITUTION TESTS PASSED")
