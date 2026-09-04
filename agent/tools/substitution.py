"""Proposes new candidate compositions when Materials Project has no existing match.

Uses isovalent substitution: swap one element in a known framework for another
element in the same periodic-table group, since same-group elements share a valence
and are the standard first move a materials chemist makes when hunting for a new
electrode composition (Fe -> Mn/Co/Ni in a phosphate framework is the textbook example,
LiFePO4 -> LiMnPO4 is a real, studied substitution). This stays a transparent,
explainable rule rather than a generative black box, deliberately, so every proposed
composition can be traced back to the one substitution that produced it.
"""
from dataclasses import dataclass
from typing import Optional

from langsmith import traceable
from pymatgen.core import Composition, Structure
from pymatgen.core.periodic_table import Element

# Elements realistic for a battery electrode framework: mobile ions plus the
# transition metals and anion-formers that actually show up in known electrode
# chemistries. Substitution candidates are pulled from this pool, not the whole
# periodic table, since a same-group swap to e.g. a noble gas is not chemically real.
_ELECTRODE_RELEVANT = [
    "Li", "Na", "K", "Mg", "Ca", "Zn",
    "Fe", "Mn", "Co", "Ni", "Cu", "Ti", "V", "Cr", "Zr", "Nb", "Mo", "Al",
    "O", "S", "F", "P", "Si",
]


@dataclass
class ProposedCandidate:
    formula: str
    parent_formula: str
    substituted_element: str
    replacement_element: str
    rationale: str
    structure: Optional[Structure] = None


def _substitution_pool(element_symbol, exclude_elements):
    """Same-group substitution is the right rule for s-block and p-block elements
    (Li -> Na -> K, O -> S, real valence-preserving periodic trends). It is the
    wrong rule for d-block transition metals: the textbook electrode substitution
    family (Fe -> Mn -> Co -> Ni in LiFePO4 and its relatives, the real olivine
    series) sits across *adjacent* groups in the *same period*, not one column,
    since first-row transition metals share similar ionic radii and 2+/3+
    accessibility rather than an identical group. Branching on block, instead of
    applying one rule everywhere, is what keeps this heuristic chemically real.
    """
    el = Element(element_symbol)
    pool = []
    for symbol in _ELECTRODE_RELEVANT:
        if symbol == element_symbol or symbol in exclude_elements:
            continue
        candidate = Element(symbol)
        if candidate.block != el.block:
            continue
        if el.block == "d":
            same_family = candidate.row == el.row   # same-period transition metals
        else:
            same_family = candidate.group == el.group   # same-group s/p-block elements
        if same_family:
            pool.append(symbol)
    return pool


@traceable(name="substitution.propose_substitutions", run_type="tool")
def propose_substitutions(parent_formula, parent_structure=None, exclude_elements=None, max_candidates=5):
    """Generates same-group substitution candidates for one framework formula.

    parent_formula: a known composition to use as the substitution template, e.g. "LiFePO4".
    parent_structure: the parent's real pymatgen Structure, when one is available (Materials
        Project's electrode documents carry one on `host_structure`). When given, each candidate
        also gets a real substituted Structure, the parent's actual lattice and site positions
        with every site of the substituted element swapped to the replacement element, a real
        starting geometry for relaxation, not just a formula with no atomic positions. When not
        given, candidates still come back with a formula and rationale, just no structure.
    exclude_elements: elements the user's constraints ruled out, never proposed as a replacement.
    Returns a list of ProposedCandidate, one per real substitution found, empty if
    the parent has no substitutable elements in the electrode-relevant pool.
    """
    exclude_elements = set(exclude_elements or [])
    parent = Composition(parent_formula)
    candidates = []

    for element_symbol in parent.chemical_system.split("-"):
        if element_symbol not in _ELECTRODE_RELEVANT:
            continue
        for replacement in _substitution_pool(element_symbol, exclude_elements | {element_symbol}):
            amount = parent[Element(element_symbol)]
            new_amounts = {str(el): amt for el, amt in parent.items()}
            new_amounts.pop(element_symbol)
            new_amounts[replacement] = amount
            new_formula = Composition(new_amounts).reduced_formula

            new_structure = None
            if parent_structure is not None:
                new_structure = parent_structure.copy()
                new_structure.replace_species({Element(element_symbol): Element(replacement)})

            same_period = Element(replacement).block == "d"
            rationale = (
                f"same-period transition-metal substitution: {replacement} sits in the same "
                f"row as {element_symbol} in {parent.reduced_formula}, the real olivine-family move"
                if same_period else
                f"same-group substitution: {replacement} shares {element_symbol}'s "
                f"periodic group in {parent.reduced_formula}, a valence-preserving swap"
            )
            candidates.append(ProposedCandidate(
                formula=new_formula,
                parent_formula=parent.reduced_formula,
                substituted_element=element_symbol,
                replacement_element=replacement,
                rationale=rationale,
                structure=new_structure,
            ))
            if len(candidates) >= max_candidates:
                return candidates

    return candidates
