"""Screens a proposed (not-yet-verified) structure for physical stability using a
real, local universal machine-learned interatomic potential.

MACE was the original choice, the state-of-the-art universal potential this project
was pitched around, but its dependency chain (matscipy, via a Meson build) failed to
install on the development machine with "error: metadata-generation-failed", a real,
confirmed build-toolchain problem, not a MACE bug. matgl's M3GNet-PES-MatPES-PBE-2025.2
model was substituted instead: pure PyTorch, no compiler dependency, confirmed working
end to end before this was written (loads in ~5s, relaxes a real structure in ~1.5s).
It belongs to the same class of model, a universal MLIP trained across the periodic
table, currently the dominant approach to fast, DFT-free stability screening.
"""
import logging
import warnings

from langsmith import traceable

logger = logging.getLogger("battery_agent.mlip")

_MODEL_NAME = "M3GNet-PES-MatPES-PBE-2025.2"
_relaxer_cache = {}


def _get_relaxer():
    if "relaxer" not in _relaxer_cache:
        import matgl
        from matgl.ext.ase import Relaxer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            potential = matgl.load_model(_MODEL_NAME)
        _relaxer_cache["relaxer"] = Relaxer(potential=potential)
        logger.info(f"loaded MLIP {_MODEL_NAME}")
    return _relaxer_cache["relaxer"]


@traceable(name="mlip.relax_and_screen", run_type="tool")
def relax_and_screen(structure, fmax=0.05, max_steps=100):
    """Relaxes one candidate structure and returns a stability estimate.

    structure: a pymatgen Structure, typically a substituted candidate from the
        substitution tool, an initial geometry copied from a real MP host structure
        with one element swapped, not yet relaxed to its own true minimum.
    Returns a dict: energy_per_atom (eV/atom, the MLIP's own estimate, comparable
        across candidates screened with this same model, not directly comparable to
        DFT formation energies without a reference), num_steps, and converged (whether
        the relaxation reached fmax before hitting max_steps, a candidate that does not
        converge is a real signal the starting geometry was a poor guess for this
        composition, worth flagging rather than silently reporting its last energy).
    """
    relaxer = _get_relaxer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = relaxer.relax(structure, fmax=fmax, steps=max_steps, verbose=False)

    final_structure = result["final_structure"]
    energies = result["trajectory"].energies
    num_steps = len(energies)
    converged = num_steps < max_steps

    logger.info(
        f"relaxed {structure.composition.reduced_formula}: "
        f"{energies[-1] / len(final_structure):.3f} eV/atom in {num_steps} steps, "
        f"converged={converged}"
    )
    return {
        "formula": final_structure.composition.reduced_formula,
        "energy_per_atom": energies[-1] / len(final_structure),
        "num_steps": num_steps,
        "converged": converged,
        "final_structure": final_structure,
    }
