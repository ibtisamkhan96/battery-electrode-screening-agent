"""A real integration test: loads the actual M3GNet-PES-MatPES-PBE-2025.2 model and
relaxes a real structure, no mocking, the same call this project verified working
before any other code was written around it. Slower than the rest of the suite
(~5-10s for the model load) since it is deliberately proving the real thing works,
not a stand-in for it."""
from pymatgen.core import Lattice, Structure

from agent.tools.mlip_relax import relax_and_screen


def test_relax_a_real_structure_produces_a_plausible_energy():
    structure = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.6), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    result = relax_and_screen(structure)

    assert result["formula"] == "NaCl"
    # a real ionic rocksalt sits somewhere in the single-digit-negative-eV/atom range,
    # not near zero (unbound) and not implausibly large (a numerically broken result)
    assert -10.0 < result["energy_per_atom"] < -1.0
    assert result["num_steps"] > 0
    print(f"PASSED: NaCl relaxed to {result['energy_per_atom']:.3f} eV/atom, converged={result['converged']}")


if __name__ == "__main__":
    test_relax_a_real_structure_produces_a_plausible_energy()
