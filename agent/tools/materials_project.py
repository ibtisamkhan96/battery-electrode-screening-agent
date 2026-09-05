"""Wraps Materials Project's electrode endpoint, the primary data source for this agent.

Verified directly against the installed mp-api client (emmet.core.electrode.InsertionElectrodeDoc)
before this was written: MP ships a dedicated electrode document with real, DFT-computed
electrochemical fields (working_ion, average_voltage, capacity_grav/vol, energy_grav/vol,
stability_charge/discharge, max_delta_volume), not just general composition/energy data. That
is why this agent queries it directly instead of reconstructing voltage curves by hand from
raw formation energies.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

from langsmith import traceable
from mp_api.client import MPRester


@dataclass
class ElectrodeCandidate:
    battery_id: str
    material_ids: list
    working_ion: str
    formula_charge: str
    formula_discharge: str
    average_voltage: Optional[float]
    capacity_grav: Optional[float]
    capacity_vol: Optional[float]
    energy_grav: Optional[float]
    stability_charge: Optional[float]
    stability_discharge: Optional[float]
    max_delta_volume: Optional[float]
    source: str = "materials_project"


def _get_api_key(api_key=None):
    api_key = api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        raise ValueError(
            "MP_API_KEY is required. Get one free at next-gen.materialsproject.org/api "
            "and set it as an environment variable or pass api_key= explicitly."
        )
    return api_key


@traceable(name="materials_project.search_electrodes", run_type="tool")
def search_electrodes(
    working_ion=None,
    elements=None,
    exclude_elements=None,
    average_voltage=None,
    capacity_grav=None,
    max_stability=0.05,
    num_candidates=10,
    api_key=None,
):
    """Searches Materials Project's real electrode dataset directly.

    working_ion: single element symbol, e.g. "Li", "Na", "Mg".
    elements / exclude_elements: constrain the electrode's chemical system.
    average_voltage / capacity_grav: (min, max) tuples, either side may be None.
    max_stability: caps stability_charge and stability_discharge (energy above hull,
        eV/atom) so both the charged and discharged phase are close to thermodynamically
        real, not just the framework in isolation, this is the field that actually
        tells you whether an electrode is physically plausible, not just on paper.
    """
    with MPRester(_get_api_key(api_key)) as mpr:
        # the registered attribute is "insertion_electrodes", not "electrodes", confirmed
        # by inspecting MaterialsRester._sub_resters directly after a live call raised
        # AttributeError on the name this was first written with; both names resolve to
        # the same ElectrodeRester/InsertionElectrodeDoc schema verified earlier
        docs = mpr.materials.insertion_electrodes.search(
            working_ion=working_ion,
            elements=elements,
            exclude_elements=exclude_elements,
            average_voltage=average_voltage,
            capacity_grav=capacity_grav,
            stability_charge=(0, max_stability),
            stability_discharge=(0, max_stability),
            num_chunks=1,
            chunk_size=num_candidates,
            fields=[
                "material_ids", "working_ion",
                "formula_charge", "formula_discharge",
                "average_voltage", "capacity_grav", "capacity_vol", "energy_grav",
                "stability_charge", "stability_discharge", "max_delta_volume",
            ],
        )

    candidates = []
    for doc in docs:
        material_ids = [str(m) for m in (doc.material_ids or [])]
        # material_ids, not the rester's nominal "battery_id", is the field actually
        # confirmed present on InsertionElectrodeDoc, so it is what builds the id here
        candidates.append(ElectrodeCandidate(
            battery_id="-".join(sorted(material_ids)) if material_ids else "",
            material_ids=material_ids,
            working_ion=str(doc.working_ion) if doc.working_ion else "",
            formula_charge=doc.formula_charge or "",
            formula_discharge=doc.formula_discharge or "",
            average_voltage=doc.average_voltage,
            capacity_grav=doc.capacity_grav,
            capacity_vol=doc.capacity_vol,
            energy_grav=doc.energy_grav,
            stability_charge=doc.stability_charge,
            stability_discharge=doc.stability_discharge,
            max_delta_volume=doc.max_delta_volume,
        ))
    return candidates


def search_summary_by_formula(formula, api_key=None):
    """Falls back to the general summary endpoint for a single proposed formula that
    has no entry in the electrode dataset yet, used to check whether Materials Project
    has computed *any* structure for a candidate the substitution tool proposed, even
    if it was never paired into a charge/discharge electrode entry."""
    with MPRester(_get_api_key(api_key)) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=["material_id", "formula_pretty", "energy_above_hull",
                    "formation_energy_per_atom", "band_gap", "is_metal"],
        )
    return docs
