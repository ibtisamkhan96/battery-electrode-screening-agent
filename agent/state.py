"""The shared state every node in the graph reads from and writes to.

Two fields use an accumulating reducer (Annotated + operator.add): `log`, since every
node appends one more line to the same running trace rather than replacing it, and
`literature`, since each pass through literature_grounder adds results for whichever
candidates are new that round without discarding what an earlier round already found.
Every other field is last-write-wins, the normal TypedDict default, since a new round
of candidates is meant to replace the previous round's rejected ones, not pile up
alongside them.
"""
import operator
from typing import Annotated, Optional, TypedDict


class Constraints(TypedDict, total=False):
    working_ion: Optional[str]
    application: Optional[str]           # "cathode" or "anode", informs which search this becomes
    elements_include: list
    elements_exclude: list
    average_voltage_min: Optional[float]
    average_voltage_max: Optional[float]
    capacity_grav_min: Optional[float]


class AgentState(TypedDict, total=False):
    raw_query: str
    constraints: Constraints

    mp_candidates: list            # dicts from ElectrodeCandidate, DFT-verified, from Materials Project
    proposed_candidates: list      # dicts from ProposedCandidate, not yet verified
    screened_candidates: list      # proposed candidates after MLIP relaxation, with a stability estimate
    literature: Annotated[dict, operator.or_]   # formula -> list of literature hit dicts

    iteration: int
    max_iterations: int
    critic_verdict: dict           # {"sufficient": bool, "reason": str}

    final_report: str
    log: Annotated[list, operator.add]
