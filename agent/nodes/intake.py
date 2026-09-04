"""Turns a plain-language target spec into the structured constraints every later
node actually queries against. This is the one node where an LLM does something a
rule-based parser genuinely could not: "a cathode that doesn't use cobalt" has to
become elements_exclude=["Co"], application="cathode", which needs real language
understanding, not keyword matching.
"""
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger("battery_agent.intake")

_SYSTEM_PROMPT = (
    "You extract structured battery electrode search constraints from a materials "
    "scientist's plain-language request. Only set fields the request actually implies, "
    "leave everything else null. working_ion should be a single element symbol (Li, Na, "
    "Mg, Ca, Zn, K). elements_exclude should list element symbols the user explicitly "
    "ruled out. Do not invent constraints the request did not state."
)


class ConstraintsOutput(BaseModel):
    working_ion: str | None = Field(None, description="single element symbol, e.g. 'Li'")
    application: str | None = Field(None, description="'cathode' or 'anode'")
    elements_include: list[str] = Field(default_factory=list)
    elements_exclude: list[str] = Field(default_factory=list)
    average_voltage_min: float | None = None
    average_voltage_max: float | None = None
    capacity_grav_min: float | None = None


def make_intake_node(llm):
    structured_llm = llm.with_structured_output(ConstraintsOutput)

    def intake(state):
        query = state["raw_query"]
        result = structured_llm.invoke([
            ("system", _SYSTEM_PROMPT),
            ("human", query),
        ])
        constraints = result.model_dump()
        logger.info(f"parsed constraints from query: {constraints}")
        return {
            "constraints": constraints,
            "iteration": 0,
            "max_iterations": 2,
            "log": [f"intake: parsed '{query}' into {constraints}"],
        }

    return intake
