"""Literature grounding via arXiv's public API.

Semantic Scholar was the original choice for this, but its keyless tier shares one
rate-limited pool across every unauthenticated caller worldwide and returned HTTP 429
on repeated attempts during development, unreliable to build a live demo around.
arXiv's API has no such shared-pool problem and was confirmed working directly before
this was written, so it is the literature source this agent actually depends on.
"""
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import wraps

import requests
from langsmith import traceable

logger = logging.getLogger("battery_agent.arxiv")

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_API = "http://export.arxiv.org/api/query"


@dataclass
class LiteratureHit:
    title: str
    published: str
    arxiv_id: str
    url: str
    summary: str


def retry_with_backoff(max_attempts=3, base_delay=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except requests.RequestException as e:
                    if attempt == max_attempts:
                        raise
                    logger.warning(f"arxiv request attempt {attempt} failed ({e}), retrying in {delay}s")
                    time.sleep(delay)
                    delay *= 2
        return wrapper
    return decorator


def _build_query(composition, context_term, category):
    parts = [f"cat:{category}"] if category else []
    parts.append(f"all:{composition}")
    if context_term:
        parts.append(f"all:{context_term}")
    return " AND ".join(parts)


@retry_with_backoff(max_attempts=3, base_delay=1.0)
def _run_query(search_query, max_results):
    response = requests.get(
        _ARXIV_API,
        params={"search_query": search_query, "start": 0, "max_results": max_results},
        timeout=30,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)

    hits = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        title = entry.find("atom:title", _ATOM_NS).text.strip().replace("\n", " ")
        published = entry.find("atom:published", _ATOM_NS).text
        entry_id = entry.find("atom:id", _ATOM_NS).text
        summary = entry.find("atom:summary", _ATOM_NS).text.strip().replace("\n", " ")
        hits.append(LiteratureHit(
            title=title,
            published=published[:10],
            arxiv_id=entry_id.rsplit("/", 1)[-1],
            url=entry_id,
            summary=summary[:400],
        ))
    return hits


@traceable(name="arxiv.search_literature", run_type="tool")
def search_literature(composition, context_term="battery", max_results=5, category="cond-mat.mtrl-sci"):
    """Searches arXiv for papers relevant to one candidate composition.

    composition: the single anchor term the search is actually built around, e.g.
        "LiMnPO4". arXiv's search is a keyword match, not semantic, and ANDing every
        word of a free-text question together reliably returns zero hits, confirmed
        directly: "LiMnPO4 cathode stability" as one AND-of-three-terms query matched
        nothing, even though "LiMnPO4" alone matches five real, on-topic papers. One
        composition, one optional context word, is what actually works.
    context_term: a single extra word to narrow the results (default "battery").
        Left in place only if it does not empty the result set, if the narrowed
        search returns nothing this falls back to the plain composition search
        rather than silently returning zero hits from an over-specific query.
    category: an arXiv subject category ANDed against the search. Left unscoped, a
        term like "cathode" collides with an entirely different arXiv subfield,
        electron-emission cathodes in ion thrusters and plasma discharge, confirmed
        directly: the same composition query returned hollow-cathode plasma physics
        papers with no category filter, and genuine battery-materials papers with one.
    """
    if context_term:
        narrowed_query = _build_query(composition, context_term, category)
        hits = _run_query(narrowed_query, max_results)
        if hits:
            logger.info(f"arxiv search '{composition}'+'{context_term}' returned {len(hits)} hits")
            return hits
        logger.info(f"arxiv search '{composition}'+'{context_term}' returned nothing, "
                     f"falling back to '{composition}' alone")

    plain_query = _build_query(composition, None, category)
    hits = _run_query(plain_query, max_results)
    logger.info(f"arxiv search '{composition}' returned {len(hits)} hits")
    return hits
