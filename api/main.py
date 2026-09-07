"""FastAPI backend wrapping the LangGraph agent.

The LLM key is bring-your-own: each request supplies its own provider and API key
(QueryRequest.provider / .api_key), used to build that one request's own graph and
never persisted, logged, or reused for any other request. A single deployment of
this API can otherwise serve many callers from one shared process, and baking any
one caller's key into a module-level global (the way a naive cache would) risks
reusing it for someone else's query, or billing this service's owner for traffic
that isn't theirs. MP_API_KEY and LangSmith's key remain server-side environment
variables, since they are shared, low-cost API keys rather than per-token billed
ones. An optional bearer token (API_AUTH_TOKEN) locks the API itself down when set;
left unset it stays open, the right default for a local demo, not for a real
deployment.
"""
import logging
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.graph import build_graph
from agent.llm import get_chat_model
from agent.tools.arxiv_search import search_literature
from agent.tools.materials_project import search_electrodes
from agent.tools.mlip_relax import relax_and_screen
from agent.tools.substitution import propose_substitutions
from api.jobs import JobStore
from api.schemas import JobStatus, QueryRequest, QueryResponse
from observability.logging_config import configure_langsmith, configure_logging

configure_logging()
configure_langsmith()
logger = logging.getLogger("battery_agent.api")

app = FastAPI(
    title="Battery Electrode Screening Agent",
    description="Screens candidate battery electrode materials against Materials "
                 "Project, a local ML interatomic potential, and arXiv literature.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_job_store = JobStore()


def _build_graph(provider, api_key):
    """Builds a fresh graph for one request. Not cached: caching by (provider, key)
    would mean holding every visitor's key in server memory for the process
    lifetime, and caching without the key in it would reuse whichever caller's key
    built the graph first for every subsequent caller. Graph construction is cheap,
    in-memory node wiring, no network calls, so building it per request costs
    nothing that matters next to the 30-90s the agent itself takes to run."""
    llm = get_chat_model(provider=provider, api_key=api_key)
    return build_graph(llm, search_electrodes, propose_substitutions, relax_and_screen, search_literature)


def _check_auth(authorization: str = Header(default=None)):
    required_token = os.environ.get("API_AUTH_TOKEN")
    if not required_token:
        return
    if authorization != f"Bearer {required_token}":
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_check_auth)])
def submit_query(request: QueryRequest):
    try:
        graph = _build_graph(request.provider, request.api_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    job_id = _job_store.create()
    _job_store.run_async(job_id, graph.invoke, {"raw_query": request.query})
    logger.info(f"submitted job {job_id}: {request.query!r}")
    return QueryResponse(job_id=job_id, status="pending")


@app.get("/query/{job_id}", response_model=JobStatus, dependencies=[Depends(_check_auth)])
def get_query(job_id: str):
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return JobStatus(job_id=job_id, **job)
