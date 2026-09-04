"""FastAPI backend wrapping the LangGraph agent.

Secure handling means: no API key (MP, LLM provider, LangSmith) is ever accepted from
a request, all of them come from server-side environment variables only, so a client
of this API can never exfiltrate or override the service's own credentials. An
optional bearer token (API_AUTH_TOKEN) locks the API itself down when set; left unset
it stays open, the right default for a local demo, not for a real deployment.
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
_graph = None  # built lazily, on first request, so importing this module never requires a live MP_API_KEY


def _get_graph():
    global _graph
    if _graph is None:
        llm = get_chat_model()
        _graph = build_graph(llm, search_electrodes, propose_substitutions, relax_and_screen, search_literature)
        logger.info("agent graph built")
    return _graph


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
    graph = _get_graph()
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
