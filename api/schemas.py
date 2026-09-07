from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    # Bring-your-own-key: the LLM key is never a server-side secret, each caller
    # supplies their own so a shared deployment never bills or shares one caller's
    # credential with another. See the note at the top of api/main.py.
    provider: str = "anthropic"
    api_key: str


class QueryResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str          # "pending" | "running" | "completed" | "failed"
    log: list[str] = []
    result: Optional[dict] = None
    error: Optional[str] = None
