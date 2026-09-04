from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str          # "pending" | "running" | "completed" | "failed"
    log: list[str] = []
    result: Optional[dict] = None
    error: Optional[str] = None
