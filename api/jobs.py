"""A minimal in-memory job store: submit a query, get a job id back immediately, poll
for the result. A real multi-instance deployment would back this with Redis or a task
queue (Celery, RQ) instead of process memory, called out plainly here rather than
pretending an in-memory dict scales past a single-process demo deployment.
"""
import logging
import threading
import uuid

logger = logging.getLogger("battery_agent.api.jobs")


class JobStore:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def create(self):
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {"status": "pending", "log": [], "result": None, "error": None}
        return job_id

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def _update(self, job_id, **fields):
        with self._lock:
            self._jobs[job_id].update(fields)

    def run_async(self, job_id, graph_invoke_fn, initial_state):
        def _run():
            self._update(job_id, status="running")
            try:
                result = graph_invoke_fn(initial_state)
                self._update(job_id, status="completed", result=result, log=result.get("log", []))
                logger.info(f"job {job_id} completed")
            except Exception as e:
                logger.exception(f"job {job_id} failed")
                self._update(job_id, status="failed", error=str(e))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
