"""Structured logging and LangSmith tracing setup, called once at process start by
both the FastAPI app and the Streamlit UI.

LangSmith needs no code changes to the graph itself to trace every node and every LLM
call: LangChain and LangGraph auto-instrument when LANGCHAIN_TRACING_V2=true and
LANGCHAIN_API_KEY are set in the environment, this module's job is making sure that
happens consistently and logging whether it actually took effect, rather than silently
running untraced if the key is missing.
"""
import logging
import os
import sys


def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    # noisy third-party loggers that would otherwise drown out the agent's own trace
    for noisy in ("urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def configure_langsmith(project_name="battery-electrode-screening-agent"):
    logger = logging.getLogger("battery_agent.observability")
    api_key = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")

    if not api_key:
        logger.warning(
            "no LANGCHAIN_API_KEY / LANGSMITH_API_KEY set, running without LangSmith "
            "tracing, set one to see the agent's full run trace at smith.langchain.com"
        )
        return False

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", project_name)
    logger.info(f"LangSmith tracing enabled, project={project_name}")
    return True
