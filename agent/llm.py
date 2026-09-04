"""Builds the chat model the LLM-driven nodes (intake, critic, report_writer) use.

Provider is chosen at runtime, not hardcoded, so the same graph runs on whichever key
is actually available. Every node takes its llm as a parameter rather than importing
this module directly, so tests can substitute a stub chat model with no network or key
involved at all, real dependency injection, not a convenience shortcut.
"""
import os


def get_chat_model(provider=None, model=None, temperature=0.0):
    provider = provider or os.environ.get("LLM_PROVIDER", "anthropic")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return ChatAnthropic(
            model=model or "claude-sonnet-4-5-20250929",
            temperature=temperature,
            api_key=api_key,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=temperature,
            api_key=api_key,
        )

    raise ValueError(f"unknown LLM_PROVIDER: {provider!r}, expected 'anthropic' or 'openai'")
