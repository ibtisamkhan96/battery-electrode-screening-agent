---
title: Battery Electrode Screening Agent
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Battery Electrode Screening Agent

A LangGraph multi-agent system that screens candidate battery electrode materials
against real Materials Project data, a local ML interatomic potential, and arXiv
literature, then writes a ranked, evidenced report. Full project, source, tests, and
architecture docs: https://github.com/ibtisamkhan96/battery-electrode-screening-agent

This Space runs the FastAPI backend and the Streamlit UI together in one container.
Set `MP_API_KEY`, `ANTHROPIC_API_KEY`, and optionally `LANGCHAIN_API_KEY` as Space
secrets (Settings > Repository secrets) before it will answer real queries.
