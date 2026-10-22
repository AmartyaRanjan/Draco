"""Adapters translate external components into Draco contracts. Nothing here is imported by the core.

Provider-neutral contracts (no optional dependencies): `model_provider`, `retrieval`.
LangGraph (extra `agents`): `langgraph`.
Starter-application integrations (need the `app` package and its stack): `app_*`, `conversation`, `errors`."""
from draco.adapters.app_agent import agent_runner_workload
from draco.adapters.app_ingestion import ingestion_workload
from draco.adapters.app_retrieval import (  # app RetrievalService; generic: adapters.retrieval
    rag_workload,
    retrieval_workload,
)
from draco.adapters.errors import map_app_error
