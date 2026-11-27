"""Adapters translate external components into Draco contracts.

Each adapter is its own module and pulls in its optional dependency only when imported:
`retrieval`, `ingestion`, `model_provider`, `providers`, `langgraph`, `langgraph_checkpoint`.
Application-specific adapters live with the application (for example `app.draco_adapters` in draco-rag).
"""
