"""Builds the capability registry (the 21 logical capabilities) and the group index."""
from app.agent.capabilities import (artifact_creation, computation, context_management, file_analysis, knowledge,
                                    knowledge_management, validation, visualization, web_research)
from app.agent.registry import CapabilityRegistry

MODULES = (knowledge, web_research, knowledge_management, artifact_creation, visualization, file_analysis,
           computation, validation, context_management)

EXPECTED = ["search_knowledge_base", "get_document_metadata", "expand_document_context", "rewrite_search_query",
            "web_search", "fetch_webpage", "extract_web_evidence",
            "propose_knowledge_ingestion", "ingest_approved_source", "inspect_source_provenance",
            "generate_document", "generate_presentation", "generate_spreadsheet",
            "generate_chart_or_table", "generate_image", "compose_multimodal_response",
            "analyze_uploaded_file", "run_calculation", "validate_citations", "record_feedback", "summarize_evidence"]


def build_registry() -> CapabilityRegistry:
    reg = CapabilityRegistry()
    for m in MODULES:
        for cap in m.CAPS:
            reg.register(cap)
    missing = set(EXPECTED) - {c.name for c in reg.all()}
    if missing:
        raise RuntimeError(f"catalog incomplete: {sorted(missing)}")
    return reg
