import json
import os
from dataclasses import dataclass
from functools import lru_cache


def _f(name, default):
    return float(os.getenv(name, default))


def _i(name, default):
    return int(os.getenv(name, default))


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    dense_top_k: int
    bm25_top_k: int
    rrf_k: int
    dense_timeout_s: float
    bm25_timeout_s: float
    colbert_model: str
    colbert_device: str
    colbert_share_backbone: bool
    colbert_batch_size: int
    colbert_max_candidates: int
    colbert_max_length: int
    score_threshold: float
    calibration_file: str | None
    min_selected: int
    max_selected: int
    max_context_tokens: int
    tokenizer: str | None
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str | None
    llm_timeout_s: float
    llm_max_output_tokens: int
    llm_temperature: float
    log_level: str
    diagnostics_verbose: bool

    def validate(self) -> "RetrievalSettings":
        if min(self.dense_top_k, self.bm25_top_k, self.colbert_batch_size,
               self.colbert_max_candidates, self.colbert_max_length,
               self.max_selected, self.llm_max_output_tokens) < 1:
            raise ValueError("Candidate counts, batch size and token limits must be >= 1.")
        if self.rrf_k < 1:
            raise ValueError("RETRIEVAL_RRF_K must be >= 1.")
        if self.min_selected < 0 or self.min_selected > self.max_selected:
            raise ValueError("RETRIEVAL_MIN_SELECTED must be in [0, RETRIEVAL_MAX_SELECTED].")
        if self.max_context_tokens < 256:
            raise ValueError("RETRIEVAL_MAX_CONTEXT_TOKENS must be >= 256.")
        if min(self.dense_timeout_s, self.bm25_timeout_s, self.llm_timeout_s) <= 0:
            raise ValueError("Timeouts must be positive.")
        if self.calibration_file and not 0 < self.score_threshold < 1:
            raise ValueError(
                "With a calibration file the threshold is a probability in (0, 1)."
            )
        return self


@lru_cache
def get_settings() -> RetrievalSettings:
    cal = os.getenv("RETRIEVAL_CALIBRATION_FILE") or None
    return RetrievalSettings(
        dense_top_k=_i("RETRIEVAL_DENSE_TOP_K", 30),
        bm25_top_k=_i("RETRIEVAL_BM25_TOP_K", 30),
        rrf_k=_i("RETRIEVAL_RRF_K", 60),
        dense_timeout_s=_f("RETRIEVAL_DENSE_TIMEOUT_S", 10),
        bm25_timeout_s=_f("RETRIEVAL_BM25_TIMEOUT_S", 10),
        colbert_model=os.getenv("COLBERT_MODEL", "BAAI/bge-m3"),
        colbert_device=os.getenv("COLBERT_DEVICE", os.getenv("EMBEDDING_DEVICE", "cuda")),
        colbert_share_backbone=os.getenv("COLBERT_SHARE_BACKBONE", "true").lower() == "true",
        colbert_batch_size=_i("COLBERT_BATCH_SIZE", 8),
        colbert_max_candidates=_i("COLBERT_MAX_CANDIDATES", 60),
        colbert_max_length=_i("COLBERT_MAX_LENGTH", 512),
        # Raw BGE-M3 ColBERT score (mean MaxSim, ~[0,1]); NOT a probability.
        # Provisional; set from labelled data. See RETRIEVAL_LOG.md.
        score_threshold=_f("RETRIEVAL_SCORE_THRESHOLD", 0.55 if not cal else 0.8),
        calibration_file=cal,
        min_selected=_i("RETRIEVAL_MIN_SELECTED", 3),
        max_selected=_i("RETRIEVAL_MAX_SELECTED", 20),
        max_context_tokens=_i("RETRIEVAL_MAX_CONTEXT_TOKENS", 6000),
        tokenizer=os.getenv("RETRIEVAL_TOKENIZER") or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        llm_base_url=os.getenv("LLM_BASE_URL") or None,
        llm_api_key=os.getenv("LLM_API_KEY") or None,
        llm_model=os.getenv("LLM_MODEL") or None,
        llm_timeout_s=_f("LLM_TIMEOUT_S", 60),
        llm_max_output_tokens=_i("LLM_MAX_OUTPUT_TOKENS", 800),
        llm_temperature=_f("LLM_TEMPERATURE", 0.0),
        log_level=os.getenv("RETRIEVAL_LOG_LEVEL", "INFO"),
        diagnostics_verbose=os.getenv("RETRIEVAL_DIAGNOSTICS_VERBOSE", "false").lower() == "true",
    ).validate()


def load_calibration(path: str) -> list[tuple[float, float]]:
    """Monotone (raw_score, probability) points, e.g. from isotonic regression."""
    with open(path) as fh:
        pts = sorted((float(s), float(p)) for s, p in json.load(fh))
    if len(pts) < 2 or any(b[1] < a[1] for a, b in zip(pts, pts[1:])):
        raise ValueError("Calibration points must be >= 2 and monotone non-decreasing.")
    return pts
