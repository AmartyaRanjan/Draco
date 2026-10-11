
import os

import torch
from sentence_transformers import SentenceTransformer

from app.ingestion.schema import ChunkRecord


MODEL_ID = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "2"))
MAX_LENGTH = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))
EXPECTED_DIMENSIONS = int(
    os.getenv("EMBEDDING_DIMENSIONS", "1024")
)

_model = None


def get_embedding_model() -> SentenceTransformer:
    """Load the model lazily and reuse it within this process."""
    global _model

    if _model is None:
        model_kwargs = {}

        if DEVICE == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but is unavailable."
                )

            model_kwargs["torch_dtype"] = torch.float16

        _model = SentenceTransformer(
            MODEL_ID,
            device=DEVICE,
            model_kwargs=model_kwargs,
        )

        _model.max_seq_length = MAX_LENGTH

    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate normalized dense vectors for one bounded batch."""
    if not texts:
        return []

    model = get_embedding_model()

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if vectors.shape[1] != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSIONS} dimensions, "
            f"received {vectors.shape[1]}."
        )

    return vectors.tolist()


def embed_chunks(chunks: list[ChunkRecord]) -> list[list[float]]:
    return embed_texts([chunk.text for chunk in chunks])
