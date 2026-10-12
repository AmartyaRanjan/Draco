"""Regression tests for retrieval settings validation (no heavy deps)."""
import unittest

from app.retrieval.settings import RetrievalSettings


def settings(**kw):
    base = dict(dense_top_k=30, bm25_top_k=30, rrf_k=60, dense_timeout_s=5, bm25_timeout_s=5,
                colbert_model="m", colbert_device="cpu", colbert_share_backbone=True,
                colbert_batch_size=2, colbert_max_candidates=60, colbert_max_length=512,
                score_threshold=0.5, calibration_file=None, min_selected=3, max_selected=20,
                max_context_tokens=6000, tokenizer=None, llm_base_url=None, llm_api_key=None,
                llm_model=None, llm_timeout_s=5, llm_max_output_tokens=100, llm_temperature=0.0,
                log_level="INFO", diagnostics_verbose=False)
    return RetrievalSettings(**{**base, **kw})


class SettingsValidationTests(unittest.TestCase):
    def test_colbert_max_length_must_be_positive(self):
        with self.assertRaises(ValueError):
            settings(colbert_max_length=0).validate()
        with self.assertRaises(ValueError):
            settings(colbert_max_length=-5).validate()
        # Sane values still pass.
        settings(colbert_max_length=1).validate()

    def test_existing_limits_still_enforced(self):
        with self.assertRaises(ValueError):
            settings(dense_top_k=0).validate()
        with self.assertRaises(ValueError):
            settings(rrf_k=0).validate()


if __name__ == "__main__":
    unittest.main()
