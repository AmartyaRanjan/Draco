import importlib.util
import os
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("opensearchpy"), "opensearch-py is not installed")
class OpenSearchSettingsTests(unittest.TestCase):
    def test_https_verifies_certificates_by_default(self):
        import app.db.opensearch as module

        seen = {}

        class FakeClient:
            def __init__(self, **kwargs):
                seen.update(kwargs)

        with patch.dict(os.environ, {"OPENSEARCH_URL": "https://search.example:9200"}, clear=False), \
                patch.object(module, "OpenSearch", FakeClient), \
                patch.object(module, "_client", None):
            os.environ.pop("OPENSEARCH_VERIFY_CERTS", None)
            module.get_opensearch_client()

        self.assertTrue(seen["use_ssl"])
        self.assertTrue(seen["verify_certs"])
        self.assertTrue(seen["ssl_assert_hostname"])

    def test_production_rejects_explicitly_disabled_verification(self):
        import app.db.opensearch as module

        with patch.dict(os.environ, {
            "APP_ENV": "production",
            "OPENSEARCH_URL": "https://search.example:9200",
            "OPENSEARCH_VERIFY_CERTS": "false",
        }, clear=False), patch.object(module, "_client", None):
            with self.assertRaises(ValueError):
                module.get_opensearch_client()


if __name__ == "__main__":
    unittest.main()
