import unittest
from dataclasses import replace

from app.agent.config import get_agent_settings
from app.core.http_settings import get_platform_settings


class ProductionSafetyTests(unittest.TestCase):
    def test_private_web_hosts_are_rejected_in_production(self):
        settings = get_agent_settings()
        unsafe = replace(settings, web_allow_private_hosts=True)
        with self.assertRaises(ValueError):
            unsafe.validate(production=True)

    def test_proxy_hops_require_proxy_cidrs(self):
        settings = get_platform_settings()
        with self.assertRaises(ValueError):
            replace(settings, trusted_proxy_hops=1, trusted_proxy_cidrs=()).validate()


if __name__ == "__main__":
    unittest.main()
