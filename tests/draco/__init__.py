import unittest


def load_tests(loader, standard_tests, pattern):
    """These are pytest tests (`pytest tests/draco`). The legacy `unittest discover -s tests` command
    skips this package instead of failing on imports."""
    return unittest.TestSuite()
