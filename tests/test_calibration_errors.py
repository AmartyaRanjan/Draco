"""Regression tests: calibration file errors must be explicit ValueErrors.

load_calibration() let missing files raise FileNotFoundError and malformed
payloads raise cryptic unpack/TypeErrors. Service startup then crashes with
an OSError instead of a clear config validation error. All load failures must
surface as ValueError with the path and expected shape in the message.
"""
import json
import tempfile
import unittest
from pathlib import Path

from app.retrieval.settings import load_calibration


class LoadCalibrationErrorTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write(self, name, content):
        path = self.dir / name
        path.write_text(content)
        return str(path)

    def test_missing_file_raises_value_error(self):
        missing = str(self.dir / "nope.json")
        with self.assertRaises(ValueError) as ctx:
            load_calibration(missing)
        self.assertIn("nope.json", str(ctx.exception))

    def test_invalid_json_raises_value_error(self):
        path = self.write("bad.json", "not json")
        with self.assertRaises(ValueError) as ctx:
            load_calibration(path)
        self.assertIn("bad.json", str(ctx.exception))

    def test_wrong_shape_raises_value_error(self):
        path = self.write("shape.json", json.dumps({"a": 1}))
        with self.assertRaises(ValueError) as ctx:
            load_calibration(path)
        self.assertIn("shape.json", str(ctx.exception))

    def test_non_numeric_values_raise_value_error(self):
        path = self.write("nums.json", json.dumps([["x", "y"], ["z", "w"]]))
        with self.assertRaises(ValueError):
            load_calibration(path)

    def test_valid_file_still_loads(self):
        path = self.write("ok.json", json.dumps([[0.0, 0.0], [1.0, 1.0]]))
        self.assertEqual(load_calibration(path), [(0.0, 0.0), (1.0, 1.0)])

    def test_monotone_check_still_enforced(self):
        path = self.write("mono.json", json.dumps([[0.0, 0.9], [1.0, 0.1]]))
        with self.assertRaises(ValueError):
            load_calibration(path)


if __name__ == "__main__":
    unittest.main()
