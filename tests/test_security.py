import os
import tempfile
import time
import unittest

from support_assistant.security import tokens
from support_assistant.security.apikeys import ApiKeyRegistry

SECRET = "x" * 40


class TokenTests(unittest.TestCase):
    def test_roundtrip(self):
        t = tokens.issue(SECRET, "sid1", "CMP-101", "customer", int(time.time()) + 60)
        d = tokens.verify(SECRET, t)
        self.assertEqual((d["sid"], d["tid"], d["role"]), ("sid1", "CMP-101", "customer"))

    def test_tamper_detected(self):
        t = tokens.issue(SECRET, "sid1", "CMP-101", "customer", int(time.time()) + 60)
        v, payload, sig = t.split(".")
        forged = tokens.issue(SECRET, "sid1", "CMP-999", "agent", int(time.time()) + 60).split(".")[1]
        with self.assertRaises(tokens.InvalidToken):
            tokens.verify(SECRET, f"{v}.{forged}.{sig}")

    def test_wrong_secret_and_expiry(self):
        t = tokens.issue(SECRET, "s", "CMP-101", "customer", int(time.time()) + 60)
        with self.assertRaises(tokens.InvalidToken):
            tokens.verify("y" * 40, t)
        old = tokens.issue(SECRET, "s", "CMP-101", "customer", int(time.time()) - 1)
        with self.assertRaises(tokens.InvalidToken):
            tokens.verify(SECRET, old)

    def test_garbage(self):
        for bad in ("", "a.b", "v1.x.y", "v2.a.b"):
            with self.assertRaises(tokens.InvalidToken):
                tokens.verify(SECRET, bad)


class ApiKeyTests(unittest.TestCase):
    def test_create_and_authenticate(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "keys.json")
            key = ApiKeyRegistry.create(path, "CMP-103", "agent")
            reg = ApiKeyRegistry.load(path)
            p = reg.authenticate(key)
            self.assertEqual((p.tenant_id, p.role), ("CMP-103", "agent"))
            self.assertIsNone(reg.authenticate("nope"))
            self.assertIsNone(reg.authenticate(None))
            with open(path) as fh:
                self.assertNotIn(key, fh.read())           # only hashes at rest


if __name__ == "__main__":
    unittest.main()
