import tempfile
import unittest

import numpy as np

from support_assistant.models import PUBLIC, Chunk
from support_assistant.stores.memory import InMemoryVectorStore


def _vec(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


class StoreTests(unittest.TestCase):
    def test_save_load_and_dim_guard(self):
        with tempfile.TemporaryDirectory() as d:
            s = InMemoryVectorStore(path=d, dim=3)
            ch = [Chunk("a:c0", "T1", "a", PUBLIC, "alpha"), Chunk("b:c0", "T2", "b", PUBLIC, "beta")]
            s.upsert(ch, np.stack([_vec(1, 0, 0), _vec(0, 1, 0)]))
            s.save()
            s2 = InMemoryVectorStore(path=d, dim=3)
            self.assertEqual(s2.count(), 2)
            self.assertEqual(s2.search("T1", _vec(1, 0, 0), 5, (PUBLIC,))[0].chunk.text, "alpha")
            self.assertEqual(s2.search("T1", _vec(0, 1, 0), 5, (PUBLIC,))[0].chunk.tenant_id, "T1")  # never T2
            with self.assertRaises(ValueError):
                InMemoryVectorStore(path=d, dim=4)

    def test_upsert_replaces_and_delete_tenant(self):
        s = InMemoryVectorStore(dim=2)
        c = Chunk("a:c0", "T1", "a", PUBLIC, "v1")
        s.upsert([c], np.stack([_vec(1, 0)]))
        s.upsert([Chunk("a:c0", "T1", "a", PUBLIC, "v2")], np.stack([_vec(1, 0)]))
        self.assertEqual(s.count("T1"), 1)
        self.assertEqual(s.search("T1", _vec(1, 0), 3, (PUBLIC,))[0].chunk.text, "v2")
        s.delete_tenant("T1")
        self.assertEqual(s.count(), 0)


if __name__ == "__main__":
    unittest.main()
