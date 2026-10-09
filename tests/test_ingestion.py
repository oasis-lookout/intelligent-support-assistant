import unittest

from support_assistant.ingestion.chunker import chunk_ticket, parse_turns, window_turns
from support_assistant.ingestion.redact import redact
from support_assistant.models import INTERNAL, PUBLIC, Ticket


class RedactTests(unittest.TestCase):
    def test_pii_removed(self):
        t = redact("Call 9876543210 or a.b@x.com, card 4111 1111 1111 1111, PAN ABCDE1234F, acct 123456789012")
        for bad in ("9876543210", "a.b@x.com", "4111", "ABCDE1234F", "123456789012"):
            self.assertNotIn(bad, t)

    def test_meaningful_numbers_kept(self):
        t = redact("Loan ending 8821, debited INR 2,40,000 on the 5th")
        self.assertIn("8821", t)
        self.assertIn("2,40,000", t)


class ChunkerTests(unittest.TestCase):
    T = Ticket("TKT-1", "CMP-101", "Customer: My account is blocked.\nAgent: Sorry, checking.\nCustomer: Please hurry.",
               "Resolved", "Rule FRD-017 triggered. Do NOT disclose.")

    def test_visibility_split(self):
        chunks = chunk_ticket(self.T)
        pub = [c for c in chunks if c.visibility == PUBLIC]
        internal = [c for c in chunks if c.visibility == INTERNAL]
        self.assertEqual(len(pub), 1)
        self.assertEqual(len(internal), 1)
        self.assertNotIn("FRD-017", pub[0].text)          # internal vocabulary never enters public chunks
        self.assertIn("FRD-017", internal[0].text)
        self.assertTrue(all(c.tenant_id == "CMP-101" for c in chunks))

    def test_parse_turns_continuation_lines(self):
        self.assertEqual(parse_turns("Customer: hi\nthere\nAgent: yo"), ["Customer: hi there", "Agent: yo"])

    def test_windowing_makes_progress_with_overlap(self):
        turns = [f"Customer: {'w ' * 30}"] * 6
        wins = window_turns(turns, max_words=70, overlap=1)
        self.assertGreater(len(wins), 1)
        self.assertEqual(wins[0][-1], wins[1][0])


if __name__ == "__main__":
    unittest.main()
