from __future__ import annotations

import unittest

import torch

from rnn.model import ModelCfg, SettleCharLM
from rnn.session import SettleSession
from rnn.vocab import Vocab


class SessionTraceTests(unittest.TestCase):
    def test_generate_with_trace_matches_deterministic_generate(self) -> None:
        torch.manual_seed(123)
        vocab = Vocab.from_text("T:A S:ABC")
        cfg = ModelCfg(d_model=8, n_layers=1, k_settle=1)
        model_a = SettleCharLM(vocab.size, cfg)
        model_b = SettleCharLM(vocab.size, cfg)
        model_b.load_state_dict(model_a.state_dict())

        sess_a = SettleSession(model_a, vocab, device=torch.device("cpu"))
        sess_b = SettleSession(model_b, vocab, device=torch.device("cpu"))
        for sess in (sess_a, sess_b):
            sess.reset()
            sess.ingest("T:A S:")

        plain = sess_a.generate(1, temperature=0.0)
        traced, trace = sess_b.generate_with_trace(1, expected=plain, temperature=0.0)

        self.assertEqual(traced, plain)
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["selected"], plain)
        self.assertEqual(trace[0]["predicted"], plain)
        self.assertEqual(trace[0]["expected"], plain)
        self.assertIsInstance(trace[0]["expected_logit"], float)
        self.assertIsInstance(trace[0]["predicted_logit"], float)
        self.assertIsInstance(trace[0]["margin"], float)
        self.assertGreaterEqual(len(trace[0]["topk"]), 1)


if __name__ == "__main__":
    unittest.main()
